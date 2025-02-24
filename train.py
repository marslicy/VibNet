import os
import random
import time

import cv2
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataset import SeqDataset
from model.vibnet import VibNet
from utils import reverse_all_hough_space, reverse_max_hough_space, vis_result


def setup_seed(seed):
    # random package
    random.seed(seed)

    # torch package
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # numpy package
    np.random.seed(seed)

    # os
    os.environ["PYTHONHASHSEED"] = str(seed)


def modified_focal_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    loss = -(pos_loss + neg_loss).mean()
    return loss


def get_model_dataset_expname(config):
    win = config.get("model").get("win")
    stride = config.get("model").get("stride")
    enc_init = config.get("model").get("enc_init")
    fic_init = config.get("model").get("fic_init")

    model = VibNet(
        num_angle=config["model"]["num_angle"],
        num_rho=config["model"]["num_rho"],
        seq_len=config["model"]["seq_length"],
        win=win if win is not None else 10,
        stride=stride if stride is not None else 5,
        enc_init=enc_init if enc_init is not None else True,
        fic_init=fic_init if fic_init is not None else True,
    )

    if config.get("expriment_name") is None:
        expriment_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        expriment_name = config["expriment_name"] + time.strftime(
            "_(%Y-%m-%d-%H-%M-%S)", time.localtime()
        )

    dataset_train = SeqDataset(
        data_path=config["data"]["data_path"],
        split="train",
        size=config["data"]["size"],
        seq_length=config["model"]["seq_length"],
        num_angle=config["model"]["num_angle"],
        num_rho=config["model"]["num_rho"],
        augment=True,
    )

    dataset_val = SeqDataset(
        data_path=config["data"]["data_path"],
        split="val",
        size=config["data"]["size"],
        seq_length=config["model"]["seq_length"],
        num_angle=config["model"]["num_angle"],
        num_rho=config["model"]["num_rho"],
        augment=False,
    )

    return model, dataset_train, dataset_val, expriment_name


def train(config):
    model, dataset_train, dataset_val, expriment_name = get_model_dataset_expname(
        config
    )

    device = config["train"]["device"]

    log_path = f"logs/{expriment_name}"
    writer = SummaryWriter(log_path)
    figure_path = f"{log_path}/figures_val"
    os.makedirs(figure_path, exist_ok=True)

    val_every_n = config["train"]["val_every_n"]
    print_every_n = config["train"]["print_every_n"]
    early_stop_thres = config["train"]["early_stop"]

    # save config
    with open(f"{log_path}/config.txt", "w") as f:
        f.write(str(config))

    best_val_loss = 100
    early_stop_cnt = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    if config["model"]["FocalLoss"]:
        loss_fn = modified_focal_loss
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss_fn.to(device)

    train_loader = DataLoader(
        dataset_train,
        batch_size=config["train"]["batch_size_train"],
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=config["train"]["batch_size_val"],
        shuffle=True,
    )

    model.train()
    model.to(device)

    loss_shaft_curr, loss_tip_curr, loss_curr = 0, 0, 0
    loss_shaft_print, loss_tip_print, loss_print = 0, 0, 0
    for epoch in range(config["train"]["epoch"]):
        for i, (img, hough_space_label, _, _, _, _) in enumerate(train_loader):
            img = img.to(device)
            hough_space_label = hough_space_label.to(device)

            optimizer.zero_grad()

            pred = model(img)
            loss_shaft_curr = loss_fn(pred[:, 0, :, :], hough_space_label[:, 0, :, :])
            loss_shaft_print += loss_shaft_curr

            loss_tip_curr = loss_fn(pred[:, 1, :, :], hough_space_label[:, 1, :, :])
            loss_tip_print += loss_tip_curr

            loss_curr = (
                config["train"]["w_shaft"] * loss_shaft_curr
                + config["train"]["w_tip"] * loss_tip_curr
            )
            loss_print += loss_curr

            loss_curr.backward()
            optimizer.step()

            if (i + 1) % print_every_n == 0 or i == len(train_loader) - 1:
                loss_shaft_print /= print_every_n
                loss_tip_print /= print_every_n
                loss_print /= print_every_n
                print(
                    f"Epoch {epoch} | Iter {i} | Loss {loss_print} (Shaft {loss_shaft_print}, Tip {loss_tip_print})"
                )
                writer.add_scalar(
                    "loss/train", loss_print, epoch * len(train_loader) + i
                )
                writer.add_scalar(
                    "loss_shaft/train",
                    loss_shaft_print,
                    epoch * len(train_loader) + i,
                )
                writer.add_scalar(
                    "loss_tip/train",
                    loss_tip_print,
                    epoch * len(train_loader) + i,
                )

                loss_shaft_print, loss_tip_print, loss_print = 0, 0, 0

            # validation
            if (epoch > 0 or (i + 1) >= 3000) and (
                ((i + 1) % val_every_n == 0) or i == len(train_loader) - 1
            ):
                val_loss, val_loss_shaft, val_loss_tip = validate(
                    model, config, val_loader, loss_fn, epoch, i, figure_path
                )

                print("======================================================")
                print(
                    f"Epoch {epoch} | Iter {i} | Val Loss {val_loss} (Shaft {val_loss_shaft}, Tip {val_loss_tip})"
                )
                print("======================================================")
                writer.add_scalar("loss/val", val_loss, epoch * len(train_loader) + i)
                writer.add_scalar(
                    "loss_shaft/val",
                    val_loss_shaft,
                    epoch * len(train_loader) + i,
                )
                writer.add_scalar(
                    "loss_tip/val",
                    val_loss_tip,
                    epoch * len(train_loader) + i,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f"{log_path}/model.pth")
                    early_stop_cnt = 0
                else:
                    print("No improvement!!")
                    early_stop_cnt += 1
                    if early_stop_cnt >= early_stop_thres:
                        print("Early stop!!!")
                        return

                model.train()


def validate(model, config, val_loader, loss_fn, epoch, i, figure_path):
    device = config["train"]["device"]
    with torch.no_grad():
        model.eval()
        val_loss_shaft = 0
        val_loss_tip = 0
        k = 0
        for j, (img, hough_space_label, label, _, _, _) in enumerate(val_loader):
            img = img.to(device)
            hough_space_label = hough_space_label.to(device)
            pred = model(img)
            val_loss_shaft += loss_fn(pred[:, 0, :, :], hough_space_label[:, 0, :, :])
            val_loss_tip += loss_fn(pred[:, 1, :, :], hough_space_label[:, 1, :, :])

            # save
            if k < 10:
                for j in range(5):
                    try:
                        # visualize the shaft prediction
                        line = reverse_max_hough_space(
                            torch.zeros(img.shape[-2:], device=device),
                            pred[j][0],
                            num_angle=config["model"]["num_angle"],
                            num_rho=config["model"]["num_rho"],
                        )
                        img_shaft = vis_result(img[j][-1], line, label[j])
                        cv2.imwrite(
                            f"{figure_path}/{epoch}_{i}_shaft_{k}.jpg",
                            img_shaft,
                        )

                        # visualize the tip prediction
                        line = reverse_all_hough_space(
                            torch.zeros(img.shape[-2:], device=device),
                            pred[j][1].sigmoid(),
                            num_angle=config["model"]["num_angle"],
                            num_rho=config["model"]["num_rho"],
                        )
                        img_tip = vis_result(img[j][-1], line, label[j])
                        cv2.imwrite(
                            f"{figure_path}/{epoch}_{i}_tip_{k}.jpg",
                            img_tip,
                        )

                        k += 1
                    except IndexError:
                        pass
        val_loss = (
            config["train"]["w_shaft"] * val_loss_shaft
            + config["train"]["w_tip"] * val_loss_tip
        )
        val_loss /= len(val_loader)
        val_loss_tip /= len(val_loader)
        val_loss_shaft /= len(val_loader)

    return val_loss, val_loss_shaft, val_loss_tip


if __name__ == "__main__":
    from config import config_list

    setup_seed(42)
    for config in config_list:
        train(config)
