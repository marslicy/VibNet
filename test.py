import os
import time
from math import sqrt
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqDataset
from infer.inference import Inference


def test(
    model_path,
    data_path,
    split,
    config,
    batch_size,
    save_path,
):
    num_angle = config["model"]["num_angle"]
    num_rho = config["model"]["num_rho"]
    seq_length = config["model"]["seq_length"]
    size = config["data"]["size"]
    win = config["model"]["win"]
    stride = config["model"]["stride"]

    inference = Inference(model_path, num_angle, num_rho, seq_length, win, stride)
    dataset = SeqDataset(
        data_path=data_path,
        split=split,
        size=size,
        seq_length=seq_length,
        num_angle=num_angle,
        num_rho=num_rho,
        augment=False,
    )

    print(f"total: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    theta_diffs = torch.tensor([], device="cuda")
    rho_diffs = torch.tensor([], device="cuda")
    tip_diffs = torch.tensor([], device="cuda")
    time_list = []

    for batch in tqdm(dataloader):
        img, _, _, theta, rho, tip = batch
        theta = theta.to("cuda")
        rho = rho.to("cuda")
        tip = tip.to("cuda")
        with torch.no_grad():
            time_start = time.time()
            theta_pred, rho_pred, tip_pred, _, _, _ = inference(img)
            time_end = time.time()
            time_list.append(time_end - time_start)
            theta_diffs_curr = torch.abs(theta_pred - theta)
            rho_diffs_curr = (
                torch.abs(rho_pred - rho)
                * sqrt(size[0] ** 2 + size[1] ** 2)
                / num_rho
                * 50
                / size[0]
            )
            tip_diffs_curr = torch.norm(tip_pred - tip, dim=1) * 50 / size[0]
            theta_diffs = torch.cat([theta_diffs, theta_diffs_curr], dim=0)
            rho_diffs = torch.cat([rho_diffs, rho_diffs_curr], dim=0)
            tip_diffs = torch.cat([tip_diffs, tip_diffs_curr], dim=0)

    print(f"theta_diffs: mean {theta_diffs.mean()}, std {theta_diffs.std()}")
    print(f"rho_diffs: mean {rho_diffs.mean()}, std {rho_diffs.std()}")
    print(f"tip_diffs: mean {tip_diffs.mean()}, std {tip_diffs.std()}")
    print(
        f"time: mean {torch.tensor(time_list).mean()}, std {torch.tensor(time_list).std()}"
    )

    print("test done!!!")

    time_list = torch.tensor(time_list)
    os.makedirs(save_path, exist_ok=True)
    theta_diffs = theta_diffs.cpu().numpy()
    rho_diffs = rho_diffs.cpu().numpy()
    tip_diffs = tip_diffs.cpu().numpy()
    time_list = time_list.cpu().numpy()

    np.save(f"{save_path}/theta_diffs.npy", theta_diffs)
    np.save(f"{save_path}/rho_diffs.npy", rho_diffs)
    np.save(f"{save_path}/tip_diffs.npy", tip_diffs)


if __name__ == "__main__":
    model_path = "./logs/beef/model.pth"
    config_path = "./logs/beef/config.txt"
    
    batch_size = 4
    
    save_path = Path("./results")

    data_path_prefix = Path("./dataset")
    tissues = ["Beef", "Pork"]
    splits = ["challenging", "normal"]

    with open(config_path, "r") as f:
        config = eval(f.read())

    save_path_prefix = save_path / config["expriment_name"]

    exp_name = config["expriment_name"]
    for t in tissues:
        for split in splits:
            data_path = data_path_prefix / t
            save_path = str(save_path_prefix / (t.lower() + "_" + split))
            os.makedirs(save_path, exist_ok=True)
            
            print(f"Testing {t} {split}...")
            test(
                model_path,
                data_path,
                split,
                config,
                batch_size,
                save_path,
            )
