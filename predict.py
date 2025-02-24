import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from infer.inference import Inference
from infer.iterator import ImageIterator
from utils import theta_rho_to_xy


def vis_res_video(
    image_dir,
    anno_path,
    model_path,
    output_dir,
    batch_size,
    config,
):
    os.makedirs(output_dir, exist_ok=True)

    size = config["data"]["size"]
    seq_length = config["model"]["seq_length"]
    num_angle = config["model"]["num_angle"]
    num_rho = config["model"]["num_rho"]
    win = config["model"]["win"]
    stride = config["model"]["stride"]

    iterator = ImageIterator(image_dir, anno_path, size, seq_length, batch_size)
    inference = Inference(model_path, num_angle, num_rho, seq_length, win, stride)
    label = iterator.anno

    H, W = size
    # each frame contrains: img, heatmaps_shaft + label, heatmaps_tip + leabel + tip_loc
    frame = np.zeros((H, W * 3), dtype=np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_name = str(Path(output_dir) / Path(image_dir).name) + ".mp4"
    out = cv2.VideoWriter(vid_name, fourcc, 30.0, (W * 3, H), False)

    cnt = 0
    for seqs in tqdm(iterator, desc=f"Video {vid_name}"):
        theta, rho, tip_loc, heatmaps_shaft, heatmaps_tip, _ = inference(seqs)

        H, W = seqs.shape[-2:]
        p1, p2 = theta_rho_to_xy((H, W), theta, rho, num_angle, num_rho)
        p1 = p1.int().cpu().numpy()
        p2 = p2.int().cpu().numpy()
        tip_loc = tip_loc.int().cpu().numpy()
        heatmaps_shaft = heatmaps_shaft.cpu().numpy()
        heatmaps_tip = heatmaps_tip.cpu().numpy()

        for i in range(len(seqs)):
            img = (seqs[i, -1, :, :].cpu().numpy() + 1) * 127.5

            zeros = np.zeros((H, W), dtype=np.float32)
            shaft = cv2.addWeighted(zeros, 1, label.astype(np.float32) * 255, 0.5, 0)
            shaft = cv2.addWeighted(shaft, 1, heatmaps_shaft[i], 0.8, 0)
            shaft = cv2.line(shaft, (p1[i, 1], p1[i, 0]), (p2[i, 1], p2[i, 0]), 255, 1)

            zeros = np.zeros((H, W), dtype=np.float32)
            tip = cv2.addWeighted(zeros, 1, label.astype(np.float32) * 255, 0.5, 0)
            tip = cv2.addWeighted(tip, 1, heatmaps_tip[i], 0.8, 0)
            tip = cv2.circle(tip, (tip_loc[i, 1], tip_loc[i, 0]), 3, 255, -1)

            frame[:, :W] = img
            frame[:, W : W * 2] = shaft
            frame[:, W * 2 : W * 3] = tip

            out.write(frame)
            cnt += 1
    print(f"Video {vid_name} Done! Total {cnt} frames")
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    batch_size = 4
    dataset_dir = Path("./dataset/Beef")
    output_dir = "./output_videos_beef"
    model_path = "./logs/beef/model.pth"
    config_path = "./logs/beef/config.txt"

    seq_name = open("dataset/Beef/test.txt").read().split("\n")
    # seq_name = ["37", "41"]

    imgs_dir = dataset_dir / "imgs"
    annos_dir = dataset_dir / "annos"

    with open(config_path, "r") as f:
        config = eval(f.read())

    total = len(seq_name)
    for i, d in enumerate(seq_name):
        image_dir = str(imgs_dir / d)
        anno_path = str(annos_dir / f"{d}.png")

        vis_res_video(
            image_dir,
            anno_path,
            model_path,
            output_dir,
            batch_size,
            config,
        )

        print(f"Video {i+1}/{total} Done!")
