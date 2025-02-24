import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset


def gaussian(num_theta, num_rho, center, sig):
    """
    Gaussian blurring for Hough space
    """
    # create nxn zeros
    y = np.linspace(0, num_theta - 1, num_theta)
    x = np.linspace(0, num_rho - 1, num_rho)
    x, y = np.meshgrid(x, y)
    x0 = center[1]
    y0 = center[0]
    res = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sig**2))

    return res


class SeqDataset(Dataset):
    """
    Image sequence + label of the last image in the sequence.
    Each video shared the same label.
    The dataset directory should be organized as follows:
    -- data_path
        |-- imgs
            |-- seq_1: images of a video
            |-- seq_2: images of another video
            |-- ...
        |-- annos: labels of all videos (seq_1.png, seq_2.png, ...)
    The file names of the sequnces and of their label images should be the same.

    Args:
        data_path (string or Path): Path to the dataset
        split (string): Split of the dataset, there should be a file named "{split}.txt" in the data_path
        size (tuple of int, optional): The size of the output images. If the size is different from the original image size, the images will be resized.
        num_angle (int, optional): Number of angles in the prediction. Defaults to 180.
        num_rho (int, optional): Number of rhos in the prediction. Defaults to 100.
        augment (bool, optional): Augmentation is required or not. Defaults to True.
            Augmentation includes:
            - horizontally flip of the images
            - Contrast and Brightness adjustment
            - Gaussian blur
    """

    def __init__(
        self,
        data_path,
        split,
        size,  # H, W
        seq_length=30,
        num_angle=180,
        num_rho=100,
        augment=True,
    ):
        super().__init__()

        self.num_angle = num_angle
        self.num_rho = num_rho
        self.size = size  # H, W
        self.resize = (size[1], size[0])  # W, H
        self.augment = augment

        self.data_path = Path(data_path)
        self.seq_length = seq_length

        self.img_path = self.data_path / "imgs"
        self.anno_path = self.data_path / "annos"

        self.seq_names = natsorted(
            open(Path(data_path) / f"{split}.txt").read().splitlines()
        )
        self.all_file_names = [
            natsorted(os.listdir(self.img_path / name)) for name in self.seq_names
        ]

        self.length_list = [
            len(os.listdir(self.img_path / name)) - self.seq_length + 1
            for name in self.seq_names
        ]

    def calc_coords(self, label):
        """
        calulate the coordinates of the beginning and the end of the needle line region.

        Args:
            label: Segmentation mask of the needle

        Returns:
            x0, y0, x1, y1: location of points in the image space.
                The origin is the midpoint of the image. The x axis is from left to the right, the y axis is from the top to the bottom.
        """
        H, W = self.size
        coords = np.argwhere(label)
        try:
            x0 = coords[:, 1].min()
            x1 = coords[:, 1].max()
            y0 = coords[coords[:, 1] == x0][:, 0].min()
            y1 = coords[coords[:, 1] == x1][:, 0].max()

            x0 -= W / 2
            x1 -= W / 2
            y0 -= H / 2
            y1 -= H / 2
        except ValueError:
            x0, y0, x1, y1 = 0, 0, 0, 0

        return x0, y0, x1, y1

    def calc_rho_theta(self, x0, y0, x1, y1):
        """
        calculate the rho and theta of the line.
        """
        # hough transform
        theta = np.arctan2(y1 - y0, x1 - x0) + np.pi / 2
        rho = x0 * np.cos(theta) + y0 * np.sin(theta)
        return theta, rho

    def line_shaft(self, theta, rho):
        """
        create the hough space label for the shaft

        Returns:
            hough_space_shaft, theta, rho:
                - hough_space_shaft: the hough space label for the shaft, which is a gaussian distribution
                - theta: the index of gt theta in the hough space
                - rho: the index of gt rho in the hough space
        """
        # rho is the distance from the line to the middle point of the image
        H, W = self.size
        # calculate resolution of rho and theta
        irho = np.sqrt(H * H + W * W) / self.num_rho
        itheta = np.pi / self.num_angle

        # rho can be a negative value, so we need to shift the index
        rho_idx = int(np.round(rho / irho)) + int((self.num_rho) / 2)
        theta_idx = int(np.round(theta / itheta))
        if theta_idx >= self.num_angle:
            theta_idx = self.num_angle - 1
        hough_space_shaft = gaussian(
            self.num_angle, self.num_rho, (theta_idx, rho_idx), sig=2
        )

        return hough_space_shaft, theta_idx, rho_idx

    def all_line_cross_tip(self, y, x):
        """
        create the hough space label for the tip. The tip is the intersection of all the lines.
        """
        H, W = self.size
        irho = np.sqrt(H * H + W * W) / self.num_rho

        hough_space_tip = np.zeros((self.num_angle, self.num_rho))
        for i in range(self.num_angle):
            theta = i * np.pi / self.num_angle
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho = int(np.round(rho / irho)) + int((self.num_rho) / 2)
            hough_space_tip[i] = gaussian(1, self.num_rho, (0, rho), sig=3)
        return hough_space_tip

    def process_label(self, label):
        """
        It will process the label (segmentation mask) to get the hough space label and the theta and rho of the line and the tip location.
        """
        # find the coordinates of the line
        x0, y0, x1, y1 = self.calc_coords(label)
        # H, W = self.size
        # cv2.line(img, (int(x0 + W / 2), int(y0 + H / 2)), (int(x1 + W / 2), int(y1 + H / 2)), 255, 2)
        # cv2.imwrite('coordscheck.jpg',img)

        # no line in the image
        if y0 == y1 and x0 == x1:
            return np.zeros((2, self.num_angle, self.num_rho)), 0, 0

        # calculate the rho and theta
        # rho is the distance from the line to the middle of the image
        theta, rho = self.calc_rho_theta(x0, y0, x1, y1)
        # cos = np.cos(theta)
        # sin = np.sin(theta)
        # x0 = cos * rho
        # y0 = sin * rho
        # x1 = int(x0 + 1000 * (-sin))
        # y1 = int(y0 + 1000 * cos)
        # x2 = int(x0 - 1000 * (-sin))
        # y2 = int(y0 - 1000 * cos)
        # cv2.line(img, (int(x1 + W / 2), int(y1 + H / 2)), (int(x2 + W / 2), int(y2 + H / 2)), 255, 2)
        # cv2.imwrite("houghlinescheck.jpg", img)

        # create the hough space label
        hough_space_label = np.zeros((2, self.num_angle, self.num_rho))
        hough_space_label[0], theta, rho = self.line_shaft(theta, rho)

        # sort (y0, x0) and (y1, x1) based on y
        if y0 > y1:
            hough_space_label[1] = self.all_line_cross_tip(y0, x0)
            tip = np.array([y0, x0])
        else:
            hough_space_label[1] = self.all_line_cross_tip(y1, x1)
            tip = np.array([y1, x1])

        # y0, x0, y1, x1 was calculated by seen the middle point of the image as the origin
        # tip location in the tensor space
        tip[0] += self.size[0] / 2
        tip[1] += self.size[1] / 2

        return hough_space_label, theta, rho, tip

    def aug(self, img_seq, label):
        """
        data augmentation
        """
        img_seq = np.array(img_seq).astype(np.int32)

        augseq = A.ReplayCompose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    p=1, brightness_limit=(0, 0.2), contrast_limit=(0, 0.2)
                ),
                A.GaussianBlur(sigma_limit=(0, 1)),
            ]
        )

        img_seq = img_seq.astype(np.uint8)
        img_seq_aug = []
        data = augseq(image=img_seq[0])
        img_seq_aug.append(data["image"])

        for i in range(1, len(img_seq) - 1):
            img = A.ReplayCompose.replay(data["replay"], image=img_seq[i])["image"]
            img_seq_aug.append(img)

        transformed = A.ReplayCompose.replay(
            data["replay"], image=img_seq[-1], mask=label
        )
        img_seq_aug.append(transformed["image"])
        label = transformed["mask"]

        # visualize the augmented label
        # cv2.imwrite("augcheck_img.jpg", img_seq_aug[-1])
        # cv2.imwrite("augcheck_label.jpg", label * 255)

        return img_seq_aug, label

    def __len__(self):
        return sum(self.length_list)

    def __getitem__(self, index):
        i = 0
        while self.length_list[i] <= index:
            index -= self.length_list[i]
            i += 1

        seq_file_names = self.all_file_names[i][index : index + self.seq_length]

        assert len(seq_file_names) == self.seq_length, "sequence length not match"

        img_seq = []
        for file_name in seq_file_names:
            img = cv2.imread(
                str(self.img_path / self.seq_names[i] / file_name), cv2.IMREAD_GRAYSCALE
            )
            img = cv2.resize(img, self.resize)
            img_seq.append(img)

        label = cv2.imread(
            str(self.anno_path / (self.seq_names[i] + ".png")), cv2.IMREAD_GRAYSCALE
        )
        label = cv2.resize(label, self.resize)

        if self.augment:
            img_seq, label = self.aug(img_seq, label)

        hough_space_label, theta, rho, tip = self.process_label(label)

        return (
            np.expand_dims(np.array(img_seq), 1).astype(np.float32) / 127.5 - 1.0,
            hough_space_label,
            label,
            theta,
            rho,
            tip,
        )


if __name__ == "__main__":
    # can used for test dataset and utils
    import torch
    from torch.utils.data import DataLoader

    from utils import (reverse_all_hough_space, reverse_max_hough_space,
                       vis_result)

    seq_dataset = SeqDataset(
        data_path="dataset/Beef",
        split="test",
        size=(657 // 2, 671 // 2),
        seq_length=30,
        num_angle=180,
        num_rho=100,
        augment=True,
    )
    # print(len(seq_dataset))
    seq_dataloader = DataLoader(seq_dataset, batch_size=2, shuffle=True)
    for batch in seq_dataloader:
        img, hough_space_label, label, theta, rho, tip = batch
        img = img[0][-1]
        # print(img.shape)
        hough_space_label_shaft = hough_space_label[0][0]
        hough_space_label_tip = hough_space_label[0][1]
        label = label[0]
        theta = theta[0]
        rho = rho[0]
        tip = tip[0]

        lines = reverse_all_hough_space(
            torch.zeros(img.shape[-2:]), hough_space_label_tip, 180, 100
        )
        seq_tip = vis_result(img, lines, label)
        W, H = img.shape[-2:]
        # find the max value's location in the lines, using pytorch
        tip_loc = torch.argmax(lines)
        x_pos = tip_loc / H
        y_pos = tip_loc % H
        # print(x_pos, y_pos)
        # print(tip)
        cv2.imwrite("seq_tip.png", seq_tip)
        line = reverse_max_hough_space(
            torch.zeros(img.shape[-2:]), hough_space_label_shaft, 180, 100
        )
        seq_shaft = vis_result(img, line, label)
        cv2.imwrite("seq_shaft.png", seq_shaft)
        break
