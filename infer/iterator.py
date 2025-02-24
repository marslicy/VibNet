from math import ceil
from pathlib import Path

import cv2
import numpy as np
import torch
from natsort import natsorted


class ImageIterator:
    def __init__(self, image_dir, anno_path, size, seq_len, batch_size=8):
        self.image_dir = Path(image_dir)
        self._index = 0
        self.file_list = self.image_dir.glob("*.png")
        self.file_list = [str(self.image_dir / f.name) for f in self.file_list]
        self.file_list = natsorted(self.file_list)
        self.length = len(self.file_list) - seq_len + 1
        self.size = size
        self.resize = (size[1], size[0])
        self.seq_len = seq_len
        if anno_path is not None:
            self.anno = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
            self.anno = cv2.resize(self.anno, self.resize)
        else:
            self.anno = None
        self.batch_size = batch_size

    def __len__(self):
        return ceil(self.length / self.batch_size)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < self.length:
            imgs = self.get(self._index)
            self._index += self.batch_size
            return imgs
        else:
            raise StopIteration

    def get(self, idx):
        if idx >= self.length:
            raise IndexError("index out of range")
        seqs = []
        imgs = []
        for i in range(self.seq_len):
            img = cv2.imread(self.file_list[idx + i], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.resize)
            img = img.reshape(1, *img.shape)
            imgs.append(img / 127.5 - 1)
        for i in range(self.batch_size):
            seqs.append(np.stack(imgs, axis=0))
            try:
                img = cv2.imread(
                    self.file_list[idx + i + self.seq_len], cv2.IMREAD_GRAYSCALE
                )
                img = cv2.resize(img, self.resize)
                img = img.reshape(1, *img.shape)
                imgs.append(img / 127.5 - 1)
                imgs.pop(0)
            except IndexError:
                break

        seqs = np.stack(seqs, axis=0)
        seqs = torch.tensor(seqs, dtype=torch.float32)
        return seqs


if __name__ == "__main__":
    image_dir = "./dataset/Beef/imgs/0"
    anno_path = "./dataset/Beef/annos/0.png"
    iterator = ImageIterator(image_dir, anno_path, (657 // 2, 671 // 2), 30)
    print(iterator.length)
    print(iterator.size)
    print(iterator.seq_len)
    print(iterator.anno.shape)
    for i, seqs in enumerate(iterator):
        print(i, seqs.shape)
