import time

import cv2
import torch
import torch.nn as nn

from model.dht import DHT_Layer
from model.encoder import Encoder
from model.fic import FIC


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, dim_intermediate, ks=3, s=1):
        super(ResBlk, self).__init__()
        p = (ks - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_intermediate, ks, s, p, bias=None),
            nn.BatchNorm2d(dim_intermediate),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim_intermediate, dim_out, ks, s, p, bias=None),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return y + x


class VibNet(nn.Module):
    def __init__(
        self,
        num_angle,
        num_rho,
        enc_init=True,
        fic_init=True,
        seq_len=30,
        win=10,
        stride=5,
    ):
        super(VibNet, self).__init__()
        if enc_init:
            pretained_dict_path = "model/magnet_epoch12_loss7.28e-02.pth"
        else:
            pretained_dict_path = None
        self.encoder = Encoder(pretained_dict_path=pretained_dict_path)
        self.batch_norm = nn.BatchNorm2d(32)
        self.fusion1d_1 = nn.Conv1d(32, 24, 1)

        self.fic = FIC(win, stride, init=fic_init)

        t = (seq_len - win) // stride + 1

        self.fusion2d_1 = nn.Sequential(
            ResBlk(24, 24, 24),
            ResBlk(24, 24, 24),
        )

        self.fusion_stft = nn.Sequential(
            nn.Conv1d(t, 1, 1, 1),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )

        self.fusion1d_2 = nn.Sequential(
            nn.Conv1d(24, 16, 7, 1, 3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 12, 7, 1, 3),
            nn.BatchNorm1d(12),
            nn.ReLU(),
        )

        self.fusion2d_2 = nn.Sequential(
            ResBlk(12, 12, 12),
            ResBlk(12, 12, 12),
        )

        out_length = 2 * int(win / 2) * 12

        self.fm_conv = nn.Sequential(
            ResBlk(out_length, out_length, out_length),
            ResBlk(out_length, out_length, out_length),
        )

        self.dht_detector = DHT_Layer(
            out_length, out_length, numAngle=num_angle, numRho=num_rho
        )
        self.last_conv = nn.Sequential(nn.Conv2d(out_length, 2, 1))
        self.num_angle = num_angle
        self.num_rho = num_rho

        if enc_init:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        # ================== Encoder ==================
        (N, T, C, H, W) = x.shape
        x = x.reshape(N * T, C, H, W)
        # reapeat each image 3 times to form 3 channels
        x = torch.repeat_interleave(x, 3, dim=1)
        _, x = self.encoder(x)  # 32 channels (N*T, C, H, W)
        x = self.batch_norm(x)
        x = x.reshape(N, T, -1, x.shape[-2], x.shape[-1])  # (N, T, C, H, W)
        (N, T, C, H, W) = x.shape

        # =============== Channel Fusion ===============
        x = x.permute(0, 3, 4, 2, 1)  # (N, H, W, C, T)
        x = x.reshape(N * H * W, -1, T)
        x = self.fusion1d_1(x)

        # =============== STFT Module ===============
        x = self.fic(x)  # (N*H*W, 24, 10, 5)
        (_, _, F, t) = x.shape
        x = x.reshape(-1, F, t)
        x = x.permute(0, 2, 1)  # (N*H*W*24, 5, 10)

        # =============== STFT Fusion ===============
        x = self.fusion_stft(x)  # (N*H*W*24, 1, 10)
        x = x.reshape(N, H, W, -1, F)  # (N, H, W, C, F)

        # ========= Spatial & Channel Conv =========
        x = self.permute_conv2d(self.fusion2d_1, x)
        x = self.permute_conv1d(self.fusion1d_2, x)
        x = self.permute_conv2d(self.fusion2d_2, x)

        # =============== Concatenate & 2D Conv ===============
        (N, _, _, H, W) = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (N, C, F, H, W)
        x = x.reshape(N, -1, H, W)  # (N, 120, H, W)
        x = self.fm_conv(x)

        # ========= Deep Hough & Classification =========
        x = x.contiguous()
        x = self.dht_detector(x)
        x = self.last_conv(x)
        return x

    def permute_conv2d(self, conv2d_layer, x):
        # input in shape (N, H, W, C, F)
        (N, H, W, _, F) = x.shape
        x = x.permute(0, 4, 3, 1, 2)  # (N, F, C, H, W)
        x = x.reshape(N * F, -1, H, W)
        x = conv2d_layer(x)
        x = x.reshape(N, F, -1, x.shape[-2], x.shape[-1])  # (N, F, C, H, W)
        # output in shape (N, F, C, H, W)
        return x

    def permute_conv1d(self, conv1d_layer, x):
        # input in shape (N, F, C, H, W)
        (N, F, _, H, W) = x.shape
        x = x.permute(0, 3, 4, 2, 1)  # (N, H, W, C, F)
        x = x.reshape(N * H * W, -1, F)
        x = conv1d_layer(x)
        x = x.reshape(N, H, W, -1, F)  # (N, H, W, C, F)
        # output in shape (N, H, W, C, F)
        return x
