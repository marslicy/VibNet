import math

import torch
import torch.nn as nn


class FIC(nn.Module):
    """
    Fourier Initialized Convolution
    For every 1D convolution, do fourier transform or short-time fourier transform
    Set window_size = sequence_length to do fourier transform
    (From UniTS: Short-Time Fourier Inspired Neural Networks for Sensory Time Series Classification, https://doi.org/10.1145/3485730.3485942)
    """

    def __init__(self, window_size, stride, init=True):
        super(FIC, self).__init__()
        self.window_size = window_size
        self.k = int(window_size / 2)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=2 * int(window_size / 2),
            kernel_size=window_size,
            stride=stride,
            padding=0,
            bias=False,
        )
        if init:
            self.init()

    def forward(self, x):
        # x: (batch, channel, in_length)
        B, C = x.shape[:2]

        # x: (batch, channel, in_length)
        x = x.reshape(B * C, 1, -1)
        # x: (batch*channel, 1, in_length)
        x = self.conv(x)
        # x: (batch*channel, fc, out_length)
        x = x.reshape(B, C, -1, x.shape[-1])
        # x: (batch, channel, fc, out_length)

        return x

    def init(self):
        """
        Fourier weights initialization
        """
        basis = torch.tensor(
            [math.pi * 2 * j / self.window_size for j in range(self.window_size)]
        )

        # print('basis size: ', basis.size())
        # print('basis: ', basis)

        weight = torch.zeros((self.k * 2, self.window_size))

        # print('weight size: ', weight.size())

        for i in range(self.k * 2):
            f = int(i / 2) + 1
            if i % 2 == 0:
                weight[i] = torch.cos(f * basis)
            else:
                weight[i] = torch.sin(-f * basis)

        self.conv.weight = torch.nn.Parameter(weight.unsqueeze(1), requires_grad=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    ft = FIC(60, 1)
    # cos in frequency 5hz
    # x = (
    #     torch.cos(torch.tensor([math.pi * 2 * 3.5 * j / 30 for j in range(30)]))
    #     .unsqueeze(0)
    #     .unsqueeze(0)
    # )

    x = (
        torch.cos(torch.tensor([math.pi * 2 * 7 * j / 59 for j in range(128)]))
        .unsqueeze(0)
        .unsqueeze(0)
    )

    print("x shape: ", x.shape)

    # # visualize the fourier transform
    # # plt.plot(x.squeeze())
    # # plt.savefig("cos.png")
    ft_res = ft(x).detach().numpy().squeeze()

    print(ft_res)

    print("result shape: ", ft_res.shape)

    # plt.plot(np.arange(0, 15, 0.5), ft_res)
    plt.imshow(ft_res[::2, :])
    # plt.imshow(ft_res)
    # print(ft_res.shape)
    plt.savefig("ft.png")
