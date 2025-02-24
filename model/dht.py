import torch.nn as nn

from model._cdht.dht_func import C_dht


class ConvAct(nn.Module):
    def __init__(self, dim_in, dim_out, dim_intermediate, ks=3, s=1):
        super(ConvAct, self).__init__()
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
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DHT_Layer(nn.Module):
    def __init__(self, input_dim, dim, numAngle, numRho):
        super(DHT_Layer, self).__init__()
        self.fist_conv = nn.Sequential(
            nn.Conv2d(input_dim, dim, 1), nn.BatchNorm2d(dim), nn.ReLU()
        )
        self.dht = DHT(numAngle=numAngle, numRho=numRho)
        self.convs = nn.Sequential(
            ConvAct(dim, dim, dim),
            ConvAct(dim, dim, dim),
        )

    def forward(self, x):
        x = self.fist_conv(x)
        x = self.dht(x)
        x = self.convs(x)
        return x


# import time
class DHT(nn.Module):
    def __init__(self, numAngle, numRho):
        super(DHT, self).__init__()
        self.line_agg = C_dht(numAngle, numRho)

    def forward(self, x):
        # start_time = time.perf_counter()
        accum = self.line_agg(x)  # Most time consuming part
        # end_time = time.perf_counter()
        # elapsed_time = end_time - start_time
        # print(f"DHT Elapsed time: {elapsed_time:.4f} seconds")
        return accum
