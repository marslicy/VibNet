import cv2
import torch

from model.vibnet import VibNet
from utils import reverse_all_hough_space


class Inference:
    def __init__(
        self, pth_path, num_angle, num_rho, sequence_length=30, win=10, stride=5
    ):
        self.num_angle = num_angle
        self.num_rho = num_rho
        model = VibNet(
            num_angle, num_rho, seq_len=sequence_length, win=win, stride=stride
        )
        model.load_state_dict(torch.load(pth_path))

        self.model = model
        self.model.eval()
        self.model.to("cuda")

    def __call__(self, imgs):
        imgs = imgs.to("cuda")
        with torch.no_grad():
            out = torch.sigmoid(self.model(imgs))
        hough_shaft = out[:, 0, :, :]
        hough_tip = out[:, 1, :, :]
        H, W = imgs.shape[-2:]
        theta, rho, heatmaps_shaft = self.cal_theta_rho_idx(H, W, hough_shaft)
        tip_loc, heatmaps_tip = self.cal_tip_location(H, W, hough_tip)
        return theta, rho, tip_loc, heatmaps_shaft, heatmaps_tip, out

    def cal_heatmaps(self, H, W, hough_space, threshold=1e-3):
        heatmaps = []
        for i in range(hough_space.size(0)):
            lines = reverse_all_hough_space(
                torch.zeros((H, W), device="cuda"),
                hough_space[i],
                self.num_angle,
                self.num_rho,
                threshold,
            )
            heatmaps.append(lines)
        return torch.stack(heatmaps)

    def cal_theta_rho_idx(self, H, W, hough_space, percent=0.999):

        theta_rho_pred = []
        heatmaps = []
        for i in range(hough_space.size(0)):
            max_indices = torch.nonzero(hough_space[i] == torch.max(hough_space[i]))
            if max_indices.size(0) == 1:
                max_index = max_indices
            else:
                max_index = max_indices[0].reshape(1, -1)
            theta_rho_pred.append(max_index)
            threshold = torch.quantile(hough_space[i], percent, interpolation="lower")
            heatmaps.append(self.cal_heatmaps(H, W, hough_space[i].unsqueeze(0), threshold))
            
        heatmaps = torch.stack(heatmaps, dim=0).squeeze(1)
        theta_rho_pred = torch.stack(theta_rho_pred, dim=0).squeeze(1)
        theta_pred = theta_rho_pred[:, 0]
        rho_pred = theta_rho_pred[:, 1]

        return theta_pred, rho_pred, heatmaps

    def cal_tip_location(self, H, W, hough_space, threshold=1e-3):
        res = []
        # threshold = torch.quantile(hough_space, 1 - percent, interpolation="lower")
        heatmaps = self.cal_heatmaps(H, W, hough_space, threshold)
        for i in range(hough_space.size(0)):
            lines = heatmaps[i]
            # find the max value's location in the lines, using pytorch
            tip_loc = torch.argmax(lines)
            # x, y in the tensor space
            x_pos = tip_loc // W
            y_pos = tip_loc % W
            res.append(torch.tensor([x_pos, y_pos], device="cuda"))
        return torch.stack(res), heatmaps


if __name__ == "__main__":
    from infer.iterator import ImageIterator

    image_dir = "./dataset/Beef/imgs/0"
    anno_path = "./dataset/Beef/annos/0.png"
    iterator = ImageIterator(
        image_dir, anno_path, (657 // 2, 671 // 2), 30, batch_size=1
    )

    infer = Inference("./logs/beef/model.pth", 180, 100)
    for seqs in iterator:
        print(seqs.shape)
        theta, rho, tip_loc, heatmaps_shaft, heatmaps_tip, out = infer(seqs)
        cv2.imwrite(
            "test.png",
            heatmaps_shaft[0].cpu().numpy()
            * 255
            / heatmaps_shaft[0].cpu().numpy().max(),
        )
        exit(0)
