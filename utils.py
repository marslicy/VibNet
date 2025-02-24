import cv2
import numpy as np
import torch


def theta_rho_to_xy(img_shape, theta, rho, num_angle, num_rho):
    """
    Convert the theta and rho to the coordinates of the line (start point and end point of each line).

    Args:
        img_shape (tuple): The shape of the image (H, W), where the lines will be drawn.
        theta (torch.tensor): Thetas of the lines in the shape of (num_line,).
        rho (torch.tensor): Rhos of the lines in the shape of (num_line,).
        num_angle (int): the number of angles in the hough space.
        num_rho (int): the number of rhos in the hough space.

    Returns:
        torch.tensor: 2 tensors, the coordinates of the point in the shape of (num_line, 2).
    """
    # calculate resolution of rho and theta
    H, W = img_shape
    if theta.dim() == 0:
        theta = theta.unsqueeze(0)
    if rho.dim() == 0:
        rho = rho.unsqueeze(0)
    # theta_idx, rho_idx = theta, rho
    l = torch.sqrt(torch.tensor(H * H + W * W))
    irho = l / num_rho
    itheta = torch.pi / num_angle
    theta = theta * itheta
    # shift the index back
    rho = rho - int((num_rho) / 2)
    rho = rho * irho
    # calculate the coordinates of the line
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    # x, y in image coordinate
    x0 = cos * rho
    y0 = sin * rho

    steps = l / 2
    x1 = x0 + steps * (-sin) + W / 2
    y1 = y0 + steps * cos + H / 2
    x2 = x0 - steps * (-sin) + W / 2
    y2 = y0 - steps * cos + H / 2

    # shift to tensor coordinate
    p1 = torch.stack([y1, x1], dim=1)
    p2 = torch.stack([y2, x2], dim=1)

    return p1, p2


def draw_lines(img, p1, p2, weight, width=5):
    """
    Draw lines into an image. The number of weights should be the same as the number of lines.

    Args:
        img(torch.tensor): the input image, where to draw the lines in the shape of (H, W).
        p1(torch.tensor): the start point [x y] of the line in the shape of(B, 2).
        p2(torch.tensor): the end point [x y] of the line in the shape of of (B, 2).
        weight(torch.tensor): the weight of the line in the shape of of (B, ).
        width(int): the width of the line. The default value is 5, which is best values for num_rho = 100.
            the best value is calculated by: width = sqrt(H^2 + W^2) / num_rho
    Return:
        (torch.tensor): the image with containing the line.
    """

    H, W = img.shape
    step = int(torch.sqrt(torch.tensor(H * H + W * W)) + 1)
    # filter out all nan, line is not in the image
    mask = torch.isnan(p1).any(dim=1) | torch.isnan(p2).any(dim=1)
    p1 = p1[~mask]
    p2 = p2[~mask]
    weight = weight[~mask]

    dx = ((p2[:, 0] - p1[:, 0]) / step).unsqueeze(1)
    dy = ((p2[:, 1] - p1[:, 1]) / step).unsqueeze(1)
    new_x = torch.repeat_interleave(p1[:, 0], step).reshape(p1.shape[0], step)
    new_y = torch.repeat_interleave(p1[:, 1], step).reshape(p1.shape[0], step)
    weight = torch.repeat_interleave(weight, step).reshape(p1.shape[0], step)
    i = torch.arange(step, device=dx.device).unsqueeze(0)

    new_x += dx * i
    new_y += dy * i
    new_x = new_x.long()
    new_y = new_y.long()

    idx = torch.arange(p1.shape[0], device=img.device)
    idx = torch.repeat_interleave(idx, step).reshape(p1.shape[0], step)
    mask = (new_x >= 0) & (new_x < H) & (new_y >= 0) & (new_y < W)
    new_x = new_x[mask]
    new_y = new_y[mask]
    idx = idx[mask]
    weight = weight[mask]

    idx = torch.stack([idx, new_x, new_y], dim=0).long()
    weight = weight.to(torch.float16)  # for saving space

    img_temp = torch.zeros((p1.shape[0], H, W), dtype=weight.dtype, device=img.device)
    # print(img_temp.element_size() * img_temp.nelement() / 1024 / 1024)
    bound = torch.tensor(H - 1)
    for i in range(width):
        # not the best way to set the width, but it's ok
        img_temp[idx[0], torch.min(bound, idx[1] + i), idx[2]] = weight
    img += img_temp.sum(dim=0)
    return img


def reverse_max_hough_space(img, hough_space, num_angle, num_rho, width=5):
    """
    Reverse the line with highest value in the hough space to the image.

    Args:
        img (torch.tensor): the tensor image in the shape of (H, W), where the lines will be drawn.
        hough_space (torch.tensor): the hough space in the shape of (1, num_angle, num_rho) or (num_angle, num_rho).
        num_angle (int): the number of angles in the hough space.
        num_rho (int): the number of rhos in the hough space.
        width (int): the width of the line to be drawn.

    Returns:
        (torch.tensor): an image with a line drawn.
    """
    hough_space = torch.squeeze(hough_space)

    # find the index of the max value of the hough space
    max_loc = torch.argmax(hough_space)
    theta = max_loc // num_rho
    rho = max_loc % num_rho

    p1, p2 = theta_rho_to_xy(img.shape, theta, rho, num_angle, num_rho)

    img = draw_lines(img, p1, p2, torch.tensor([255], device=img.device), width=width)

    return img


def reverse_all_hough_space(
    img, hough_space, num_angle, num_rho, threshold=1e-3, width=5
):
    """
    Reverse the hough space (contains a lot of lines) to the image.

    Args:
        img (torch.tensor): the tensor image in the shape of (H, W), where the lines will be drawn.
        hough_space (torch.tensor): the hough space in the shape of (1, num_angle, num_rho) or (num_angle, num_rho).
        num_angle (int): the number of angles in the hough space.
        num_rho (int): the number of rhos in the hough space.
        thereshold (float): the threshold to filter the hough space.
        width (int): the width of the line to be drawn.

    Returns:
        (torch.tensor): an image with lines drawn.
    """
    hough_space = torch.squeeze(hough_space)
    hough_space[hough_space < threshold] = 0

    theta, rho = torch.nonzero(hough_space, as_tuple=True)

    if not theta.size(0):
        return img

    value = hough_space[theta, rho]

    p1, p2 = theta_rho_to_xy(img.shape, theta, rho, num_angle, num_rho)

    img = draw_lines(img, p1, p2, value, width=width)

    img = (img - img.min()) / (img.max() - img.min())

    img = img * 255

    return img


def vis_result(input_img, line, label=None):
    input_img = input_img.squeeze().cpu().numpy()
    input_img = (input_img + 1) * 127.5
    input_img = input_img.astype(np.uint8)
    line = line.cpu().numpy().astype(input_img.dtype)

    if label is not None:
        label = label.cpu().numpy().astype(input_img.dtype)
        res = cv2.addWeighted(input_img, 1, label * 255, 0.5, 0)
        res = cv2.addWeighted(res, 1, line, 0.8, 0)
    else:
        res = cv2.addWeighted(input_img, 1, line, 0.8, 0)

    return res
