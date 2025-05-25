import torch
from dataset_loader import get_full_dataset_loader
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description="Compare loss values from two CSV files.")
parser.add_argument("--dataset_type", type=str, help="Dataset type", default='carla_001')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_mean_and_sd(loader):
    cnt = 0
    fst_moment = torch.empty(3).to(device)
    snd_moment = torch.empty(3).to(device)

    for images, _ in tqdm(loader):
        images = images.to(device)
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    return mean.cpu(), std.cpu()


if __name__ == '__main__':
    args = parser.parse_args()
    mean, std = batch_mean_and_sd(get_full_dataset_loader(dataset_type=args.dataset_type, train_only_center=False))

    print(f"mean {mean}, and std: {std}")
    # mean tensor([-0.1056,  0.0304, -0.0214]), and std: tensor([0.3843, 0.0327, 0.0475])  wit left, right, center