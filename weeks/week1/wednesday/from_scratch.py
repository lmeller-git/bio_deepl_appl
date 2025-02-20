from torch import nn
import torch
from torch import utils
import pandas as pd
from torchvision import transforms
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# dealing with images
from PIL import Image

# Pytorch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import torchmetrics

# Pytorch Lightning
import lightning as L

torch.manual_seed(42)


class MRIModel(nn.Module):
    def __init__(self, params: tuple[int]):
        super(MRIModel, self).__init__()
        shape_in = params["shape_in"]
        channels_out = params["initial_depth"]
        fc1_size = params["fc1_size"]
        conv1 = nn.Conv2d(shape_in[0], channels_out, kernel_size=8)
        current_data_shape = get_conv2d_out_shape(shape_in, conv1)
        conv2 = nn.Conv2d(
            current_data_shape[0], current_data_shape[0] * 2, kernel_size=4
        )
        current_data_shape2 = get_conv2d_out_shape(current_data_shape, self.conv2)
        self.conv_layer = nn.Sequential(
            [
                conv1,
                nn.MaxPool2d(2, 2),
                conv2,
            ]
        )
        self.lin_layer = nn.Sequential(
            [
                nn.Flatten(start_dim=1),
                nn.Linear(
                    current_data_shape2[0]
                    * current_data_shape2[1]
                    * current_data_shape2[2],
                    fc1_size,
                ),
                nn.ReLU(),
                nn.Linear(fc1_size, 2),
            ]
        )

    def forward(self, X):
        return F.log_softmax(self.lin_layer(self.conv_layer(X)), dim=1)


def main():
    data_path = "../data/tutorial_mri_data"
    labels = pd.read_csv(data_path + "/metadata.csv")
    image_dir = data_path + "/Brain Tumor Data Set/Brain Tumor Data Set"

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # from ImageNet
        ]
    )
    full_dataset = torchvision.datasets.ImageFolder(image_dir, transform=transform)

    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [0.7, 0.15, 0.15]
    )

    # define the dataloaders for train, validation and test (use shuffle for train only)
    batch_size_train = 256

    batch_size = 128  # for eval and test
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=16
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=16
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=16
    )
    batch = train_loader[0][0]
    cnn_model = MRIModel(
        params={"shape_in": batch[0].shape, "initial_depth": 4, "fc1_size": 128}
    )
    out = cnn_model(batch)
    model_parameters = {
        "shape_in": (3, 256, 256),  # size of our images
        "initial_depth": 4,
        "fc1_size": 128,
    }


def get_conv2d_out_shape(tensor_shape, conv, pool=2):
    # return the new shape of the tensor after a convolution and pooling

    # tensor_shape: (channels, height, width)

    # convolution arguments

    kernel_size = conv.kernel_size

    stride = conv.stride  # 2D array

    padding = conv.padding  # 2D array

    dilation = conv.dilation  # 2D array

    out_channels = conv.out_channels

    height_out = np.floor(
        (tensor_shape[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
        / stride[0]
        + 1
    )

    width_out = np.floor(
        (tensor_shape[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
        / stride[1]
        + 1
    )

    if pool:
        # adjust dimensions to pooling

        height_out /= pool

        width_out /= pool

    return int(out_channels), int(height_out), int(width_out)


if __name__ == "__main__":
    main()
