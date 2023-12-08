"""
This script calculates the mean and standard deviation of images from a given dataset.
"""

# native imports
import multiprocessing

# Third-party imports
import torch
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader


def train_dl(train_path):
    """
    Loads the images from the provided path and returns a DataLoader.

    Args:
        train_path (str): Path to the directory containing training images.

    Returns:
        DataLoader: DataLoader object for the training dataset with applied transformations.
    """

    # applying transformations to images
    train_dset = ImageFolder(
        train_path,
        transform=tt.Compose(
            [
                tt.Resize((64, 48)),
                tt.ToTensor(),
            ]
        ),
    )
    # creating a DataLoader object with specified parameters
    loader = DataLoader(
        dataset=train_dset,
        batch_size=64,
        shuffle=True,
        num_workers=multiprocessing.cpu_count() // 2 - 1,
        pin_memory=True,
    )
    return loader


def main(train_path):
    """
    Computes the mean and standard deviation of images in the provided dataset.

    Args:
        train_path (str): Path to the directory containing training images.

    Returns:
        tuple: A tuple containing mean and standard deviation of the dataset.
    """
    sum_, squared_sum, batches = 0, 0, 0

    # Iterating over the DataLoader to compute statistics
    for data, _ in train_dl(train_path):
        sum_ += torch.mean(data, dim=([0, 2, 3]))
        squared_sum += torch.mean(data**2, dim=([0, 2, 3]))
        batches += 1

    # Calculating mean and standard deviation
    mean = sum_ / batches
    std = (squared_sum / batches - mean**2) ** 0.5

    return mean, std


if __name__ == "__main__":
    main(train_path)
