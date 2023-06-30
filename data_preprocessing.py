import torchvision.transforms as tt
from pathlib import Path
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from utils_funcs import get_default_device
from utils_classes import DeviceDataLoader
import os

def prepare_data(batch_size):
    """
    Prepare the data by applying the necessary transformations and 
    splitting it into training and validation datasets.
    """
    device = get_default_device()
    stats = ((0.4301, 0.4574, 0.4537), (0.2482, 0.2467, 0.2806))

    train_transform = tt.Compose([
        tt.Resize(64),
        tt.RandomCrop(64),
        tt.RandomHorizontalFlip(),
        tt.ToTensor(),
        tt.Normalize(*stats, inplace=True)
    ])

    test_transform = tt.Compose([
        tt.Resize(64),
        tt.RandomCrop(64),
        tt.ToTensor(),
        tt.Normalize(*stats, inplace=True)
    ])

    train = ImageFolder(os.path.join("images", "train"), transform=train_transform)
    test = ImageFolder(os.path.join("images", "test"), transform=test_transform)

    val_size = int(len(train) * 0.2)
    train_size = len(train) - val_size

    train_ds, val_ds = random_split(train, [train_size, val_size])

    train_dl = DeviceDataLoader(DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True), device)
    valid_dl = DeviceDataLoader(DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True), device)
    test_dl = DeviceDataLoader(DataLoader(test, batch_size * 2, num_workers=2, pin_memory=True), device)

    no_of_classes = len(train.classes)

    return train_dl, valid_dl, test_dl, no_of_classes
