### Introduction
# This is image data of Natural Scenes around the world.

# This Data contains around 25k images of size 150x150 distributed under 6 categories.
# - 'buildings' -> 0,
# - 'forest' -> 1,
# - 'glacier' -> 2,
# - 'mountain' -> 3,
# - 'sea' -> 4,
# - 'street' -> 5 

# The Train, Test and Prediction data is separated in each zip files. There are around 14k images in Train, 3k in Test and 7k in Prediction.

### Importing necessary library

import numpy as np
import pandas as pd
import os
import time
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torchvision.transforms as tt
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from utils import *

random_seed = 42
torch.manual_seed(random_seed);

### Exploring the dataset

# dataloaders
train = ImageFolder("input\seg_test", transform = tt.Compose([
    tt.Resize(64),
    tt.RandomCrop(64),
    tt.ToTensor(),
]))

train_dl = DataLoader(train, 64, shuffle=True, num_workers=3, pin_memory=True)

def main():
    # We will normalize the image tensors by subtracting the mean and 
    # dividing by the standard deviation across each channel. As a result, 
    # the mean of the data across each channel is 0, and standard deviation is 1. 
    # Normalizing the data prevents the values from any one channel from 
    # disproportionately affecting the losses and gradients while training, 
    # simply by having a higher or wider range of values that others
    # mean, std = get_mean_std(train_dl)
    stats = get_mean_std(train_dl)
    train_transform = tt.Compose([
        tt.Resize(64),
        tt.RandomCrop(64),
        tt.RandomHorizontalFlip(),
        tt.ToTensor(),
        tt.Normalize(*stats,inplace=True)
    ])
    test_transform = tt.Compose([
        tt.Resize(64),
        tt.RandomCrop(64),
        tt.ToTensor(),
        tt.Normalize(*stats,inplace=True)
    ])

    train = ImageFolder("input\seg_train", transform = train_transform)
    test = ImageFolder("input\seg_test", transform = test_transform)

    # We will split our dataset into two parts:
    # - train_ds: for training the data.
    # - valid_ds: for testing our model accuracy.
    # (this will tell you how well your model will 
    # perform on dataset which model has never seen)

    # `Note: `  we will not use test data as the part 
    # of validation, this data will actually determine your model accuracy.

    val_size = int(len(train) * 0.2)
    train_size = len(train) - val_size

    train_ds, val_ds = random_split(train, [train_size, val_size])
    print(len(train_ds), len(val_ds))
    # PyTorch data loaders
    return train_ds, val_ds, test

# Next, we can create data loaders for retrieving images in batches.
# We'll use a relatively large batch size of 128 to utlize a larger 
# portion of the GPU RAM. You can try reducing the batch size & 
# restarting the kernel if you face an "out of memory" error.

if __name__ == '__main__':
    print(__name__)
    train_ds, val_ds, test = main()

    batch_size = 128

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_dl = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test, batch_size*2, num_workers=2, pin_memory=True)
    print("done")

    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
            
        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl: 
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)
else:
    print(__name__)

# # PyTorch data loaders
# batch_size = 128

# train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
# valid_dl = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)
# test_dl = DataLoader(test, batch_size*2, num_workers=2, pin_memory=True)

# # mean, std = get_mean_std(train_dl)
# # print(mean, std)
