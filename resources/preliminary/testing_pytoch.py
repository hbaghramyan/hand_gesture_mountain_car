# source
# https://www.kaggle.com/code/mihirpaghdal/intel-image-classification-with-pytorch

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
from utils_funcs import *
import multiprocessing

random_seed = 42
torch.manual_seed(random_seed);

### Exploring the dataset

# train dataset
train_dset = ImageFolder("input\seg_train", transform = tt.Compose([
    tt.Resize(64), # This resizes the images to 64x64 pixels.
    tt.RandomCrop(64), # This randomly crops the images to size 64x64 pixels- I don't think this is a neccessary step
    tt.ToTensor(), # This converts the PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
    # to a PyTorch tensor (C x H x W) in the range [0.0, 1.0].
]))

train_dl = DataLoader(dataset = train_dset, batch_size = 64, shuffle=True, num_workers=multiprocessing.cpu_count() - 1, pin_memory=True)
# I have 16 cores, so I set num_workes=15
def main():

    stats = get_mean_std(train_dl)

    train_transform = tt.Compose([
        tt.Resize(64),
        tt.RandomCrop(64),
        tt.RandomHorizontalFlip(),
        tt.ToTensor(),
        tt.Normalize(*stats,inplace=True)
    ])

    # Why do we normalize test data on the parameters of the training data?
    # https://stats.stackexchange.com/questions/495357/why-do-we-normalize-test-data-on-the-parameters-of-the-training-data
    test_transform = tt.Compose([
        tt.Resize(64),
        tt.RandomCrop(64),
        tt.ToTensor(),
        tt.Normalize(*stats,inplace=True)
    ])

    # normalized and transformed datasets
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

        # Next, we can create data loaders for retrieving images in batches.
    # We'll use a relatively large batch size of 128 to utlize a larger 
    # portion of the GPU RAM. You can try reducing the batch size & 
    # restarting the kernel if you face an "out of memory" error.

    # batch_size = 128

    # train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # valid_dl = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)
    # test_dl = DataLoader(test, batch_size*2, num_workers=2, pin_memory=True)

    # class DeviceDataLoader():
    #     """Wrap a dataloader to move data to a device"""
    #     def __init__(self, dl, device):
    #         self.dl = dl
    #         self.device = device
            
    #     def __iter__(self):
    #         """Yield a batch of data after moving it to device"""
    #         for b in self.dl: 
    #             yield to_device(b, self.device)

    #     def __len__(self):
    #         """Number of batches"""
    #         return len(self.dl)
    
    # class ImageClassificationBase(nn.Module):
    #     def training_step(self, batch):
    #         images, labels = batch 
    #         out = self(images)                  # Generate predictions
    #         loss = F.cross_entropy(out, labels) # Calculate loss
    #         return loss
        
    #     def validation_step(self, batch):
    #         images, labels = batch 
    #         out = self(images)                    # Generate predictions
    #         loss = F.cross_entropy(out, labels)   # Calculate loss
    #         acc = accuracy(out, labels)           # Calculate accuracy
    #         return {'val_loss': loss.detach(), 'val_acc': acc}
            
    #     def validation_epoch_end(self, outputs):
    #         batch_losses = [x['val_loss'] for x in outputs]
    #         epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    #         batch_accs = [x['val_acc'] for x in outputs]
    #         epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    #         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
    #     def epoch_end(self, epoch, result):
    #         print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
    #             epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

    # # Moving our data into gpu
    # device = get_default_device()
    # print(device)

    # train_dl = DeviceDataLoader(train_dl, device)
    # valid_dl = DeviceDataLoader(valid_dl, device)
    # test_dl = DeviceDataLoader(test_dl, device)
    # print("done")

    # ### Building the model
    # # We will extend `ImageClassificationBase` to develop the `ResNet9` 
    # # model which consist of `Residual Blocks` after every two CNN layer

    # class ResNet9(ImageClassificationBase):
    #     def __init__(self, in_channels, num_classes):
    #         super().__init__()
            
    #         self.conv1 = conv_block(in_channels, 64)
    #         self.conv2 = conv_block(64, 128, pool=True)
    #         self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
            
    #         self.conv3 = conv_block(128, 256, pool=True)
    #         self.conv4 = conv_block(256, 512, pool=True)
    #         self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
            
    #         self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1), 
    #                                         nn.Flatten(), 
    #                                         nn.Dropout(0.2),
    #                                         nn.Linear(512, num_classes))
            
    #     def forward(self, xb):
    #         out = self.conv1(xb)
    #         out = self.conv2(out)
    #         out = self.res1(out) + out
    #         out = self.conv3(out)
    #         out = self.conv4(out)
    #         out = self.res2(out) + out
    #         out = self.classifier(out)
    #         return out

    # no_of_classes = len(train.classes)
    # print(no_of_classes)

    # model = to_device(ResNet9(3, no_of_classes), device)
    # model

    # history = [evaluate(model, valid_dl)]
    # print(history)

    # epochs = 12
    # max_lr = 0.01
    # grad_clip = 0.1
    # weight_decay = 1e-4
    # opt_func = torch.optim.Adam

    # start_time = time.time()
    # history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
    #                          grad_clip=grad_clip, 
    #                          weight_decay=weight_decay, 
    #                          opt_func=opt_func)
    # end_time = time.time()

    # elapsed_time = end_time - start_time
    # print("Elapsed time:", elapsed_time, "seconds")

if __name__ == '__main__':
    print(__name__)
    main()
else:
    print(__name__)
