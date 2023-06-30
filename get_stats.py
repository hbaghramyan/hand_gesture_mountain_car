import torch
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import multiprocessing

# train dataset
train_dset = ImageFolder(r"G:\Meine Ablage\mountain_car\images", transform = tt.Compose([
    tt.Resize(64), # This resizes the images to 64x64 pixels.
    tt.RandomCrop(64), # This randomly crops the images to size 64x64 pixels- I don't think this is a neccessary step
    tt.ToTensor(), # This converts the PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
    # to a PyTorch tensor (C x H x W) in the range [0.0, 1.0].
]))

train_dl = DataLoader(dataset = train_dset, batch_size = 64, shuffle=True, num_workers=multiprocessing.cpu_count() // 2 - 1, pin_memory=True)
# I have 16 cores, so I set num_workes=8-1=7

def main():

    sum_, squared_sum, batches = 0,0,0
    for data, _ in train_dl:
        sum_ += torch.mean(data, dim = ([0,2,3]))
        squared_sum += torch.mean(data**2, dim = ([0,2,3]))
        batches += 1
    mean = sum_/batches
    std = (squared_sum/batches - mean**2)**0.5
    print(mean, std)
    return mean, std

if __name__ == '__main__':
    main()