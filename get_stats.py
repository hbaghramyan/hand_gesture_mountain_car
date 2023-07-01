import torch
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import multiprocessing

# train dataset
train_dset = ImageFolder(r"G:\Meine Ablage\mountain_car\images\train", transform = tt.Compose([
    tt.Resize(64),
    tt.RandomCrop(64),
    tt.ToTensor(), 
]))

train_dl = DataLoader(dataset = train_dset, batch_size = 64, shuffle=True, num_workers=multiprocessing.cpu_count() // 2 - 1, pin_memory=True)

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