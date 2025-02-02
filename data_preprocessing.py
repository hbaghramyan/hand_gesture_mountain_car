import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from utils_funcs import get_default_device
from utils_classes import DeviceDataLoader

def prepare_data(batch_size, stats, train_path):
    """
    Prepare the data by applying the necessary transformations and 
    splitting it into training and validation datasets.
    """
    device = get_default_device()

    train_transform = tt.Compose([
        tt.Resize((64, 48)),
        tt.ToTensor(),
        tt.Normalize(*stats, inplace=True),
    ])

    # test_transform = tt.Compose([
    #     tt.Resize(64),
    #     tt.RandomCrop(64),
    #     tt.ToTensor(),
    #     tt.Normalize(*stats, inplace=True)
    # ])

    train = ImageFolder(train_path, 
                        transform=train_transform)
    # test = ImageFolder(r"G:\Meine Ablage\mountain_car\images\test", transform=test_transform)

    val_size = int(len(train) * 0.2)
    train_size = len(train) - val_size

    train_ds, val_ds = random_split(train, [train_size, val_size])

    train_dl = DeviceDataLoader(DataLoader(train_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True), device)
    valid_dl = DeviceDataLoader(DataLoader(val_ds, batch_size, num_workers=2, pin_memory=True), device)
    # test_dl = DeviceDataLoader(DataLoader(test, batch_size * 2, num_workers=2, pin_memory=True), device)

    no_of_classes = len(train.classes)

    return train_dl, valid_dl, no_of_classes
