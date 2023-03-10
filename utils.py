import torch
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

def get_mean_std(dl):
    # We will normalize the image tensors by subtracting the mean and 
    # dividing by the standard deviation across each channel. As a result, 
    # the mean of the data across each channel is 0, and standard deviation is 1. 
    # Normalizing the data prevents the values from any one channel from 
    # disproportionately affecting the losses and gradients while training, 
    # simply by having a higher or wider range of values that others
    sum_, squared_sum, batches = 0,0,0
    for data, _ in dl:
        sum_ += torch.mean(data, dim = ([0,2,3]))
        squared_sum += torch.mean(data**2, dim = ([0,2,3]))
        batches += 1
    mean = sum_/batches
    std = (squared_sum/batches - mean**2)**0.5
    return mean, std

# We will define few function and classes to move data into gpu, 
# which will boost the training time to 10 times faster even more from cpu
# `Note:`You need to on the gpu accelerator. In the "Settings" section 
# of the sidebar, select "GPU" from the "Accelerator" dropdown. 
# Use the button on the top-right to open the sidebar.

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# def main():
#     # We will normalize the image tensors by subtracting the mean and 
#     # dividing by the standard deviation across each channel. As a result, 
#     # the mean of the data across each channel is 0, and standard deviation is 1. 
#     # Normalizing the data prevents the values from any one channel from 
#     # disproportionately affecting the losses and gradients while training, 
#     # simply by having a higher or wider range of values that others
#     # mean, std = get_mean_std(train_dl)
#     stats = get_mean_std(train_dl)
#     train_transform = tt.Compose([
#         tt.Resize(64),
#         tt.RandomCrop(64),
#         tt.RandomHorizontalFlip(),
#         tt.ToTensor(),
#         tt.Normalize(*stats,inplace=True)
#     ])
#     test_transform = tt.Compose([
#         tt.Resize(64),
#         tt.RandomCrop(64),
#         tt.ToTensor(),
#         tt.Normalize(*stats,inplace=True)
#     ])

#     train = ImageFolder("input\seg_train", transform = train_transform)
#     test = ImageFolder("input\seg_test",transform = test_transform)

#     # We will split our dataset into two parts:
#     # - train_ds: for training the data.
#     # - valid_ds: for testing our model accuracy.
#     # (this will tell you how well your model will 
#     # perform on dataset which model has never seen)

#     # `Note: `  we will not use test data as the part 
#     # of validation, this data will actually determine your model accuracy.

#     val_size = int(len(train) * 0.2)
#     train_size = len(train) - val_size

#     train_ds, val_ds = random_split(train, [train_size, val_size])
#     print(len(train_ds), len(val_ds))
#     return train_ds, val_ds, test