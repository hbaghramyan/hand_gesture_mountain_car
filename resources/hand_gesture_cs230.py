# Making a PyTorch Dataset
# source https://cs230.stanford.edu/blog/handsigns/
# prerequisits 
# install pytorch https://pytorch.org/get-started/locally/ 
# I had no gpu and used pip so for me it was pip3 install torch torchvision torchaudio

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as transforms
from torch.autograd import Variable

# paths
train_data_path = r"dset\train"
val_data_path = r"dset\valid"

# params
hyperparams = {}
hyperparams["batch_size"] = 4
hyperparams["num_workers"] = 2

class SIGNSDataset(Dataset):
    def __init__(self, data_dir, transform):      
        #store filenames
        self.filenames = list(set(os.listdir(data_dir)) - {'desktop.ini'})
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames]

        #the first character of the filename contains the label
        self.labels = [int(filename.split(os.sep)[-1][0]) for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        #return size of dataset
        return len(self.filenames)
    
    def __getitem__(self, idx):
        #open image, apply transforms and return with label
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]

train_transformer = transforms.Compose([
    transforms.Resize(64),              # resize the image to 64x64 
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.ToTensor()])             # transform it into a PyTorch Tensor

eval_transformer = transforms.Compose([
    transforms.Resize(64),              # resize the image to 64x64
    transforms.ToTensor()])             # transform it into a PyTorch Tensor

train_dataset = SIGNSDataset(train_data_path, train_transformer)
val_dataset = SIGNSDataset(val_data_path, eval_transformer)
# test_dataset = SIGNSDataset(test_data_path, eval_transformer)

# Loading Batches of Data

train_dataloader = DataLoader(SIGNSDataset(train_data_path, train_transformer), 
                   batch_size=hyperparams["batch_size"], shuffle=True,
                   num_workers=hyperparams["num_workers"])

for train_batch, labels_batch in train_dataloader:
    # wrap Tensors in Variables
    train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

    # pass through model, perform backpropagation and updates
    output_batch = model(train_batch)
