# source
# https://www.kaggle.com/code/mihirpaghdal/intel-image-classification-with-pytorch

### Importing necessary library

import torch
import torchvision.transforms as tt
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from utils_funcs import *
from utils_classes import *
import time
from data_preprocessing import prepare_data


# ARG:
#   1. Why do we need the random_seed and torch.manual_seed(random_seed) lines?
#   2. What are stats for? 
# -----------------------------------------------------------------------------

random_seed = 42
torch.manual_seed(random_seed);

stats = ((0.4301, 0.4574, 0.4537), (0.2482, 0.2467, 0.2806))

def principal():
    
    # Let's take a look at some sample images from the training dataloader. 
    # To display the images, we'll need to denormalize the pixels values 
    # to bring them back into the range (0,1).

    # ARG Notes:
    #   * prepare_data(32): 
    train_dl, valid_dl, test_dl, no_of_classes = prepare_data(64)

    # save_path = "results/batch_images.png"
    # show_batch(train_dl, stats, save_path)
    # print("done")

    # Moving our data into gpu
    device = get_default_device()
    print(device)

    ## Building the model
    # We will extend `ImageClassificationBase` to develop the `ResNet9` 
    # model which consist of `Residual Blocks` after every two CNN layer

    model = to_device(ResNet9(3, no_of_classes), device)
    print(model)

    history = [evaluate(model, valid_dl)]
    print(history)

    epochs = 1
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    start_time = time.time()
    history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

    plot_losses(history, "results/loss_vs_epochs.png")

    plot_lrs(history, "results/lr_vs_batch_no.png")
    print("done")

if __name__ == '__main__':
    print(__name__)
    principal()
else:
    print(__name__)
