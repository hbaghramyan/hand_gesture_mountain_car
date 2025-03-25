# This script provides an implementation of the image classification task using PyTorch.
# It uses ResNet9 and is structured to allow for the training, validation,
# and testing of the model.

# native depndencies
import pickle
import time
import os

# third-party imports
import torch
from omegaconf import OmegaConf
from dotenv import load_dotenv


# local dependencies
from utils_funcs import (
    show_batch,
    get_default_device,
    to_device,
    evaluate,
    fit_one_cycle,
)
from utils_funcs import plot_losses, plot_lrs
from utils_classes import ResNet9
from data_preprocessing import prepare_data
from get_stats import main as get_stats

# setting seed reproducibility
torch.manual_seed(42)


def main():
    """
    main function that orchestrates the data loading, pre-processing, model training
    and validation.
    """

    load_dotenv()

    config_path = os.getenv("CONFIG_PATH", None)

    # loading the config file
    conf = OmegaConf.load(config_path)

    # get paths to config files and data from the command line arguments
    train_configs = conf.train
    batch_images_path = conf.paths.batches
    loss_image_path = conf.paths.loss
    lr_image_path = conf.paths.lr

    # get stats for the data
    # training path
    train_path = train_configs.train_path
    stats = get_stats(train_path)
    # stats = (
    #     torch.Tensor([0.6537, 0.5984, 0.5382]),
    #     torch.Tensor([0.2901, 0.2970, 0.2958]),
    # )
    # prepare the data loaders
    with open("stats_new.pkl", "wb") as file:
        pickle.dump(stats, file)

    train_dl, valid_dl, no_of_classes = prepare_data(
        train_configs.n_batch, stats, train_path
    )

    # save a batch of images from the training set
    show_batch(train_dl, stats, batch_images_path)

    # Get the default device (GPU if available else CPU)
    device = get_default_device()
    print(f"The device is {device}.")

    # create the model and move it to the device
    model = to_device(ResNet9(no_of_classes), device)

    # evaluate the model on the validation data
    history = [evaluate(model, valid_dl)]
    print(history)

    # extract the hyperparameters from the config
    epochs = train_configs.epochs
    max_lr = train_configs.max_lr
    grad_clip = train_configs.grad_clip
    weight_decay = train_configs.weight_decay
    opt_func = getattr(torch.optim, train_configs.opt_func)

    # record the start time
    start_time = time.time()
    # train the model
    history += fit_one_cycle(
        epochs,
        max_lr,
        model,
        train_dl,
        valid_dl,
        grad_clip=grad_clip,
        weight_decay=weight_decay,
        opt_func=opt_func,
    )
    # record the end time
    end_time = time.time()

    # compute and print the elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

    # plot and save the loss vs. epochs graph
    plot_losses(history, loss_image_path)

    # plot and save the learning rate vs. batch number graph
    plot_lrs(history, lr_image_path)


if __name__ == "__main__":
    main()
