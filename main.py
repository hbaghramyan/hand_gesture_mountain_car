# source
# https://www.kaggle.com/code/mihirpaghdal/intel-image-classification-with-pytorch

# This script provides an implementation of the image classification task using PyTorch. It uses traines a ResNet9
# It is structured to allow for the training, validation, and testing of the model.

# local function imports
import time
import torch
from omegaconf import OmegaConf

# local imports
from utils_funcs import (
    show_batch,
    get_default_device,
    to_device,
    evaluate,
    fit_one_cycle,
)
from utils_funcs import plot_losses, plot_lrs
from utils_funcs import create_parser
from utils_classes import ResNet9
from data_preprocessing import prepare_data
from get_stats import main as get_stats

# Set seed for generating random numbers for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)


def main():
    """
    main function that orchestrates the data loading, pre-processing, model training
    and validation.
    """

    # create command line parser and parse the command line arguments
    cmd_parser = create_parser()
    cmd_args = cmd_parser.parse_args()

    # get paths to config files and data from the command line arguments
    hyper_params_config_path = cmd_args.hyper_params_config_path
    batch_images_path = cmd_args.batch_images_path
    loss_image_path = cmd_args.loss_image_path
    lr_image_path = cmd_args.lr_image_path

    # load the hyper-parameters config file
    train_configs = OmegaConf.load(hyper_params_config_path)
    train_configs = OmegaConf.to_object(train_configs)

    # get stats for the data
    # training path
    train_path = train_configs["train_path"]
    stats = get_stats(train_path)
    # stats = (
    #     torch.Tensor([0.6537, 0.5984, 0.5382]),
    #     torch.Tensor([0.2901, 0.2970, 0.2958]),
    # )
    # prepare the data loaders
    train_dl, valid_dl, no_of_classes = prepare_data(train_configs["n_batch"], stats)

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
    epochs = train_configs["epochs"]
    max_lr = train_configs["max_lr"]
    grad_clip = train_configs["grad_clip"]
    weight_decay = train_configs["weight_decay"]
    opt_func = getattr(torch.optim, train_configs["opt_func"])

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
