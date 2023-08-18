# native imports
import os
import argparse

# third-party imports
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np


def denormalize(images, means, stds):
    """
    Denormalizes image tensors using given means and standard deviations.

    Args:
        images (torch.Tensor): Images tensor to be denormalized.
        means (torch.Tensor): Means used for denormalization.
        stds (torch.Tensor): Standard deviations used for denormalization.

    Returns:
        torch.Tensor: Denormalized images.
    """
    means = means.clone().detach()
    means = means.reshape(1, 3, 1, 1)
    stds = stds.clone().detach()
    stds = stds.reshape(1, 3, 1, 1)
    means = means.to(get_default_device())
    stds = stds.to(get_default_device())
    return images * stds + means


def show_batch(dl, stats, save_path):
    """
    Displays and saves a batch of images from the dataloader after denormalization.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader instance to fetch data from.
        stats (tuple): A tuple containing means and standard deviations for denormalization.
        save_path (str): Path where the figure of images should be saved.
    """
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        denorm_images = denormalize(images, *stats)
        denorm_images = denorm_images.cpu()
        ax.imshow(make_grid(denorm_images[:64], nrow=8).permute(1, 2, 0).clamp(0, 1))
        plt.savefig(save_path)  # Save the figure to a file
        plt.close(fig)  # Close the figure to free up memory
        break


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def accuracy(outputs, labels):
    """
    Calculates accuracy for the given batch of outputs and labels.

    Args:
        outputs (torch.Tensor): Model's predictions.
        labels (torch.Tensor): Actual labels.

    Returns:
        torch.Tensor: Accuracy value.
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# building the model functions


def conv_block(in_channels, out_channels, pool=False):
    """
    Returns a block of convolutional layers with optional pooling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        pool (bool, optional): Whether to include max pooling. Defaults to False.

    Returns:
        nn.Sequential: Convolutional block.
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model, val_loader):
    """
    Evaluates the model on the given validation loader.

    Args:
        model (nn.Module): Model to be evaluated.
        val_loader (torch.utils.data.DataLoader): DataLoader instance for validation data.

    Returns:
        dict: Dictionary containing validation loss and accuracy.
    """
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    """
    Retrieves the current learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer instance.

    Returns:
        float: Current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit_one_cycle(
    epochs,
    max_lr,
    model,
    train_loader,
    val_loader,
    weight_decay=0,
    grad_clip=None,
    opt_func=torch.optim.SGD,
    checkpoint_dir="checkpoints",
    top_k=3,
):
    """
    Trains a model using the one-cycle policy.

    Args:
        epochs (int): Number of epochs for training.
        max_lr (float): Maximum learning rate.
        model (nn.Module): Model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader instance for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader instance for validation data.
        weight_decay (float, optional): Weight decay parameter. Defaults to 0.
        grad_clip (float, optional): Gradient clipping threshold. Defaults to None.
        opt_func (torch.optim.Optimizer, optional): Optimizer class to be used. Defaults to torch.optim.SGD.
        checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to "checkpoints".
        top_k (int, optional): Unused parameter in the current function. Defaults to 3.

    Returns:
        list[dict]: List of dictionaries containing training results for each epoch.
    """
    torch.cuda.empty_cache()
    history = []

    # create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        model.epoch_end(epoch, result)
        history.append(result)

        # save model checkpoint for the current epoch
        checkpoint_file = f"{checkpoint_dir}/model_epoch.pth"
        torch.save(model.state_dict(), checkpoint_file)

    return history


def plot_losses(history, save_path):
    train_losses = [x.get("train_loss") for x in history]
    val_losses = [x["val_loss"] for x in history]
    plt.plot(train_losses, "-bx")
    plt.plot(val_losses, "-rx")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training", "Validation"])
    plt.title("Loss vs. No. of epochs")
    plt.savefig(save_path)  # Save the figure to a file
    plt.close()  # Close the figure to free up memory


def plot_lrs(history, save_path):
    lrs = np.concatenate([x.get("lrs", []) for x in history])
    plt.plot(lrs)
    plt.xlabel("Batch no.")
    plt.ylabel("Learning rate")
    plt.title("Learning Rate vs. Batch no.")
    plt.savefig(save_path)  # Save the figure to a file
    plt.close()  # Close the figure to free up memory


def create_parser():
    """
    Create a parser for command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hyper_params_config_path",
        help="path to hyper-parameters config file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--batch_images_path",
        help="path to save batch of images",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--loss_image_path",
        help="path to save loss plot",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--lr_image_path",
        help="path to save learning rate plot",
        type=str,
        required=True,
    )

    return parser
