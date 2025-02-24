# nativ dependencies
from enum import Enum
import os
import gym

# third-party dependencies
from PIL import Image
import torch
import torchvision.transforms as tt
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pynput.keyboard import Key, Controller

# local dependencies
from utils_funcs import to_device, accuracy, conv_block

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

# contoller to send keyboard signals
keyboard = Controller()


class Car:
    def __init__(self, env: gym.Env):
        """
        Initialize the Car class.

        Args:
           env: MountainCar-v0 environment
        """
        # Action size = 3
        # Go left, Go right, Don't move
        self.action_size = env.action_space.n
        print(f"MountainCAr-v0 action size: {self.action_size}")

        # Define actions in a key map
        self.KEY_MAPPING = {
            pygame.K_LEFT: 0,  # Accelerating to the left: Action(0)
            pygame.K_RIGHT: 2,  # Accelrating to the right: Action(2)
            pygame.K_DOWN: 1,  # No acceleration: Action(1)
        }

    def get_action(self, pressed_key: int):
        """
        Map pressed key to a corresponding action.

        Args:
           pressed_key: Key code of the pressed key

        Returns:
           int: Action corresponding to the pressed key
        """
        # Set default behavior to DOWN button. This way the mountain car won't move if
        # no signal is detected.
        return self.KEY_MAPPING.get(pressed_key, self.KEY_MAPPING[pygame.K_DOWN])


class DeviceDataLoader:
    """
    Utility class to move batches of data to a desired device.
    """

    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for batch in self.data_loader:
            yield to_device(batch, self.device)

    def __len__(self):
        """
        Returns the number of batches in the DataLoader.
        """
        return len(self.data_loader)


class ImageClassificationBase(nn.Module):
    """
    Base class for image classification tasks. Provides methods
    for training and validation steps.
    """

    def training_step(self, batch):
        """
        Compute the loss for a batch of training data.

        Args:
            batch (tuple): A tuple containing images and their respective labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        """
        Compute the loss and accuracy for a batch of validation data.

        Args:
            batch (tuple): A tuple containing images and their respective labels.

        Returns:
            dict: A dictionary containing the computed validation loss and accuracy.
        """
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        """
        Compute the average validation loss and accuracy over an epoch.

        Args:
            outputs (list): List of dictionaries containing
            individual batch losses and accuracies.

        Returns:
            dict: A dictionary containing the average validation loss
            and accuracy for the epoch.
        """
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        """
        Print the results at the end of an epoch.

        """
        print(
            f"Epoch [{epoch}], last_lr: {result['lrs'][-1]:.5f}, "
            f"train_loss: {result['train_loss']:.4f}, "
            f"val_loss: {result['val_loss']:.4f}, "
            f"val_acc: {result['val_acc']:.4f}"
        )


class ResNet9(ImageClassificationBase):
    """
    A simplified implementation of the ResNet-9 architecture for image classification.
    """

    def __init__(self, num_classes: int, in_channels: int):
        """
        Initialize the ResNet9 model.

        Args:
            num_classes (int): Number of output classes.
            in_channels (int): Number of input channels.
        """
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, batch):
        """
        Forward pass of the ResNet9 model.

        Args:
            batch (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        out = self.conv1(batch)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


class Gesture(Enum):
    GO_LEFT = 0
    DONT_MOVE = 1
    GO_RIGHT = 2


class GestureRecognizer:
    def __init__(
        self,
        model_path: str,
    ):
        """
        Initialize the GestureRecognizer.

        Args:
            model_path (str): Path to the pre-trained model checkpoint.
        """
        self.model_path = model_path

    def _preprocess_frame(self, frame, stats: torch.Tensor):
        """
        Preprocess the input frame.

        Args:
            frame: Input frame captured from the video source.

        Returns:
            stats (torch.Tensor): Preprocessed input tensor for the model.
        """

        # Define the same set of transformations applied during training
        # to the captured frame.
        transform = tt.Compose(
            [
                tt.Resize((64, 48)),
                tt.ToTensor(),
                tt.Normalize(*stats, inplace=True),
            ]
        )

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_image).unsqueeze(0)
        return input_tensor

    def recognize_gesture(self, frame, stats, model):
        """
        Recognize the gesture from the input frame.

        Args:
            frame: Input frame captured from the video source.

        Returns:
            int: Predcited class index representing the recognized gesture.
        """
        input_tensor = self._preprocess_frame(frame, stats)

        with torch.no_grad():
            output = model(input_tensor)

        _, predicted = torch.max(output, 1)

        class_index = predicted.item()

        return class_index

    def handle_gesture(self, gesture):
        """
        Perform action based on the recognized gesture.

        Args:
            gesture (Gesture): Recognized gesture class.
        """
        if gesture == Gesture.GO_LEFT:
            keyboard.press(Key.left)
            keyboard.release(Key.left)
        elif gesture == Gesture.DONT_MOVE:
            keyboard.press(Key.down)
            keyboard.release(Key.down)
        elif gesture == Gesture.GO_RIGHT:
            keyboard.press(Key.right)
            keyboard.release(Key.right)
