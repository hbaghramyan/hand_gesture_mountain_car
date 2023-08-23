# File Name: hand_gesture_control.py
#
# Description: This script instantiates the ResNet9 model defined in utils_classes.py
# and loads our latest checkpoint generated after training. The model is set to eval()
# mode to generate predictions on frames captured by the webcam using OpenCV.
# Based on the detected class a Keyboard signal is sent to emulate the desired behavior of
# mountain car action space.
#
# Class 0 (GO LEFT) sends a "left arrow" key signal
# Class 1 (DONT MOVE) sends a "down arrow" key signal
# Class 2 (GO RIGHT) sends a "right arrow" key signal
#
# The sent signals to stdin are then read by src/mountain_car_play.py to controll the
# mountain car and play the game.
# ---------------------------------------------------------------------------------------

from PIL import Image
import torch
import torchvision.transforms as tt
import cv2
from pynput.keyboard import Key, Controller
from enum import Enum
from typing import Tuple
from utils_funcs import get_default_device
from utils_classes import ResNet9

keyboard = Controller()

device = get_default_device()

# Define an enumeration for gesture classes
class Gesture(Enum):
    GO_LEFT = 0
    DONT_MOVE = 1
    GO_RIGHT = 2

class GestureRecognizer:
    def __init__(self, model_path: str, font, font_scale: float, font_color: Tuple[int, int, int], text_position: Tuple[int, int], thickness: int):
        """
        Initialize the GestureRecognizer.

        Args:
            model_path (str): Path to the pre-trained model checkpoint.
            font: OpenCV font type for displaying text.
            font_scale (float): Font scale for diplaying text.
            font_color (Tuple[int, int, int]): RGB color tuple for text.
            text_position (Tuple[int, int]): Position to display text in frame.
            thickness (int): Thickness of the displayed text.
        """
        self.model = ResNet9(3,3)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.font = font
        self.font_scale = font_scale
        self.font_color = font_color
        self.text_position = text_position
        self.thickness = thickness
        self.keyboard = keyboard
    
    def preprocess_frame(self, frame):
        """
        Preprocess the input frame.

        Args:
            frame: Input frame captured from the video source.

        Returns:
            torch.Tensor: Preprocessed input tensor for the model.
        """
        stats = ((0.4301, 0.4574, 0.4537), (0.2482, 0.2467, 0.2806))

        # Define the same set of transformations applied during training
        # to the captured frame.
        transform = tt.Compose([
            tt.Resize(64),
            tt.RandomCrop(64),
            tt.RandomHorizontalFlip(),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True),
        ])

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_image).unsqueeze(0)
        return input_tensor
    
    def recognize_gesture(self, frame):
        """
        Recognize the gesture from the input frame.

        Args:
            frame: Input frame captured from the video source.
        
        Returns:
            int: Predcited class index representing the recognized gesture.
        """
        input_tensor = self.preprocess_frame(frame)

        with torch.no_grad():
            output = self.model(input_tensor)
        
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
            self.keyboard.press(Key.left)
            self.keyboard.release(Key.left)
        elif gesture == Gesture.DONT_MOVE:
            self.keyboard.press(Key.down)
            self.keyboard.release(Key.down)
        elif gesture == Gesture.GO_RIGHT:
            self.keyboard.press(Key.right)
            self.keyboard.release(Key.right)


# Instantiate the GestureRecognizer
recognizer = GestureRecognizer(
    model_path='checkpoints/model_epoch_AR_21_08_2023.pth',
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.0,
    font_color=(0, 255, 0),
    text_position=(100, 200),
    thickness=2
)

# Open the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Recognize gesture and handle action
    class_index = recognizer.recognize_gesture(frame)
    recognizer.handle_gesture(Gesture(class_index))
    
    # Display recognized gesture on the frame
    cv2.putText(frame, str(Gesture(class_index)), recognizer.text_position, recognizer.font, recognizer.font_scale, recognizer.font_color, recognizer.thickness)
    cv2.imshow('Hand gesture control demo', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
