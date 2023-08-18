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
#---------------------------------------------------------------------------------------

from pathlib import Path
from PIL import Image
from torchvision.datasets import ImageFolder
from utils_funcs import get_default_device
from utils_classes import DeviceDataLoader, ResNet9
import torch
import torchvision.transforms as tt
import cv2
from pynput.keyboard import Key, Controller

keyboard = Controller()

device = get_default_device()


# The stats must not be hard coded. Update this to get stats from
# Henrikh's script. 
stats = ((0.4301, 0.4574, 0.4537), (0.2482, 0.2467, 0.2806))

# 1. instantiate the ResNet9 class. 
# 2. Load checkpoint to the model.
# 3. Set model to eval state. 
model = ResNet9(3, 3)
model.load_state_dict(torch.load('checkpoints/model_epoch_HB_01_07_2023.pth'))
model.eval()

# Log to console the model structure and initialization
print(model)

# Set font parameters to display in frame
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
font_color = (0, 255, 0)
text_position = (100, 200)
thickness = 2

cap = cv2.VideoCapture(0) 

# Define the same set of transformations applied during training 
# to the captured frame. 
transform = tt.Compose([
    tt.Resize(64),
    tt.RandomCrop(64),
    tt.RandomHorizontalFlip(),
    tt.ToTensor(),
    tt.Normalize(*stats, inplace=True)
])

# class_names list with strings indicating the action to performed based on the prediction.
# For example: If the current gesture is the index finger, then the predicted class will be 2
# and "GO RIGHT" will be the current action performed by the MountainCar. 
class_names = ['GO LEFT', 'DONT MOVE', 'GO RIGHT']
while True:
    ret, frame = cap.read()

    # Preprocess the frame
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Apply previously defined transforms to the current frame
    input_tensor = transform(pil_image).unsqueeze(0)
    
    # Class names
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Interpret the predictions
    _, predicted = torch.max(output, 1)
    class_index = predicted.item()
    print(class_index, class_names[class_index])
    
    # Display recognized gesture on frame
    cv2.putText(frame, class_names[class_index], text_position, font, font_scale, thickness)
    cv2.imshow('Hand gesture controll demo...', frame)
    
    # Perform action based on the recognized gesture.
    if class_index == 0:
         keyboard.press(Key.left)
         keyboard.release(Key.left)
    if class_index == 1: 
         keyboard.press(Key.down)
         keyboard.release(Key.down)
    if class_index == 2:
         keyboard.press(Key.right)
         keyboard.release(Key.right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
