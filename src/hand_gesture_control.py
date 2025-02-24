# Description: This script instantiates a trained ResNet9 model for gesture recognition.
# and loads our latest checkpoint generated after training.
# The model generates predictions on frames captured by the webcam using OpenCV.
# Based on the detected class a keyboard signal is sent to emulate the desired
# behavior of mountain car action space.
#
# Class 0 (GO LEFT) sends a "left arrow" key signal
# Class 1 (DONT MOVE) sends a "down arrow" key signal
# Class 2 (GO RIGHT) sends a "right arrow" key signal
#
# The sent signals to stdin are then read by src/mountain_car_play.py to controll the
# mountain car and play the game.
# ---------------------------------------------------------------------------------------------------------------------

# local dependencies
from utils_classes import GestureRecognizer, Gesture, ResNet9

# native dependencies
import time
import pickle
import os
import ast

# third-party dependencies
from omegaconf import OmegaConf
from dotenv import load_dotenv
import cv2
import torch

load_dotenv()

config_path = os.getenv("CONFIG_PATH", None)

# loading the config file
conf = OmegaConf.load(config_path)
stats_path = conf.paths.stats
model_path = conf.paths.model

show = conf.show

# loading the pre-trained model
model = ResNet9(num_classes=conf.prep.n_classes, in_channels=conf.prep.n_channels)
model.load_state_dict(torch.load(model_path))
model.eval()

# get the saved stats
with open(stats_path, "rb") as file:
    stats_saved = pickle.load(file)

# instantiate the GestureRecognizer
recognizer = GestureRecognizer(model_path)

# the video capture
cap = cv2.VideoCapture(1)

# frame rate
fps = conf.prep.frame

# the frame interval in seconds
frame_interval = 1.0 / fps

while True:
    start_time = time.time()

    # capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # remaining time for the frame
    time_to_wait = frame_interval - (time.time() - start_time)

    if time_to_wait > 0:
        time.sleep(time_to_wait)

    # recognize gesture and handle action
    class_index = recognizer.recognize_gesture(frame, stats_saved, model)
    recognizer.handle_gesture(Gesture(class_index))

    # display recognized gesture on the frame
    cv2.putText(
        frame,
        str(Gesture(class_index)),
        ast.literal_eval(show.position),
        getattr(cv2, show.font),
        show.scale,
        ast.literal_eval(show.color),
        show.thickness,
    )
    cv2.imshow("Hand gesture control demo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
