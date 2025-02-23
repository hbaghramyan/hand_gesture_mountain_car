# Hand Gesture Controlled MountainCar-v0 (OpenAI-Gym)

A collaborative space to implement hand gesture control for OpenAI Gym's classic MountainCar-v0 environment using a Convolutional Neural Network (CNN).

## Demo

![MountainCar controll demo](https://github.com/hbaghramyan/hand_gesture_mountain_car/blob/dev_AR/demo.gif?raw=true)

## Setup and Requirements

Create and activate a suitable conda environment named `car`:

For macOS and Linux
```bash
wget -qO- https://astral.sh/uv/install.sh | sh 
uv sync
source .venv/bin/activate
```

For Windows
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync
.venv\Scripts\Activate.ps1
```

## Execution

To properly control the MountainCar using hand gestures, ensure you run `hand_gesture_control.py` and `mountain_car_play.py` scripts at the same time. When `gesture-controll.py` starts recognizing your hand you can now run `mountain-car-play.py`. The game window will now be displayed. Click on the window and then start moving your index finger to left or right.

### hand_gesture_control.py

This script initializes the ResNet9 model from `utils_classes.py` and loads our latest trained checkpoint. Once set in evaluation mode, it captures video frames in real-time and predicts hand gestures. Depending on the recognized gesture, a corresponding keyboard signal is sent:

- **Gesture Classes**:
  - Class 0 (GO LEFT): sends a "left arrow" key signal.
  - Class 1 (DONT MOVE): sends a "down arrow" key signal.
  - Class 2 (GO RIGHT): sends a "right arrow" key signal.

The signals are intercepted by `src/mountain_car_play.py` to control the MountainCar game.

### mountain_car_play.py

Allows you to play the MountainCar-v0 environment using the hand gestures recognized by the above script. When a recognized hand gesture corresponds to a control command (e.g., go left), the car in the environment will react accordingly.

**Note**: Make sure that both scripts are running simultaneously for the hand gesture control to work correctly with the MountainCar environment.

### Training 

To train your own gesture classifier run `src/train.py` and specify the training directory and all the other parameters in `configs/training.yaml`.

Directories with different gestures should be present in the training directory following the name convention for each class: `acc_l`, `acc_n` and `acc_r`. The checkpoints are being saved in the `checkpoints` directory following the date and time. Please make sure to have the same number of training examples per class.

### References

- [MountainCar Documentation](https://www.gymlibrary.dev/environments/classic_control/mountain_car/)