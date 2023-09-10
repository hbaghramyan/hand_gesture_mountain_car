# Hand Gesture Controlled MountainCar-v0 (OpenAI-Gym)

A collaborative space to implement hand gesture control for OpenAI Gym's classic MountainCar-v0 environment using a Convolutional Neural Network (CNN).

## References

- [MountainCar Documentation](https://www.gymlibrary.dev/environments/classic_control/mountain_car/)

## Setup and Requirements

1. Create and activate a suitable conda environment named `mnt_car`:

    ```bash
    conda env create -f environment.yaml
    conda activate mnt_car
    ```

## Execution

To properly control the MountainCar using hand gestures, ensure you run both scripts simultaneously:

1. `hand_gesture_control.py` 
2. `src/mountain-car-control/mountain_car_play.py`

When `gesture-controll.py` starts recognizing your hand you can now run `mountain-car-play.py`. The game window
will now be displayed. Click on the window and then start moving your index finger to left or right.

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

