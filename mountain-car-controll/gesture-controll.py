#!/bin/env python3.9
# Title: gesture-controll.py
# Description:
#   Use of mediapipe and cv2 to perform hand landmark tracking.
#   Taking joints:
#       8: Index_Finger_TIP
#       5: Index_Finger_MCP
#       0: Wrist
#   To then compute the angle between them and determine the 
#   direction the index finger is currently pointing at. Using 
#   a 180 degree threshold to indicate left or right, the 
#   direction is shown in the screen and a keyboard signal simulating 
#   "LEFT KEY", "RIGHT KEY" or "DOWN KEY" will be sent to I/O. The
#   "DOWN KEY" signal is sent when an angle between 39 < a < 180
#   degrees is detected, which is the angle detected by closing your
#   fist.
#   
#   Most portions of the code were taking form here:
#   https://github.com/AslanDevbrat/gesture_VidGame
#
#   There's also a document describing the process
#   https://arxiv.org/abs/2204.11119
########################################################################
import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
from pynput.keyboard import Key, Controller

mp_drawing = mp.solutions.drawing_utils # Used to draw real-time visuals
mp_hands = mp.solutions.hands           # Used to track hand landmarks
joint_list = [[8,5,0]]                  # Landmark joint

def draw_finger_angles(image, results, joint_list):
    # Find the angle between the required landmark

    # Loop through hands
    for hand in results.multi_hand_landmarks:
        # Loop through joint sets
        for joint in joint_list:
            hl_x0 = hand.landmark[joint[0]].x
            hl_y0 = hand.landmark[joint[0]].y
            hl_x1 = hand.landmark[joint[1]].x
            hl_y1 = hand.landmark[joint[1]].y
            hl_x2 = hand.landmark[joint[2]].x
            hl_y2 = hand.landmark[joint[2]].y
            
            a = np.array([hl_x0, hl_y0])
            b = np.array([hl_x1, hl_y1])
            c = np.array([hl_x2, hl_y2])
            radians = np.arctan2(np.array(c[1] - b[1]),np.array(c[0] - b[0])) - np.arctan2(np.array(a[1] - b[1]), np.array(a[0] - b[0])) 
            angle = np.abs(radians * 180.0 / np.pi)
            cv2.putText(image,"AA-> " + str(round(angle, 2)), tuple(np.multiply(b, [640,480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    return image, angle

def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score,2))

            # Extract coordinates
            coords = tuple(np.multiply(
                            np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x,
                            hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                            [640, 480]).astype(int))
            output = text, coords

    return output

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Flip on horizontal
        image = cv2.flip(image, 1)
        # Set flag
        image.flags.writeable = False
        # Detections
        results = hands.process(image)
        # Set flag to true
        image.flags.writeable = True
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detections
        print("Detections: ", results)

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121,
                                          22, 76), thickness=2,
                                          circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250,
                                          44, 250), thickness=2,
                                          circle_radius=2),
                                          )
                # Render left right detection
                if get_label(num, hand, results):
                    text, coord = get_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255,255,255), 2, cv2.LINE_AA)

                # Draw angles to image from joint list
                image, angle= draw_finger_angles(image, results, joint_list)
                keyboard = Controller()
                if angle <= 180 and angle > 39:
                    print('if pressed, angle: ', angle)
                    keyboard.press(Key.right)
                    keyboard.release(Key.right)
                elif angle > 180:
                    print('else pressed, angle: ', angle)
                    keyboard.press(Key.left)
                    keyboard.release(Key.left)
                else:
                    print('down pressed')
                    keyboard.press(Key.down)
                    keyboard.release(Key.down)
                
                cv2.rectangle(image, (0,0), (355, 73), (214, 44, 53))
                cv2.putText(image, 'Direction', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1,
                            cv2.LINE_AA)
                cv2.putText(image, "Left" if angle > 180 else "Right", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2 ,(255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Hand tracking', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                
                
cap.release()
cv2.destroyAllWindows()
