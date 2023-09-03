#!/usr/bin/env python3.9
# 
# Title: grab_images.py
# Description: Simple script to capture images to create our hand gesture dataset.
# 
# References:
# - https://www.geeksforgeeks.org/extract-video-frames-from-webcam-and-save-to-images-using-python/
# -------------------------------------------------------------------------------
# nativ imports
import os
import time
import datetime

# third party imports
import cv2

def menu():
    opts = {
            0 : 'Accelerate to the left',
            1 : 'Don\'t accelerate',
            2 : 'Accelerate to the right.',
            3 : 'Exit.'
        }
    for option in opts.keys():
        print(f'{option} -- {opts[option]}')

def captureIMGSet(option, no_imgs_to_capture, out_dir):
    # Get the current time
    now = datetime.datetime.now()
    
    # Format it as a string
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    acronyms = {
        0 : "acc_l", # Accelerate to the left
        1 : "acc_n",  # Don't accelerate
        2 : "acc_r"  # Accelerate to the right
        }

    # Opens the built camera of laptop to capture video.
    cap = cv2.VideoCapture(0)
    for i in range(no_imgs_to_capture):
        ret, frame = cap.read()
        
        # delay between frames
        time.sleep(0.3)

        # This condition prevents from infinite looping in case video ends.
        if ret == False:
            break

        # Save frame by frame into disk using imwrite method
        path = os.path.join(out_dir, f'{acronyms[option]}_{i}_{option}_{timestamp}.jpg')
        print(path)
        cv2.imwrite(path, frame)
        filename = os.path.basename(path)

        print(f'{filename} ... captured.')
    print('\n\nCapturing images...done.')
    print(f'{"-" :->82}')
    cap.release()
    cv2.destroyAllWindows()

def check_dirs(directory):
    print(f"type({directory}): {type(directory)}") 
    dir_check = os.path.isdir(directory)
    
    if not dir_check:
        os.makedirs(directory)
        print(f'dataset directory created: {directory}')
    else:
        print(f'{directory} already exists.')

if __name__ == '__main__':
    
    #data_dir = 'dataset'
    #dir_check = os.path.isdir(data_dir)

    #if not dir_check:
    #    os.makedirs(data_dir)
    #    print(f'dataset directory created: {data_dir}')
    #else:
    #    print(f'{data_dir} already exists.')
    
    while(True):
        menu()
        option = ''
        
        try:
            option = int(input('Select option from menu: '))
        except:
            print('Wrong option, select valid option.')
        
        

        if option == 3:
            exit()
        else:
            no_imgs_to_capture = int(input('How many images do you want to capture? \n'))
            print(f'{"-":->82}')
            if option == 0:
                data_dir  = 'try_dataset/'
                check_dirs(data_dir)
                #time.sleep(3)
                captureIMGSet(option, no_imgs_to_capture, data_dir)
            if option == 1:
                data_dir = 'try_dataset/'
                check_dirs(data_dir)
                time.sleep(3)
                captureIMGSet(option, no_imgs_to_capture, data_dir)
            if option == 2:
                data_dir = 'try_dataset/'
                check_dirs(data_dir)
                time.sleep(3)
                captureIMGSet(option, no_imgs_to_capture, data_dir)

    print(f'IMAGE CAPTURE...DONE.')
