# Description: Simple script to capture images to create our hand gesture dataset.
#
# References:
# - https://www.geeksforgeeks.org/extract-video-frames-from-webcam-and-save-to-images-using-python/
# -------------------------------------------------------------------------------

# nativ imports
import os
import sys
import time
import datetime

# third party imports
import cv2


def menu():
    """
    Displays a menu with options for the user.
    """
    opts = {
        0: "Accelerate to the left",
        1: "Don't accelerate",
        2: "Accelerate to the right",
        3: "Exit",
    }
    for key, value in opts.items():
        print(f"{key} -- {value}")


def capture_img_set(option, no_imgs_to_capture, out_dir):
    """
    Capture a set of images using the webcam.

    Parameters:
    - option (int): the chosen menu option
    - no_imgs_to_capture (int): number of images to be captured
    - out_dir (str): output directory for saving captured images
    """
    now = datetime.datetime.now()

    # Format it as a string
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    acronyms = {
        0: "acc_l",  # Accelerate to the left
        1: "acc_n",  # Don't accelerate
        2: "acc_r",  # Accelerate to the right
    }

    # Opens the built camera of laptop to capture video.
    cap = cv2.VideoCapture(0)
    for i in range(no_imgs_to_capture):
        ret, frame = cap.read()

        # delay between frames
        time.sleep(0.3)

        # This condition prevents from infinite looping in case video ends.
        if not ret:
            break

        # Save frame by frame into disk using imwrite method
        path = os.path.join(out_dir, f"{acronyms[option]}_{i}_{option}_{timestamp}.jpg")
        print(path)
        cv2.imwrite(path, frame)
        filename = os.path.basename(path)

        print(f"{filename} ... captured.")
    print("\n\nCapturing images...done.")
    print(f'{"-" :->82}')
    cap.release()
    cv2.destroyAllWindows()


def check_dirs(directory):
    """
    Capture a set of images using the webcam.

    Parameters:
    - option (int): the chosen menu option
    - no_imgs_to_capture (int): number of images to be captured
    - out_dir (str): output directory for saving captured images
    """

    print(f"type({directory}): {type(directory)}")
    dir_check = os.path.isdir(directory)

    if not dir_check:
        os.makedirs(directory)
        print(f"dataset directory created: {directory}")
    else:
        print(f"{directory} already exists.")


def main():
    while True:
        menu()
        option = ""

        try:
            option = int(input("Select option from menu: "))
        except ValueError:
            print("Wrong option, select valid option.")

        if option == 3:
            sys.exit()
        else:
            no_imgs_to_capture = int(
                input("How many images do you want to capture? \n")
            )
            print(f'{"-":->82}')

            if option == 0:
                data_dir = "try_dataset/"
                check_dirs(data_dir)
                time.sleep(3)
                capture_img_set(
                    option, no_imgs_to_capture, os.path.join(data_dir, "acc_l")
                )
            if option == 1:
                data_dir = "try_dataset/"
                check_dirs(data_dir)
                time.sleep(3)
                capture_img_set(
                    option, no_imgs_to_capture, os.path.join(data_dir, "acc_n")
                )
            if option == 2:
                data_dir = "try_dataset/"
                check_dirs(data_dir)
                time.sleep(3)
                capture_img_set(
                    option, no_imgs_to_capture, os.path.join(data_dir, "acc_r")
                )

    print("IMAGE CAPTURE...DONE.")


if __name__ == "__main__":
    main()
