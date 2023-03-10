#!/usr/bin/env python3.9
# 
# Title: grab_images.py
# Description: Simple script to capture images to create our hand gesture dataset.
# 
# References:
# - https://www.geeksforgeeks.org/extract-video-frames-from-webcam-and-save-to-images-using-python/
# -------------------------------------------------------------------------------
#!/usr/bin/env python3.9
# 
# Title: grab_images.py
# Description: Simple script to capture images to create our hand gesture dataset.
# 
# References:
# - https://www.geeksforgeeks.org/extract-video-frames-from-webcam-and-save-to-images-using-python/
# -------------------------------------------------------------------------------
#!/usr/bin/env python3.9
# 
# Title: grab_images.py
# Description: Simple script to capture images to create our hand gesture dataset.
# 
# References:
# - https://www.geeksforgeeks.org/extract-video-frames-from-webcam-and-save-to-images-using-python/
# -------------------------------------------------------------------------------
import cv2
import os

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
    
    acronyms = {
        0 : "ATL", # Accelerate to the left
        1 : "DA",  # Don't accelerate
        2 : "ATR"  # Accelerate to the right
        }

    # Opens the built camera of laptop to capture video.
    cap = cv2.VideoCapture(0)
    for i in range(no_imgs_to_capture):
        ret, frame = cap.read()

        # This condition prevents from infinite looping in case video ends.
        if ret == False:
            break

        # Save frame by frame into disk using imwrite method
        cv2.imwrite(f'{out_dir}/{acronyms[option]}{i}_{option}.jpg', frame)
        print(f'{out_dir}/{acronyms[option]}{i}_{option}.jpg ... captured.')
    print('\n\nCapturing images...done.')
    print(f'{"-" :->82}')
    cap.release()
    cv2.destroyAllWindows()
import os

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
    
    acronyms = {
        0 : "ATL", # Accelerate to the left
        1 : "DA",  # Don't accelerate
        2 : "ATR"  # Accelerate to the right
        }

    # Opens the built camera of laptop to capture video.
    cap = cv2.VideoCapture(0)
    for i in range(no_imgs_to_capture):
        ret, frame = cap.read()

        # This condition prevents from infinite looping in case video ends.
        if ret == False:
            break

        # Save frame by frame into disk using imwrite method
        cv2.imwrite(f'{out_dir}/{acronyms[option]}{i}_{option}.jpg', frame)
        print(f'{out_dir}/{acronyms[option]}{i}_{option}.jpg ... captured.')
    print('\n\nCapturing images...done.')
    print(f'{"-" :->82}')
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    data_dir = 'dataset'
    dir_check = os.path.isdir(data_dir)

    if not dir_check:
        os.makedirs(data_dir)
        print(f'dataset directory created: {data_dir}')
    else:
        print(f'{data_dir} already exists.')
    
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
            captureIMGSet(option, no_imgs_to_capture, data_dir)
    print(f'IMAGE CAPTURE...DONE.')
import os

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
    
    acronyms = {
        0 : "ATL", # Accelerate to the left
        1 : "DA",  # Don't accelerate
        2 : "ATR"  # Accelerate to the right
        }

    # Opens the built camera of laptop to capture video.
    cap = cv2.VideoCapture(0)
    for i in range(no_imgs_to_capture):
        ret, frame = cap.read()

        # This condition prevents from infinite looping in case video ends.
        if ret == False:
            break

        # Save frame by frame into disk using imwrite method
        cv2.imwrite(f'{out_dir}/{acronyms[option]}{i}_{option}.jpg', frame)
        print(f'{out_dir}/{acronyms[option]}{i}_{option}.jpg ... captured.')
    print('\n\nCapturing images...done.')
    print(f'{"-" :->82}')
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    data_dir = 'dataset'
    dir_check = os.path.isdir(data_dir)

    if not dir_check:
        os.makedirs(data_dir)
        print(f'dataset directory created: {data_dir}')
    else:
        print(f'{data_dir} already exists.')
    
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
            captureIMGSet(option, no_imgs_to_capture, data_dir)
    print(f'IMAGE CAPTURE...DONE.')
