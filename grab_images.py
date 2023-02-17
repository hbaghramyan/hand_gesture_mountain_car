# source https://www.geeksforgeeks.org/extract-video-frames-from-webcam-and-save-to-images-using-python/
import cv2
  
# Opens the inbuilt camera of laptop to capture video.
cap = cv2.VideoCapture(0)
i = 0

# for i in list(range(10)):
#     ret, frame = cap.read()
      
#     # This condition prevents from infinite looping 
#     # incase video ends.
#     if ret == False:
#         break
      
#     # Save Frame by Frame into disk using imwrite method
#     cv2.imwrite('dl/' +  'Frame' + str(i) + ".jpg", frame)
#     i += 1
  
for i in list(range(1000)):
    ret, frame = cap.read()
      
    # This condition prevents from infinite looping 
    # incase video ends.
    if ret == False:
        break
      
    # Save Frame by Frame into disk using imwrite method
    cv2.imwrite('stop/' +  'Frame' + str(i) + ".jpg", frame)
    i += 1

print("Phase 'stop' done.")

for i in list(range(100)):
    ret, frame = cap.read()
      
    # This condition prevents from infinite looping 
    # incase video ends.
    if ret == False:
        break
      
    # Save Frame by Frame into disk using imwrite method
    cv2.imwrite('back_valid/' +  '1' + str(i) + ".jpg", frame)
    i += 1

print("Phase 'back' done.")

for i in list(range(1000)):
    ret, frame = cap.read()
      
    # This condition prevents from infinite looping 
    # incase video ends.
    if ret == False:
        break
      
    # Save Frame by Frame into disk using imwrite method
    cv2.imwrite('go/' +  'Frame' + str(i) + ".jpg", frame)
    i += 1

# print("Phase 'go' done.")

cap.release()
cv2.destroyAllWindows()

