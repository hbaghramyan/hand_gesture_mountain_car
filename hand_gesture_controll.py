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
stats = ((0.4301, 0.4574, 0.4537), (0.2482, 0.2467, 0.2806))

model = ResNet9(3, 3)
model.load_state_dict(torch.load('checkpoints/model_epoch_HB_01_07_2023.pth'))
model.eval()

print(model)


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
font_color = (0, 255, 0)
text_position = (100, 200)
thickness = 2

cap = cv2.VideoCapture(0) 

transform = tt.Compose([
    tt.Resize(64),
    tt.RandomCrop(64),
    tt.RandomHorizontalFlip(),
    tt.ToTensor(),
    tt.Normalize(*stats, inplace=True)
])

class_names = ['GO LEFT', 'DONT MOVE', 'GO RIGHT']
while True:
    ret, frame = cap.read()

    # Preprocess the frame
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_image).unsqueeze(0)
    
    # Class names
    

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Interpret the predictions
    _, predicted = torch.max(output, 1)
    class_index = predicted.item()
    print(class_index, class_names[class_index])
    
    cv2.putText(frame, class_names[class_index], text_position, font, font_scale, thickness)
    cv2.imshow('Hand gesture controll demo...', frame)
    
    if class_index == 0:
         #print('LEFT KEY PRESSED')
         keyboard.press(Key.left)
         keyboard.release(Key.left)
    if class_index == 1: 
         #print('DOWN KEY PRESSED')
         keyboard.press(Key.down)
         keyboard.release(Key.down)
    if class_index == 2:
         #print('RIGHT KEY PRESSED')
         keyboard.press(Key.right)
         keyboard.release(Key.right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
