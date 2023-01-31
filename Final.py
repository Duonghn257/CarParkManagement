import tkinter
from tkinter import filedialog

import PIL.ImageTk
import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from datetime import datetime
import os
import pickle
import cvzone
import torch
import torch.nn as nn
import torchvision
from torchvision import models, datasets, transforms
from PIL import Image
from MyLib import Predictor, BaseTranform
import time
import json

carpark_path = None
json_path = None

root = Tk()
root.geometry("960x900")
root.configure(bg="grey")
img_video = Label(root, bg="grey")

img = None
video = None
count = 1
posList = None
check_mode = 0
btn = None

# Load time
try:
    with open("./CarParkProject/carPark1.json", 'r') as f:
        t_pre = json.load(f)
except:
    t_pre = []

## Load model resnet18 as pretrained
# Change last Layer to 5
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Load model
model_path = './model/cardetection2.pt'
checkpoint = torch.load(model_path)['model_state_dict']
model.load_state_dict(checkpoint)
model.eval()

# Class
class_index = ['No', 'Yes']

# Prediction
predictor = Predictor(class_index)

# Parameters to Transform
resize = 128
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = BaseTranform(resize, mean, std)
model_result = 'x'

# Model Predictor
def predicted(model, imgCropped):
    global model_result

    # PIL_Images
    PIL_image = Image.fromarray(np.uint8(imgCropped)).convert('RGB')
    img_transformed = transform(PIL_image)
    img_transformed = img_transformed.unsqueeze_(0)
    out = model(img_transformed)
    result = predictor.predict_max(out)
    return result

def chooseVideo():
    global carpark_path, video, count, posList, json_path
    if count>1:
        delete_button()
        CloseVideo()
    carpark_path = filedialog.askopenfilename(title="Select file",
                                              filetypes=(("MP4 File", "*.mp4"),
                                                         ("all files", "*.*")))
    posList_path = carpark_path[:-4]
    json_path = carpark_path[:-4] + ".json"
    try:
        with open(posList_path, 'rb') as f:
            posList = pickle.load(f)
    except:
        posList = []
    print(carpark_path)
    create_button()
    count+=1
    getVideo()

def getVideo():
    global video
    video = cv2.VideoCapture(carpark_path)
    showVideo()

def showVideo():
    global img, n_frames, key, pause_state
    # Repeat Video
    if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, img = video.read()
    cv2.waitKey(20)
    if ret == True:

        img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
        height, width = img.shape[:2]
        time_now = datetime.now()
        # DIP Result
        if check_mode == 1:
            dip_Process(img)
        # Model Result
        elif check_mode == 2:
            model_Process(img)
        time_now = str(time_now)[:-7]
        cv2.putText(img, text=time_now, org=(width-240, height-15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(img))
        img_video.configure(image=img)

        img_video.image = img
        root.after(10, showVideo)
    else:
        img_video.image = ""

def pause_video():
    pass

def CloseVideo():
    global video
    delete_button()
    video.release()

def write_json(l1):
    with open(json_path, 'w') as f:
        json.dump(l1, f)

def image_processing(imgProcess):
    imgGray = cv2.cvtColor(imgProcess, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
    return imgDilate
spaceCounter = 0

def dip_Process(imgProcess):
    global colorbtn, spaceCounter
    spaceCounter = 0
    park_state = None
    time_list = []
    imgPro = image_processing(imgProcess)
    for i, pos in enumerate(posList):
        x, y,  = pos[0]
        ix, iy = pos[1]
        imgCropped = imgPro[y:iy, x:ix]
        dienTich = (iy-y)*(ix-x)
        cvzone.putTextRect(imgProcess, f'{i}', (x, y - 3),
                           scale=1.2, thickness=1, offset=0)
        count = cv2.countNonZero(imgCropped)

        # Parked - Can't
        if count/dienTich>0.2:
            colorbtn = (0, 0, 255)
            color = "#DE3838"
        # Empty - Can
        else:
            colorbtn = (0, 255, 0)
            color = "#8EDB87"
            spaceCounter += 1

        # Configure color
        btn[i].configure(bg=color)
        cv2.rectangle(imgProcess, pos[0], pos[1], color=colorbtn, thickness=2)
    # print(time_list)
    CarCountingLabel.configure(text=f'Free : {spaceCounter}/{len(posList)}')

def model_Process(imgProcess):
    # spaceCounter = 0
    for i, pos in enumerate(posList):
        x, y, = pos[0]
        ix, iy = pos[1]
        imgCropped = imgProcess[y:iy, x:ix]
        cvzone.putTextRect(img, f'{i}', (x, y - 3), scale=1.2, thickness=1, offset=0)
        PIL_image = Image.fromarray(np.uint8(imgCropped)).convert('RGB')
        img_transformed = transform(PIL_image)
        img_transformed = img_transformed.unsqueeze_(0)
        out = model(img_transformed)
        result = predictor.predict_max(out)

        if result == 'Yes':
            colorbtn = (0, 0, 255)
            color = "#DE3838"
        else:
            colorbtn = (0, 255, 0)
            color = "#8EDB87"
            # spaceCounter += 1
        btn[i].configure(bg=color)
        # cv2.rectangle(imgProcess, pos[0], pos[1], color=colorbtn, thickness=2)
    dip_Process(imgProcess)
    CarCountingLabel.configure(text=f'Free : {spaceCounter}/{len(posList)}')

def create_button():
    global btn
    n = len(posList)
    files = []
    btn = []
    col = 1
    posButton = 1
    for i in range(n):
        files.append(str(i))
    for i in range(len(files)):
        if i % 15 == 0:
            posButton = 1
            col += 1
        posButton += 1
        btn.append(Button(root, text=files[i],height=1, width=2, bg="green", font=("times new roman", 15, "bold"), command=blink))
        btn[i].place(x=100 + posButton * 40, y=610 + 40 * col)

def delete_button():
    for i in range(len(posList)):
        btn[i].place_forget()

def blink():
    colorbtn = (255, 255, 255)


def model_mode():
    global check_mode
    if check_mode != 1:
        dipmode_button.configure(bg='grey')
        modelMode_button.configure(bg='white')
        check_mode = 1
    elif check_mode == 1:
        dipmode_button.configure(bg='grey')
        modelMode_button.configure(bg='grey')
        check_mode = 0

def choosevideo(e):
    chooseVideo()

def dip_mode():
    global check_mode
    if check_mode != 2:
        dipmode_button.configure(bg='white')
        modelMode_button.configure(bg='grey')
        check_mode = 2
    elif check_mode == 2:
        dipmode_button.configure(bg='grey')
        modelMode_button.configure(bg='grey')
        check_mode = 0

time_image = PhotoImage(file="time2.png")
menu = Menu(root)
root.config(menu=menu)

# add video file menu
add_video_menu = Menu(menu)
menu.add_cascade(label="Add", menu=add_video_menu)
add_video_menu.add_command(label="Add video files", command=chooseVideo)

# add button
button = Button(text="Choose Video", command=chooseVideo)
button1 = Button(text="Close Video", command=CloseVideo)

# add pause button
pause_button = Button(text="Close Video", command=pause_video)
pause_button.pack()

# add model mode button
modelMode_button = Button(text="DIP", height=1, width=3, bg="grey", fg="black", font=("times new roman", 15, "bold"),
                          command=model_mode)
# add dip mode button
dipmode_button = Button(text="CNN", height=1, width=3, bg="grey", fg="black", font=("times new roman", 15, "bold"),
                        command=dip_mode)

# add car counting
CarCountingLabel = Label(root, text=f'Free : ', font=("times new roman", 15, "bold"), bg="#8EDB87", fg="black")
time_button = Button(root, image=time_image, bg="grey")

# place all
y = 640
modelMode_button.place(x=550, y=y)
dipmode_button.place(x=500, y=y)
CarCountingLabel.place(x=350, y=y)
img_video.place(x=0, y=0)

# Shortcuts
root.bind('<Control-o>', choosevideo)

root.mainloop()