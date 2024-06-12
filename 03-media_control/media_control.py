import cv2
import keras
from matplotlib import pyplot as plt
from model import GestureRecognitionModel
import numpy as np
from pynput.keyboard import Key, Controller
import json
import sys
import time
import os

CONDITIONS = ['stop', 'like', 'dislike']
IMG_SIZES = [64]
COLOR_CHANNELS = 3
SIZE = 64
MODEL_READY = True

def preprocess_image(img):
    if COLOR_CHANNELS == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, (SIZE, SIZE))

    return img_resized

# had to be done with model_json and weight because i could not load a model from keras due to some version errors ?
def load_gesture_model(model_path, weights_path, custom_objects=None):
    print('Loading model …')
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = keras.models.model_from_json(loaded_model_json, custom_objects=custom_objects)
    model.load_weights(weights_path)
    return model

# get command line parameters
try:
    MODEL_READY = sys.argv[1]
    MODEL_READY = MODEL_READY.lower() == 'true'
    
    # specify path to HaGrid Dataset
    data_path = sys.argv[2]
    
    try:
        video_id = int(sys.argv[3])
    except:
        video_id = 0
except:
    print('Please enter all parameters!')
    print('python3 media_control.py <model_loaded> <data_path> <video_id>')
    os._exit(0) 

if MODEL_READY is False:
    gesture_model = GestureRecognitionModel(data_path, CONDITIONS, IMG_SIZES, COLOR_CHANNELS)
    gesture_model.run()
    # if you want to plot an overview of the models history turn plot to True
    gesture_model.print_logs(plot=True)
    
    model_instance = gesture_model.model
    label_names = gesture_model.label_names
    
else:
    model_instance = load_gesture_model('gesture_model.json', 'gesture_model.weights.h5')
    label_names = []
    with open('label_names.json', 'r') as json_file:
        label_names = json.load(json_file) 
        

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    img = preprocess_image(frame)
    img = np.array(img).astype('float32')
    img = img / 255.
    img = img.reshape(-1, SIZE, SIZE, COLOR_CHANNELS)

    prediction = model_instance.predict(img)
    prediction_label = label_names[np.argmax(prediction)]
    print(prediction_label)
    
    # controll computer keys
    my_keyboard = Controller()
    
    match(prediction_label):
        case 'like':
            my_keyboard.press(Key.media_volume_up)
            my_keyboard.release(Key.media_volume_up)
        case 'dislike':
            my_keyboard.press(Key.media_volume_down)
            my_keyboard.release(Key.media_volume_down)
        case 'stop':
            my_keyboard.press(Key.media_play_pause)
            my_keyboard.release(Key.media_play_pause)
        case _:
            print('nothing is recognized!')
            
    # works better with time.sleep but latency is obviously not corresponding to the requirements then.
    # time.sleep(1)


