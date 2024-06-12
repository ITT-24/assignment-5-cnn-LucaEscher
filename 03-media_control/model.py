import cv2
import json
from matplotlib import pyplot as plt
import numpy as np
import os
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Turned notebook from task 1 into a python class

class GestureRecognitionModel:
    def __init__(self, path, conditions, img_sizes, color_channels=1):
        self.path = path
        self.conditions = conditions
        self.img_sizes = img_sizes
        self.color_channels = color_channels
        self.annotations = self.load_annotations()
        self.times_train = []
        self.times_test = []
        self.accuracies = []
        self.losses = []
        self.confusions = []
        self.example_images = []
        self.size_hist = []
        self.model = None
        self.label_names = []

    def load_annotations(self):
        annotations = dict()
        for condition in self.conditions:
            with open(f'{self.path}/_annotations/{condition}.json') as f:
                annotations[condition] = json.load(f)
        return annotations

    def preprocess_image(self, img, img_size):
        if self.color_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            img_resized = cv2.resize(img, (img_size, img_size))
        except:
            return None
        return img_resized

    def prepare_data(self, img_size):
        images = []
        labels = []
        label_names = []
        for condition in self.conditions:
            print(f'processing: {condition}')
            for filename in tqdm(os.listdir(f'{self.path}/{condition}')):
                UID = filename.split('.')[0]
                img = cv2.imread(f'{self.path}/{condition}/{filename}')
                try:
                    annotation = self.annotations[condition][UID]
                except:
                    continue
                for i, bbox in enumerate(annotation['bboxes']):
                    x1 = int(bbox[0] * img.shape[1])
                    y1 = int(bbox[1] * img.shape[0])
                    w = int(bbox[2] * img.shape[1])
                    h = int(bbox[3] * img.shape[0])
                    x2 = x1 + w
                    y2 = y1 + h
                    crop = img[y1:y2, x1:x2]
                    preprocessed = self.preprocess_image(crop, img_size=img_size)
                    if preprocessed is not None:
                        label = annotation['labels'][i]
                        if label not in label_names:
                            label_names.append(label)
                        label_index = label_names.index(label)
                        images.append(preprocessed)
                        labels.append(label_index)
        return images, labels, label_names

    def split_prepare_train_test(self, images, labels, size):
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train = np.array(X_train).astype('float32') / 255.
        X_val = np.array(X_val).astype('float32') / 255.
        X_test = np.array(X_test).astype('float32') / 255.
        y_train_one_hot = to_categorical(y_train)
        y_val_one_hot = to_categorical(y_val)
        y_test_one_hot = to_categorical(y_test)
        X_train = X_train.reshape(-1, size, size, self.color_channels)
        X_test = X_test.reshape(-1, size, size, self.color_channels)
        X_val = X_val.reshape(-1, size, size, self.color_channels)
        return X_train, y_train_one_hot, X_val, y_val_one_hot, X_test, y_test_one_hot, y_test

    def prepare_model(self, label_names, size):
        batch_size = 32
        epochs = 100
        num_classes = len(label_names)
        activation = 'relu'
        model = Sequential([
            Conv2D(64, kernel_size=(9, 9), activation=activation, input_shape=(size, size, self.color_channels), padding='same'),
            MaxPooling2D(pool_size=(4, 4), padding='same'),
            Conv2D(32, (5, 5), activation=activation, padding='same'),
            MaxPooling2D(pool_size=(3, 3), padding='same'),
            Conv2D(32, (3, 3), activation=activation, padding='same'),
            MaxPooling2D(pool_size=(2, 2), padding='same'),
            Dropout(0.3),
            Flatten(),
            Dense(128, activation=activation),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
        stop_early = EarlyStopping(monitor='val_loss', patience=5)
        return model, batch_size, epochs, reduce_lr, stop_early

    def fit_model(self, model, batch_size, epochs, reduce_lr, stop_early, X_train, train_labels, X_val, val_labels):
        history = model.fit(X_train, train_labels, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, val_labels), callbacks=[reduce_lr, stop_early])
        return history

    def train(self, size):
        print(f'prepare data … ----------------------------')
        images, labels, label_names = self.prepare_data(size)
        print(f'process data … ----------------------------')
        X_train, train_labels, X_val, val_labels, X_test, test_labels, y_test = self.split_prepare_train_test(images, labels, size)
        print(f'prepare model … ----------------------------')
        model, batch_size, epochs, reduce_lr, stop_early = self.prepare_model(label_names, size)
        print(f'train model … ----------------------------')
        return images[0], X_test, test_labels, y_test, label_names, model, self.fit_model(model, batch_size, epochs, reduce_lr, stop_early, X_train, train_labels, X_val, val_labels)

    def run(self):
        for size in self.img_sizes:
            start_time = time.time()
            example_img, X_test, test_labels, y_test, label_names, model, hist = self.train(size)
            self.times_train.append((time.time() - start_time))
            self.losses.append(hist.history['val_loss'])
            self.accuracies.append(hist.history['val_accuracy'])
            start_time_test = time.time()
            y_predictions = model.predict(X_test)
            self.times_test.append(time.time() - start_time_test)
            self.model = model
            self.label_names = label_names
            model_json = model.to_json()
            with open("gesture_model.json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("gesture_model.weights.h5")
            with open('label_names.json', 'w') as json_file:
                json.dump(label_names, json_file)
            y_predictions = np.argmax(y_predictions, axis=1)
            conf_matrix = confusion_matrix(y_test, y_predictions)
            self.confusions.append((conf_matrix, label_names))
            self.example_images.append(example_img)
            self.size_hist.append((size, hist))

    def format_duration(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        formatted_duration = f"{hours} hours, {minutes} minutes, {int(seconds)} seconds, {milliseconds} milliseconds"
        return formatted_duration

    def print_logs(self, plot=False):
        for index, img_size in enumerate(self.img_sizes):
            formatted_train = self.format_duration(self.times_train[index])
            formatted_prediction = self.format_duration(self.times_test[index])
            print('---------')
            print(f'Image size: {img_size}\ntrain time: {formatted_train}\nprediction time: {formatted_prediction}')
            print('---------')
        for index, accuracy in enumerate(self.accuracies):
            print('---------')
            print(f'Image size: {self.img_sizes[index]}\nAccuracy: {accuracy[-1]}\nLoss: {self.losses[index][-1]}')
            print('---------')
        if plot:
            for index, history in enumerate(self.size_hist):
                size, value = history
                loss = value.history['loss']
                val_loss = value.history['val_loss']
                accuracy = value.history['accuracy']
                val_accuracy = value.history['val_accuracy']
                fig = plt.figure(figsize=(15, 7))
                ax = plt.gca()
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy (Line), Loss (Dashes)')
                ax.axhline(1, color='gray')
                plt.plot(accuracy, color='blue', label='Training accuracy')
                plt.plot(val_accuracy, color='red', label='Validation accuracy')
                plt.plot(loss, '--', color='blue', label='Training loss')
                plt.plot(val_loss, '--', color='red', label='Validation loss')
                title = f'Accuracy & Loss Curves'
                plt.title(title)
                plt.legend()
                plt.savefig(title)
                plt.show()
            for conf_matrix, label_names in self.confusions:
                fig, ax = plt.subplots(figsize=(15, 7))
                ConfusionMatrixDisplay(conf_matrix, display_labels=label_names).plot(ax=ax, cmap='Blues')
                plt.xticks(rotation=45)
                plt.savefig('confusion_matrix')
                plt.show()
