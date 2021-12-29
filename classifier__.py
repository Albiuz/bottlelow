from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
from sklearn.model_selection import train_test_split
import json
import numpy as np
from frame_list import FrameList
from cv2 import cvtColor
from threading import Thread
import os

ROI_H = 280
ROI_W = 40

class Classifier:
    def __init__(self, model_path:str):
        self.model = load_model(model_path)

    def __preprocess(self, frame):
        frame = cvtColor(frame, 6) #RGB2GRAY=6 ; BGR2GRAY=7
        frame = np.expand_dims(frame/255.0, axis=0)
        frame = np.expand_dims(frame, axis=3)
        return frame
    
    def predict(self, frame:np.ndarray):
        frame = self.__preprocess(frame)
        pred = self.model.predict(frame)
        return pred


class Trainer:
    def __init__(self) -> None:
        self.model = self.set_model()
        self.callbacks = []
        self.started = False

    def set_data(self, frame_list:FrameList):
        data = []
        targets = []
        
        for frame in frame_list:
            output_mask = np.zeros((ROI_H,1,2))
            #frame = repository.load(i)
            if frame.lev_true:
                y = frame.lev_true
                output_mask[y-4:y+4,0,0] = 1
            if frame.foam_true:
                y = frame.foam_true
                output_mask[y-2:y+2,0,1] = 1
            targets.append(output_mask[2:ROI_H-2,:,:])

            image = frame.roi_array
            #image = image[:,:,0]
            image = cvtColor(image, 6)
            image = np.expand_dims(image, axis=2)
            data.append(image)
            
        # load old imgs
        with open("imgs_preprocessed/export-2021-04-14T17-38-24.110Z.json") as json_file:
            json_data = json.load(json_file)
            for elem in json_data:
                if elem['checked']:
                    filename = elem['External ID']
                    output_mask = np.zeros((ROI_H,1,2))
                    if 'objects' in elem['Label']:
                        for label in elem['Label']['objects']:
                            if label['title'] == 'level':
                                if 'bbox' in label:
                                    y0 = label['bbox']['top']
                                    h = label['bbox']['height']
                                    y1 = y0+h
                                    output_mask[y0:y1,0,0] = 1
                            if label['title'] == 'foam':
                                y0 = label['bbox']['top']
                                h = label['bbox']['height']
                                y1 = y0+h
                                output_mask[y0:y1,0,1] = 1
                    targets.append(output_mask[2:ROI_H-2,:,:])

                    image = load_img('imgs_preprocessed/gray/%s' % filename)
                    image = img_to_array(image)
                    #image = image[:,:,0]
                    image = cvtColor(image, 6)
                    image = np.expand_dims(image, axis=2) #adding a color dimension
                    data.append(image)
        
        data = np.array(data, dtype='float32') / 255
        targets = np.array(targets, dtype='float32')

        split = train_test_split(data, targets, test_size=0.1, random_state=42)
        (self.train_imgs, self.test_imgs) = split[:2]
        (self.train_targets, self.test_targets) = split[2:]

    def set_model(self):
        model = keras.Sequential(
            [
                keras.Input(shape=(ROI_H,ROI_W,1)),
                keras.layers.Conv2D(16, (5, 40), strides=1, padding='valid', activation="relu"),
                keras.layers.Conv2D(16, (7, 1), padding='same', activation="relu"),
                keras.layers.Conv2D(16, (5, 1), padding='same', activation="relu"),
                keras.layers.Conv2D(16, (3, 1), padding='same', activation="relu"),
                keras.layers.Conv2D(2, 1, padding='same', activation='sigmoid')
            ]
        )
        opt = keras.optimizers.Adam(lr=0.01)
        model.compile(loss='mse', optimizer=opt)

        return model

    def set_callback(self, callback):
        self.callbacks = [FitCallback(callback)]
    
    def train(self):
        if self.started:
            print('training already started')
            return
        self.started = True
        self.thread = Thread(target=self._train, args=(), daemon=True)
        self.thread.start()

    def _train(self):
        H = self.model.fit(
            self.train_imgs,self.train_targets,
            validation_data=(self.test_imgs,self.test_targets),
            batch_size=10,
            epochs=100,
            verbose=1,
            callbacks = self.callbacks
        )
        self.started = False

    def save(self):
        path = os.path.join('weights','weight_gray.h5')
        self.model.save(path, save_format="h5")
        return path

class FitCallback(keras.callbacks.Callback):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def on_epoch_end(self, epoch, logs=None):
        text = "epoch {}, loss: {:7.4f}".format(epoch, logs['loss'])
        self.callback(text)