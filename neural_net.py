from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
from sklearn.model_selection import train_test_split
import json
import numpy as np
from frame import Frame, FrameList, Lev_State
from cv2 import cvtColor
from threading import Thread
import glob
import os

ROI_H = 280
ROI_W = 40

class NeuralNet:
    def __init__(self, model_path:str):
        self.path_dir = 'weights'
        self.model = load_model(model_path)

    @property
    def model_list(self):
        return glob.glob(os.path.join(self.path_dir, '*.h5'))
        #return [os.path.basename(x for x in glob.glob(self.path_dir))]
    
    def set_model(self, model_path):
        self.model = load_model(model_path)

    def __preprocess(self, frame):
        frame = cvtColor(frame, 6) #RGB2GRAY=6 ; BGR2GRAY=7
        frame = np.expand_dims(frame/255.0, axis=0)
        frame = np.expand_dims(frame, axis=3)
        return frame
    
    def predict(self, frame:Frame):
        arr = frame.roi_array
        arr = self.__preprocess(arr)
        pred = self.model.predict(arr)
        frame.levels = self._get_levels(pred)
        frame.foam = self._get_foam(pred)
        if len(frame.levels) > 0:
            if frame.levels[-1]['center'] > frame.limit[0] and frame.levels[-1]['center'] < frame.limit[1]:
                frame.passed = Lev_State.passed
            else:
                frame.passed = Lev_State.not_passed
        else:
            frame.passed = Lev_State.not_passed
        return frame.passed == Lev_State.passed

    def _get_levels(self, pred:np.ndarray):
        threshold = 0.5

        flag_start = False
        global_position_level = []
        accumulator = []
        score = 0
        for i in range(pred.shape[1]):
            if pred[0,i,0,0] > threshold:
                if flag_start == False:
                    flag_start = True
                    accumulator = [i]
                    score = pred[0,i,0,0]
                else:
                    accumulator.append(i)
                    score += pred[0,i,0,0]
            else:
                if flag_start == True:
                    flag_start = False
                    center = accumulator[len(accumulator)//2]
                    height = accumulator[-1]-accumulator[0]
                    score = score / len(accumulator)
                    global_position_level.append({'center':center, 'height':height, 'score':score })
        return global_position_level

    def _get_foam(self, pred:np.ndarray):
        threshold = 0.5

        flag_start = False
        global_position_level = []
        start_point = 0
        end_point = 0
        score = 0
        for i in range(pred.shape[1]):
            if pred[0,i,0,0] > threshold:
                if flag_start == False:
                    flag_start = True
                    start_point = i
                    score = pred[0,i,0,0]
                else:
                    score += pred[0,i,0,0]
            else:
                if flag_start == True:
                    flag_start = False
                    end_point = i
                    score = score / (end_point-start_point)
                    global_position_level.append({'start_pt':start_point, 'end_pt':end_point, 'score':score })
        return global_position_level

    def set_train_callback(self, callback):
        self.train_callbacks = [FitCallback(callback)]

    def train(self, frame_lists:FrameList) -> str:
        trainer = Trainer()
        trainer.set_data(frame_lists)
        trainer.train(self.train_callbacks)
        model_name = 'weight_test.h5'
        path = trainer.save(model_name)
        self.set_model(path)

class Trainer:
    def __init__(self) -> None:
        self.model = self.__set_model()
        self.callbacks = []
        self.started = False

    def set_data(self, frame_lists:FrameList):
        data = []
        targets = []
        
        for frame_list in frame_lists:
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

    def __set_model(self):
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

    def set_train_callback(self, callback):
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

    def save(self,model_name):
        path = os.path.join('weights','%s.h5'%model_name)
        self.model.save(path, save_format="h5")
        return path

class FitCallback(keras.callbacks.Callback):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def on_epoch_end(self, epoch, logs=None):
        text = "epoch {}, loss: {:7.4f}".format(epoch, logs['loss'])
        self.callback(text)