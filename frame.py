from drawer import *
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, save_img
from collections.abc import MutableSequence
import time
import os
import glob
import json
import numpy as np
import enum

class Lev_State(enum.Enum):
    not_checked = 'NOT YET CHECKED'
    not_passed = 'NOT PASSED'
    passed = 'PASSED'
    @property
    def color(self):
        return {'NOT YET CHECKED':(128,128,128),
                'NOT PASSED':(0,0,200),
                'PASSED':(0,200,0)}[self.value]

ROI_H = 280
ROI_W = 40

class Frame:
    roi_x0 = 100
    roi_y0 = 20
    roi = [[roi_y0,roi_x0],[roi_y0+ROI_H,roi_x0+ROI_W]]
    limit = [150,250]
    empty_array = np.zeros((320,240,3), dtype='uint8')

    def __init__(self,frame_arr:np.ndarray, metadata={}, filepath='') -> None:
        self.original = frame_arr
        self.filepath = filepath
        self.metadata = metadata
        self.is_empty = False
        self.listeners = []

        self.passed: Lev_State = Lev_State.not_checked
    
    @classmethod
    def empty(cls):
        frame = Frame(np.zeros((320,240,3), dtype='uint8'))
        frame.is_empty = True
        return frame

    @property
    def shape(self):
        return self.original.shape

    @property
    def frame(self):
        return self.original.copy()
    
    @property
    def roi_array(self):
        return self.original[self.roi[0][0]:self.roi[1][0], self.roi[0][1]:self.roi[1][1], : ]

    #def predict(self) -> bool:
    #    pred = Frame.classifier.predict(self.roi_array)
    #    self.levels = self._get_levels(pred)
    #    self.foam = self._get_foam(pred)
    #    if len(self.levels) > 0:
    #        if self.levels[-1]['center'] > self.limit[0] and self.levels[-1]['center'] < self.limit[1]:
    #            self.passed = Lev_State.passed
    #        else:
    #            self.passed = Lev_State.not_passed
    #    else:
    #        self.passed = Lev_State.not_passed
    #    return self.passed == Lev_State.passed

    def get_roi_center(self, frame, region_to_check=[100,0,140,320], draw_line=True):
        pass
    
    def move_roi(self, val, change_global:bool=False) -> None:
        if change_global:
            Frame.roi[0][1] += val
            Frame.roi[1][1] += val
        self.roi[0][1] += val
        self.roi[1][1] += val
        self.update()

    def move_max_limit(self, val) -> None:
        lim = Frame.limit[0]
        lim += val
        if lim > 0 and lim < Frame.limit[1]:
            Frame.limit[0] = lim
        self.update()

    def move_min_limit(self, val) -> None:
        lim = Frame.limit[1]
        lim += val
        if lim > Frame.limit[0] and lim < self.shape[0]:
            Frame.limit[1] = lim
        self.update()

    def move_true_lev(self, val):
        if self.lev_true:
            self.lev_true += val
        else:
            lev = self.level
            if lev:
                self.lev_true = lev['center']
            else:
                self.lev_true = self.limit[0] + (self.limit[1]-self.limit[0])//2
        self.update()

    def del_true_lev(self):
        self.lev_true = None
        self.update()

    def set_frame_checked(self):
        if self.lev_true == None and not self.level == None:
            self.lev_true = self.level['center']
        self.checked = True
        self.update()

    @property
    def level(self):
        if len(self.levels) > 0:
            return self.levels[-1]
        else:
            return None

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

    @property
    def frame_drawed(self):
        frame = self.frame
        draw_rect(frame,self.roi[0],self.roi[1],[0,0,255])
        draw_limit(frame,self.limit,self.roi)
        draw_pred(frame,self.levels,self.foam,self.roi)
        draw_passed(frame,self.passed.color)
        draw_true_lev(frame,self.lev_true,self.roi)
        return frame

    @property
    def metadata(self):
        return {'roi': self.roi,
                #'limit': self.limit,
                'levels': self.levels,
                'foam': self.foam,
                'checked': self.checked,
                'lev_true': self.lev_true,
                'foam_true': self.foam_true}
    @metadata.setter
    def metadata(self, metadata):
        if metadata == {}:
            self.levels = []
            self.foam = []
            self.checked = False
            self.lev_true = None
            self.foam_true = None
        else:
            self.roi = metadata['roi'] # shape (320,240,3)
            #self.limit = metadata['limit']
            self.levels = metadata['levels']
            self.checked = metadata['checked']
            self.lev_true = metadata['lev_true']
            self.foam = metadata['foam']
            self.foam_true = metadata['foam_true']
    
    def update(self):
        frame_arr = self.frame_drawed
        print('listeners n.%d'%len(self.listeners))
        for listener in self.listeners:
            print('update')
            listener(frame_arr)

class FrameList(MutableSequence):
    def __init__(self, directory) -> None:
        super().__init__()
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.__pathdir = directory
        self.__filepathlist = glob.glob(os.path.join(self.__pathdir,'*.jpg'))

    def __get_json_filepath(self, filepath) -> str:
        return '%s.json' % filepath[:-4]

    def __get_metadata(self, filepath) -> dict:
        with open(self.__get_json_filepath(filepath)) as file:
            return json.load(file)
        
    def __save_metadata(self, frame:Frame) -> None:
        json_filepath = self.__get_json_filepath(frame.filepath)
        with open(json_filepath,'w') as f:
            json.dump(frame.metadata, f)

    def __remove_metadata(self, filepath) -> None:
        os.remove(self.__get_json_filepath(filepath))

    def __len__(self) -> int:
        return len(self.__filepathlist)

    def __getitem__(self, ii) -> Frame:
        filepath = self.__filepathlist[ii]
        metadata = self.__get_metadata(filepath)
        frame_arr = img_to_array(load_img(filepath), dtype='uint8')
        return Frame(frame_arr=frame_arr, metadata=metadata, filepath=filepath)

    def __delitem__(self, ii) -> None:
        filepath = self.__filepathlist[ii]
        #remove jpg imgage
        os.remove(filepath)
        #remove json imgage data
        self.__remove_metadata(filepath)
        #remove filepath name
        del self.__filepathlist[ii]

    def __setitem__(self, ii, frame:Frame) -> None:
        #use it to update only metadata
        if self.__filepathlist[ii] == frame.filepath:
            self.__save_metadata(frame)
        else:
            raise NameError('Error in FrameList.__setitem__')

    def append(self, frame:Frame) -> None:
        '''create new filepath name and add it to the list'''
        frame.filepath = os.path.join(self.__pathdir, '%s.jpg'%int(time.time()))
        self.__filepathlist.append(frame.filepath)
        self.__save_metadata(frame)
        img = img_to_array(frame.original)
        save_img(frame.filepath, img)

    def insert(self, index: int, frame: Frame) -> None:
        frame.filepath = os.path.join(self.__pathdir, '%s.jpg'%int(time.time()))
        self.__filepathlist.insert(index, frame.filepath)
        self.__save_metadata(frame)
        img = img_to_array(frame.original)
        save_img(frame.filepath, img)
    
    def remove(self, frame: Frame) -> None:
        i = self.index(frame)
        self.__delitem__(i)

    def index(self, frame: Frame) -> int:
        for i in range(len(self.__filepathlist)):
            if self.__filepathlist[i] == frame.filepath:
                return i
        else:
            raise NameError('frame not found')
