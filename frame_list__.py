from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, save_img
from collections.abc import MutableSequence
from models import Frame
import os
import glob
import json
import time

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
        del self.__filepathlist[ii]

    def __setitem__(self, ii, frame:Frame) -> None:
        #use it to update only metadata
        if self.__filepathlist[ii] == frame.filepath:
            self.__save_metadata(frame)
        else:
            raise NameError('Error in FrameList.__setitem__')

    def append(self, frame:Frame) -> None:
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