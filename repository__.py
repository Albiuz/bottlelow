import json
import os
import time
import glob
from models import Frame
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, save_img
from collections.abc import MutableSequence

class Repository:
    def __init__(self, path_dir):
        self.path_dir = path_dir
        self.file_path = os.path.join(path_dir, 'data.json')

        self.list = FilepathList()
        self.current_filepath = ''
        self.__init_metadata()

        self.__len_listeners = []

    def __len__(self):
        print(len(self.list))
        return len(self.list)

    def set_filter(self, **filters):
        self.list = FilepathList(**filters)
        self.__init_metadata()

    def set_len_listener(self, callback):
        self.__len_listeners.append(callback)
        self.update_listeners()
    
    def update_listeners(self):
        for listener in self.__len_listeners:
            listener(len(self))
    
    def __init_metadata(self):
        try:
            pathname = os.path.join(self.path_dir,'*.jpg')
            for filepath in glob.glob(pathname):
                metadata = self.__get_metadata(filepath)
                self.list.append(filepath, metadata)
            self.current_filepath = self.list[-1]
        except:
            print('empty folder')
    
    def __get_json_filepath(self, filepath) -> str:
        return '%s.json' % filepath[:-4]

    def __get_metadata(self, filepath) -> dict:
        with open(self.__get_json_filepath(filepath)) as file:
            return json.load(file)

    def __save_metadata(self, filepath, metadata) -> None:
        json_filepath = self.__get_json_filepath(filepath)
        with open(json_filepath,'w') as f:
            json.dump(metadata, f)

    def __remove_metadata(self, filepath) -> None:
        os.remove(self.__get_json_filepath(filepath))
    
    def save(self, frame:Frame):
        if frame.filepath == '':
            filepath = os.path.join(self.path_dir, '%s.jpg'%int(time.time()))
            frame.filepath = filepath
            img = img_to_array(frame.original)
            self.list.append(filepath, frame.metadata)
            save_img(filepath, img)
            self.update_listeners()
        else:
            self.list.update(frame.filepath, frame.metadata)
        self.__save_metadata(frame.filepath, frame.metadata)
        self.current_filepath = frame.filepath
        
    def load(self, filepath) -> Frame:
        metadata = self.__get_metadata(filepath)
        frame_arr = img_to_array(load_img(filepath), dtype='uint8')
        self.current_filepath = filepath
        return Frame(frame_arr=frame_arr, metadata=metadata, filepath=filepath)

    def delete(self, frame:Frame) -> Frame:
        index = self.list.index(frame.filepath)
        os.remove(frame.filepath)
        self.__remove_metadata(frame.filepath)
        self.list.remove(frame.filepath)
        self.update_listeners()
        if len(self.list) == 0:
            self.current_filepath = ''
            return Frame()
        else:
            return self.load(self.list[index])

    def get_next(self) -> Frame:
        if self.current_filepath == '':
            return Frame()
        index = self.list.index(self.current_filepath)
        if index < len(self.list) -1:
            index += 1
        return self.load(self.list[index])

    def get_prev(self) -> Frame:
        index = self.list.index(self.current_filepath)
        if index > 0:
            index -= 1
        return self.load(self.list[index])


class FilepathList(MutableSequence):
    def __init__(self, **filters):
        super().__init__()
        self._list = list()
        self.__filters = filters
    
    def __repr__(self):
        return "<{0} {1}>".format(self.__class__.__name__, self._list)

    def __len__(self):
        """List length"""
        return len(self._list)

    def __getitem__(self, ii):
        """Get a list item"""
        return self._list[ii]

    def __delitem__(self, ii):
        """Delete an item"""
        del self._list[ii]

    def __setitem__(self, ii, val):
        # optional: self._acl_check(val)
        self._list[ii] = val

    def __str__(self):
        return str(self._list)

    def insert(self, ii, val):
        # optional: self._acl_check(val)
        self._list.insert(ii, val)

    def set_filters(self, **kwargs):
        self.__filters = kwargs

    def remove(self, filepath):
        if filepath in self._list:
            self._list.remove(filepath)

    def update(self, filepath, metadata):
        for attr in self.__filters:
            if attr in metadata:
                if metadata[attr] == self.__filters[attr]:
                    break
        else:        
            self._list.remove(filepath)

    def append(self, filepath, metadata):
        if len(self.__filters) == 0:
            self.insert(len(self._list), filepath)
            return
        for attr in self.__filters:
            if attr in metadata:
                if metadata[attr] == self.__filters[attr]:
                    self.insert(len(self._list), filepath)
                    continue       