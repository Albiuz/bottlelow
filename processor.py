from neural_net import NeuralNet
#from classifier import Classifier, Trainer
#from frame import Frame, FrameList
from core import Controller, Frame, FrameList
from threading import Lock, Thread
import os
import numpy
import enum

class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """

    _instances = {}

    _lock: Lock = Lock()
    """
    We now have a lock object that will be used to synchronize threads during
    first access to the Singleton.
    """

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

class CoreState(enum.Enum):
    not_running = 0
    running = 1
    train_mode = 2

class Core(metaclass=SingletonMeta):

    def __init__(self):
        self.controller = Controller()
        
        self.img_listeners = []
        self.state_listener = []
        self.setState(CoreState.not_running)

        self.frame_manager = FrameManager(self)

        self.neural_net = NeuralNet('weight_gray_3.h5')

        self.temp_framelist = FrameList('img_saved_temp')
        self.framelist = FrameList('img_saved_3')
        self.list_of_framelist:list = [] 
        self.save_img = False

        self.in_real_time = True

    @classmethod
    def get_instance(cls):
        return cls._instances[cls]
    
    def train(self):
        self.neural_net.train(self.framelist)
    
    def setStateListener(self, callback):
        self.state_listener.append(callback)
        callback(self.state)

    def setState(self, state:CoreState):
        self.state = state
        self.frame:Frame = self.controller.take_pick()
        if state == CoreState.not_running:
            self.frame_manager = FrameManager(self)
        elif state == CoreState.running:
            self.frame_manager = FrameManagerRun(self)
        elif state == CoreState.train_mode:           
            self.frame_manager = FrameManagerCollecting(self)
        for l in self.state_listener:
            l(self.state)

    @property
    def frame(self): # TODO -> change frame with current frame
        return self._frame
    @frame.setter
    def frame(self, frame:Frame):
        self._frame = frame
        self._frame.listeners = self.img_listeners

    def move_roi(self, val) -> None:
        if not self.in_real_time:
            Frame.roi[0][1] += val
            Frame.roi[1][1] += val
        self._frame.roi[0][1] += val
        self._frame.roi[1][1] += val
        self.update()

    def set_img_listener(self, callback):
        self.img_listeners.append(callback)
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
        if lim > Frame.limit[0] and lim < self._frame.shape[0]:
            Frame.limit[1] = lim
            self.update()
    
    def move_true_lev(self, val):
        if self.frame.lev_true:
            self.frame.lev_true += val
        else:
            lev = self.frame.level
            if lev:
                self.frame.lev_true = lev['center']
            else:
                self.frame.lev_true = self.frame.limit[0] + (self.frame.limit[1]-self.frame.limit[0])//2
        self.update()

    def del_true_lev(self):
        self.frame.lev_true = None
        self.update()

    def set_frame_checked(self):
        '''move the frame from temporary folder to training folder'''

        if self._frame.lev_true == None and not self._frame.level == None:
            self._frame.lev_true = self._frame.level['center']
        #self._frame.checked = True # DEPRECATED
        self.update()
        self.temp_framelist.remove(self._frame)
        self.framelist.append(self._frame)
        if len(self.temp_framelist) > 0:
            self.frame = self.temp_framelist[-1]
        else:
            self.frame = Frame.empty()

    def get_prev(self):
        if self.state == CoreState.train_mode:
            pass
    
    def get_next(self):
        if self.state == CoreState.train_mode:
            pass
            
    def del_img(self):
        if self.state == CoreState.train_mode:
            if len(self.temp_framelist) > 0:
                self.temp_framelist.remove(self._frame)
                self.frame = self.temp_framelist[-1]
            else:
                self.frame = Frame.empty()

    def update(self):
        frame_arr = self._frame.frame_drawed
        for listener in self.img_listeners:
            listener(frame_arr)

class FrameManager:
    def __init__(self, core:Core):
        self.core:Core = core
        self.core.controller.enable_sensor(self.listener_from_cam)
        self._init_current_frame()
        self.__listener = []
    def _init_current_frame(self):
        self.core._frame = Frame.empty()
    def set_listener(self,callback):
        self.__listener.append(callback)
    def listener_from_cam(self, array:numpy.ndarray):
        pass
    def get_next(self):
        pass
    def get_prev(self):
        pass
    def del_current_frame(self):
        pass
    def check_frame(self):
        pass

class FrameManagerRun(FrameManager):
    def __init__(self, core:Core) -> None:
        super().__init__(core)
        
    def listener_from_cam(self, array:numpy.ndarray):
        self.core.in_real_time = False
        self.core.frame = Frame(frame_arr=array)
        result = self.core.neural_net.predict(self.core.frame)
        self.core.update()

class FrameManagerCollecting(FrameManager):
    def __init__(self, core:Core, frame_list_dir:str='img_saved_3') -> None:
        super().__init__(core)
        self.temp_framelist = FrameList('img_saved_temp')
        self.framelist = FrameList(frame_list_dir)
    
    def listener_from_cam(self, array:numpy.ndarray):
        self.core.in_real_time = False
        self.core.frame = Frame(frame_arr=array)
        result = self.core.neural_net.predict(self.core._frame)
        self.core.update()

    def check_frame(self):
        if self.core._frame.is_empty:
            return
        #self.core.frame.set_frame_checked() #DEPRECATED
        self.framelist.append(self.core.frame)
        self.temp_framelist.remove(self.core.frame)
        if len(self.temp_framelist) > 0:
            self.core.frame = self.temp_framelist[-1]
        else:
            self.core.frame = Frame.empty()

    def del_current_frame(self):
        if self.core.frame.is_empty:
            return
        self.temp_framelist.remove(self.core.frame)
        if len(self.temp_framelist) > 0:
            self.core.frame = self.temp_framelist[-1]
        else:
            self.core.frame = Frame.empty()

class FrameManagerNavigator(FrameManager):
    def __init__(self, core:Core, frame_list_dir:str='img_saved_3'):     
        self._framelist = FrameList(frame_list_dir)
        super().__init__(core)
        self._current_index = len(self._framelist)-1
    
    def _init_current_frame(self):
        if len(self._framelist) > 0:
            self.core.frame = self._framelist[-1]
        else:
            self.core.frame = Frame.empty()

    def listener_from_cam(self, array: numpy.ndarray):
        pass

    def get_next(self):
        if self._current_index < len(self._framelist)-1:
            self._current_index += 1
            self.core.frame = self._framelist[self._current_index]
        else:
            #deactivate button
            pass
        
    def get_prev(self):
        if self._current_index > 0:
            self._current_index -= 1
            self.core.frame = self._framelist[self._current_index]
        else:
            #deactivate button
            pass

    def del_current_frame(self):
        if self.core.frame.is_empty:
            return
        self._framelist.remove(self.core.frame)
        if len(self._framelist) > self._current_index:
            self.core.frame = self._framelist[self._current_index]
        else:
            self.core.frame = Frame.empty()
            #deactivate button