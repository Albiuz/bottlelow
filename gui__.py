from PyQt5.QtWidgets import QApplication, QGridLayout, QGroupBox, QHBoxLayout, QMessageBox, QVBoxLayout, QLabel, QWidget, QMainWindow, QSizePolicy, QPushButton, QSlider, QProgressDialog
from PyQt5.QtGui import QImage, QPixmap, QPalette
from PyQt5.QtCore import Qt, QCoreApplication

from processor import Processor, ProcessorState

import sys

btn_style = 'padding : 5px ; margin : 10px;'
btn_style_active = 'padding : 5px ; margin : 10px; background-color : green;'
btn_style_danger = 'padding : 5px ; margin : 10px; background-color : red;'

class StartButton(QPushButton):
    def __init__(self, processor:Processor):
        super().__init__('Start')
        processor.setState(ProcessorState.running)
        processor.setStateListener(self.listener)
    
    def listener(self, state:ProcessorState):
        if state == ProcessorState.running:
            self.setStyleSheet('padding : 5px ; margin : 10px; background-color : green;')
        else:
            self.setStyleSheet('padding : 5px ; margin : 10px;')

class StopButton(QPushButton):
    def __init__(self, processor:Processor):
        super().__init__('Stop')
        processor.setState(ProcessorState.not_running)
        processor.setStateListener(self.listener)
    
    def listener(self, state:ProcessorState):
        if state == ProcessorState.not_running:
            self.setStyleSheet('padding : 5px ; margin : 10px; background-color : red;')
        else:
            self.setStyleSheet('padding : 5px ; margin : 10px;')

class TrainModeButton(QPushButton):
    def __init__(self, processor:Processor):
        super().__init__('Train mode')
        processor.setState(ProcessorState.train_mode)
        processor.setStateListener(self.listener)
    
    def listener(self, state:ProcessorState):
        if state == ProcessorState.train_mode:
            self.setStyleSheet('padding : 5px ; margin : 10px; background-color : green;')
        else:
            self.setStyleSheet('padding : 5px ; margin : 10px;')

    

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.title = 'my app'
        self.left = 0
        self.top = 0
        self.w = 1024
        self.h = 560

        #self.controller= Controller()
        self.processor:Processor = Processor()

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title) 
        self.setGeometry(self.left, self.top, self.w, self.h)

        widget = self.createMainWidget()
        self.setCentralWidget(widget)

        self.showFullScreen()
        #self.showMaximized()

    def setPixmap(self,label):
        def changePixmap(npArray):
            height, width, channel = npArray.shape
            bytesPerLine = 3 * width
            qImg = QImage(npArray.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            qpixmap = QPixmap(qImg)
            label.setPixmap(qpixmap)
        return changePixmap

    def createMainWidget(self):
        imageLabel = QLabel()
        #imageLabel.setBackgroundRole(QPalette.Base)
        imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored) #QSizePolicy.Ignored
        imageLabel.setScaledContents(True)
        imageLabel.setAlignment(Qt.AlignCenter)
        self.processor.img_listener = self.setPixmap(imageLabel)
        
        btn_start = StartButton(self.processor)
        btn_stop = StopButton(self.processor)
        btn_train = TrainModeButton(self.processor)
        #btn_train.clicked.connect(train_core)

        btn_saveimg = QPushButton('save images')
        btn_saveimg.setStyleSheet(btn_style)
        def save_img():
            if self.processor.save_img == True:
                self.processor.save_img = False
                btn_saveimg.setStyleSheet(btn_style)
            else:
                self.processor.save_img = True
                btn_saveimg.setStyleSheet(btn_style_active)

        btn_saveimg.clicked.connect(save_img)
        #text_saveimg = QLabel()
        #self.core.save_img_listener = text_saveimg.setText

        text_train = QLabel()
        text_train.setText('train text')
        self.processor.trainer.set_callback(text_train.setText)


        def create_UpDown_layout(title, l1, l2, callback):
            group = QGroupBox(title)
            btn_up = QPushButton(l1)
            btn_up.setStyleSheet(btn_style)
            btn_up.clicked.connect(lambda: callback(-2))
            btn_down = QPushButton(l2)
            btn_down.setStyleSheet(btn_style)
            btn_down.clicked.connect(lambda: callback(+2))
            layout = QVBoxLayout()
            layout.addWidget(btn_up)
            layout.addWidget(btn_down)
            group.setLayout(layout)
            return group
        
        def create_set_level():
            group = QGroupBox('True level')
            btn_up = QPushButton('+')
            btn_up.setStyleSheet(btn_style)
            btn_up.clicked.connect(lambda: self.processor.move_true_lev(-2))
            btn_down = QPushButton('-')
            btn_down.setStyleSheet(btn_style)
            btn_down.clicked.connect(lambda: self.processor.move_true_lev(+2))
            btn_del = QPushButton('delete')
            btn_del.setStyleSheet(btn_style)
            btn_del.clicked.connect(self.processor.del_true_lev)
            layout = QVBoxLayout()
            layout.addWidget(btn_up)
            layout.addWidget(btn_down)
            layout.addWidget(btn_del)
            group.setLayout(layout)
            return group

        btn_exit = QPushButton('exit')
        btn_exit.setStyleSheet(btn_style_danger)
        btn_exit.clicked.connect(QCoreApplication.quit)

        btn_left = QPushButton('<-')
        btn_left.setStyleSheet(btn_style)
        btn_left.clicked.connect(self.processor.get_prev)
        btn_right = QPushButton('->')
        btn_right.setStyleSheet(btn_style)
        btn_right.clicked.connect(self.processor.get_next)
        btn_last = QPushButton('->|')
        btn_last.setStyleSheet(btn_style)
        btn_last.clicked.connect(self.processor.get_last)
        btn_del = QPushButton('delete')
        btn_del.setStyleSheet(btn_style)
        btn_del.clicked.connect(self.processor.del_img)

        state_layout = QHBoxLayout()
        state_layout.addWidget(btn_start)
        state_layout.addWidget(btn_stop)
        state_layout.addWidget(btn_train)
        state_layout.addWidget(btn_exit)

        lim_rec_layout = QVBoxLayout()
        lim_rec_layout.addWidget(create_UpDown_layout('max limit', '+', '-', self.processor.move_max_limit))
        lim_rec_layout.addWidget(create_UpDown_layout('min limit', '+', '-', self.processor.move_min_limit))
        lim_rec_layout.addWidget(create_UpDown_layout('move rect.', '<-', '->', self.processor.move_roi))

        true_val_layout = QVBoxLayout()
        true_val_layout.addWidget(create_UpDown_layout('true foam', '+', '-', print))
        true_val_layout.addWidget(create_set_level())

        train_set_layout = QVBoxLayout()
        train_set_layout.addWidget(QLabel().setText('img saved: __'))
        train_set_layout.addWidget(QPushButton('train').clicked.connect(self.processor.train))

        train_layout = QHBoxLayout()
        train_layout.addLayout(true_val_layout)
        train_layout.addLayout(train_set_layout)

        control_layout = QVBoxLayout()
        control_layout.addLayout(state_layout)
        control_layout.addLayout(train_layout)

        frameButtonLayout = QHBoxLayout()
        frameButtonLayout.addWidget(btn_left)
        frameButtonLayout.addWidget(btn_right)
        frameButtonLayout.addWidget(btn_del)
        frameLayout = QVBoxLayout()
        frameLayout.addWidget(imageLabel)
        frameLayout.addLayout(frameButtonLayout)

        mainLayout = QHBoxLayout()
        mainLayout.addLayout(frameLayout, 2)
        mainLayout.addLayout(control_layout,3)
        mainWidget = QWidget()
        mainWidget.setLayout(mainLayout)

        return mainWidget

    def show_train_popup(self, processor:Processor):
        popup = QMessageBox()
        popup.setWindowTitle('Train mode')



if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()

    sys.exit(app.exec_())