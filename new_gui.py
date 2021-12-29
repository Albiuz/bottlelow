from PyQt5.QtWidgets import QApplication, QGridLayout, QGroupBox, QHBoxLayout, QMessageBox, QVBoxLayout, QLabel, QWidget, QMainWindow, QSizePolicy, QPushButton, QSlider, QProgressDialog, QLCDNumber, QStackedLayout
from PyQt5.QtGui import QImage, QPixmap, QPalette
from PyQt5.QtCore import QSize, Qt, QCoreApplication
import numpy as np

from processor import Core, CoreState
import typing

import sys

btn_style = 'margin : 10px;'
btn_style_active = 'margin : 10px; background-color : green;'
btn_style_danger = 'margin : 10px; background-color : red;'
button_size = QSize(121,51)

class StartButton(QPushButton):
    def __init__(self):
        super().__init__('Start')
        self.setFixedSize(button_size)
        core = Core.get_instance()
        self.clicked.connect(lambda: core.setState(CoreState.running))
        core.setStateListener(self.listener)
    
    def listener(self, state:CoreState):
        if state == CoreState.running:
            self.setStyleSheet(btn_style_active)
        else:
            self.setStyleSheet(btn_style)

class StopButton(QPushButton):
    def __init__(self):
        super().__init__('Stop')
        self.setFixedSize(button_size)
        core = Core.get_instance()
        self.clicked.connect(lambda: core.setState(CoreState.not_running))
        core.setStateListener(self.listener)
    
    def listener(self, state:CoreState):
        if state == CoreState.not_running:
            self.setStyleSheet(btn_style_danger)
        else:
            self.setStyleSheet(btn_style)

class TrainModeButton(QPushButton):
    def __init__(self):
        super().__init__('Train mode')
        self.setFixedSize(button_size)
        core = Core.get_instance()
        self.clicked.connect(lambda: core.setState(CoreState.train_mode))
        core.setStateListener(self.listener)
    
    def listener(self, state:CoreState):
        if state == CoreState.train_mode:
            self.setStyleSheet(btn_style_active)
        else:
            self.setStyleSheet(btn_style)

class ExitButton(QPushButton):
    def __init__(self):
        super().__init__('exit')
        self.setFixedSize(button_size)
        self.setStyleSheet(btn_style_danger)
        self.clicked.connect(QCoreApplication.quit)

class ControlWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        #self.setFixedSize(300,35)
        h_layout = QHBoxLayout()
        h_layout.addWidget(StartButton())
        h_layout.addWidget(StopButton())
        h_layout.addWidget(TrainModeButton())
        h_layout.addWidget(ExitButton())
        self.setLayout(h_layout)

class ControlROIWidget(QWidget):
    def __init__(self,parent):
        super().__init__(parent)
        core:Core = Core.get_instance()

        layout = QHBoxLayout()
        def create_UpDown_layout(title, l1, l2, callback):
            group = QGroupBox(title)
            btn_up = QPushButton(l1)
            btn_up.setFixedSize(button_size)
            btn_up.setStyleSheet(btn_style)
            btn_up.clicked.connect(lambda: callback(-2))
            btn_down = QPushButton(l2)
            btn_down.setFixedSize(button_size)
            btn_down.setStyleSheet(btn_style)
            btn_down.clicked.connect(lambda: callback(+2))
            layout = QVBoxLayout()
            layout.addWidget(btn_up)
            layout.addWidget(btn_down)
            group.setLayout(layout)
            return group
        layout.addWidget(create_UpDown_layout('max limit', '+', '-', core.move_max_limit)) #TEST
        layout.addWidget(create_UpDown_layout('min limit', '+', '-', core.frame.move_min_limit))
        layout.addWidget(create_UpDown_layout('move rect.', '<-', '->', core.frame.move_roi))
        self.setLayout(layout)

class CheckWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        core:Core = Core.get_instance()

        text = QLabel()
        text.setText('0/0')
        #processor.repository.set_len_listener(text.setText)

        group = QGroupBox('True level')
        btn_up = QPushButton('+')
        btn_up.setStyleSheet(btn_style)
        btn_up.setFixedSize(button_size)
        btn_up.clicked.connect(lambda: core.frame.move_true_lev(-2))
        btn_down = QPushButton('-')
        btn_down.setStyleSheet(btn_style)
        btn_down.setFixedSize(button_size)
        btn_down.clicked.connect(lambda: core.frame.move_true_lev(+2))
        btn_del = QPushButton('delete')
        btn_del.setStyleSheet(btn_style)
        btn_del.setFixedSize(button_size)
        btn_del.clicked.connect(core.frame.del_true_lev)
        layout = QVBoxLayout()
        layout.addWidget(btn_up)
        layout.addWidget(btn_down)
        layout.addWidget(btn_del)
        group.setLayout(layout)

        next_btn = QPushButton('OK and Next')
        next_btn.setStyleSheet(btn_style)
        next_btn.setFixedSize(button_size)
        next_btn.clicked.connect(core.set_frame_checked)

        back_btn = QPushButton('Back')
        back_btn.setStyleSheet(btn_style)
        back_btn.setFixedSize(button_size)
        back_btn.clicked.connect(core.get_prev)

        train_btn = QPushButton('Train')
        train_btn.setStyleSheet(btn_style)
        train_btn.setFixedSize(button_size)
        train_btn.clicked.connect(core.train)

        v_layout = QVBoxLayout()
        v_layout.addWidget(text)
        v_layout.addWidget(group)
        v_layout.addWidget(next_btn)
        v_layout.addWidget(back_btn)
        v_layout.addSpacing(30)
        v_layout.addWidget(train_btn)
        self.setLayout(v_layout)

class ImageWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        core:Core = Core.get_instance()

        image = QLabel()
        image.setFixedSize(360,480) # (240, 320) x 1.5
        image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        image.setScaledContents(True)

        def changePixmap(npArray:np.ndarray):
            height, width, channel = npArray.shape
            bytesPerLine = 3 * width
            qImg = QImage(npArray.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            qpixmap = QPixmap(qImg)
            image.setPixmap(qpixmap)

        core.set_img_listener(changePixmap)

        btn_left = QPushButton('<-')
        btn_left.setStyleSheet(btn_style)
        btn_left.clicked.connect(core.get_prev)
        btn_right = QPushButton('->')
        btn_right.setStyleSheet(btn_style)
        btn_right.clicked.connect(core.get_next)
        btn_del = QPushButton('delete')
        btn_del.setStyleSheet(btn_style)
        btn_del.clicked.connect(core.del_img)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_left)
        btn_layout.addWidget(btn_right)
        btn_layout.addWidget(btn_del)

        main_layout = QVBoxLayout()
        main_layout.addWidget(image)
        #main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)


class TrainWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.core:Core = Core.get_instance()
        self.train_state = 1

        self.stacked = QStackedLayout()
        self.stacked.addWidget(self.create_collecting_widget())
        self.stacked.addWidget(QLabel(text='no train mode'))

        self.setLayout(self.stacked)
        self.core.setStateListener(self.listener)

    def listener(self, state:CoreState):
        if state == CoreState.train_mode:
            self.stacked.setCurrentIndex(0)
        else:
            self.stacked.setCurrentIndex(1)

    def create_collecting_widget(self):
        contator = QLCDNumber()
        contator.display(42)
        #self.processor.repository.set_len_listener(contator.display)
        nav_group = QGroupBox('Navigator')
        btn_left = QPushButton('ok')
        btn_left.setStyleSheet(btn_style)
        btn_left.clicked.connect(self.core.set_frame_checked)
        btn_del = QPushButton('delete')
        btn_del.setStyleSheet(btn_style)
        btn_del.clicked.connect(self.core.del_img)
        layout = QVBoxLayout()
        layout.addWidget(btn_left)
        layout.addWidget(btn_del)
        nav_group.setLayout(layout)

        group = QGroupBox('True level')
        btn_up = QPushButton('+')
        btn_up.setStyleSheet(btn_style)
        btn_up.clicked.connect(lambda: self.core.frame.move_true_lev(-2))
        btn_down = QPushButton('-')
        btn_down.setStyleSheet(btn_style)
        btn_down.clicked.connect(lambda: self.core.frame.move_true_lev(+2))
        btn_del = QPushButton('delete')
        btn_del.setStyleSheet(btn_style)
        btn_del.clicked.connect(self.core.frame.del_true_lev)
        layout = QVBoxLayout()
        layout.addWidget(btn_up)
        layout.addWidget(btn_down)
        layout.addWidget(btn_del)
        group.setLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(nav_group)
        layout.addWidget(group)
        layout.addWidget(contator)

        widget = QWidget()
        widget.setLayout(layout)

        return widget

    
    def set_collecting_widget(self):
        pass
    def set_checking_widget(self):
        pass
    def set_training_widget(self):
        pass
    def set_empty_widget(self):
        pass

class MainWidget(QWidget):
    def __init__(self, parent: typing.Optional['QWidget']) -> None:
        super().__init__(parent=parent)
        self.mainUI()
        
    
    def mainUI(self):
        image = ImageWidget(parent=self)
        image.move(10,10)

        control_state = ControlWidget(parent=self)
        control_state.move(390,10)

        control_roi = ControlROIWidget(parent=self)
        control_roi.move(390,65)

        train_widget = TrainWidget(parent=self)
        train_widget.move(390,230)

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.title = 'my app'
        self.left = 0
        self.top = 0
        #self.w = 1024
        #self.h = 560
        self.setCursor(Qt.BlankCursor)

        self.initUI()

        print(self.geometry().height(),self.geometry().width())

    def initUI(self):
        self.setWindowTitle(self.title) 
        #self.setGeometry(self.left, self.top, self.w, self.h)

        widget = MainWidget(parent=self)

        self.setCentralWidget(widget)

        self.showFullScreen()
        #self.showMaximized()


if __name__ == '__main__':
    core = Core()
    app = QApplication([])
    window = MainWindow()

    sys.exit(app.exec_())