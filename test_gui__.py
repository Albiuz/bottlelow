from PyQt5.QtWidgets import QApplication, QGridLayout, QGroupBox, QHBoxLayout, QVBoxLayout, QLabel, QWidget, QMainWindow, QSizePolicy, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QPalette
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot

import cv2

import sys

class WebCamThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret,frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgbImage = cv2.rotate(rgbImage, cv2.ROTATE_90_CLOCKWISE)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.title = 'my app'
        self.left = 100
        self.top = 100
        self.w = 800
        self.h = 400

        self.taking_pickture = True

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title) 
        self.setGeometry(self.left, self.top, self.w, self.h)

        widget = self.createMainWidget()
        self.setCentralWidget(widget)

        self.show()

    def setImage(self, label):
        return lambda image : label.setPixmap(QPixmap.fromImage(image))


    def createMainWidget(self):
        imageLabel = QLabel()
        imageLabel.setBackgroundRole(QPalette.Base)
        imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        imageLabel.setScaledContents(False)

        streamLabel = QLabel()
        streamLabel.setBackgroundRole(QPalette.Base)
        streamLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        streamLabel.setScaledContents(False)
        webcamThread = WebCamThread()
        webcamThread.changePixmap.connect(self.setImage(streamLabel))
        #webcamThread.changePixmap.connect(self.setImage(imageLabel))
        
        btn_start = QPushButton('start')
        btn_start.clicked.connect(webcamThread.start)

        btn_take_picture = QPushButton('take picture')

        mainLayout = QHBoxLayout()
        mainLayout.addWidget(imageLabel)
        btnLayout = QVBoxLayout()
        btnLayout.addWidget(btn_start)
        btnLayout.addWidget(btn_take_picture)
        mainLayout.addLayout(btnLayout)
        mainLayout.addWidget(streamLabel)
        mainWidget = QGroupBox('frame')
        mainWidget.setLayout(mainLayout)

        return mainWidget
    

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()

    sys.exit(app.exec_())