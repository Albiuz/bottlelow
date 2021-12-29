from time import sleep
from videoStream import WebcamVideoStream
from gpiozero import Button, LED
from frame import Frame, FrameList

class Controller:
    def __init__(self) -> None:
        self.vs = WebcamVideoStream().start()

        self.sensor = Button(26)
        self.led = LED(19)

    def enable_sensor(self, frame_listener):
        self.frame_listener = frame_listener
        self.sensor.when_deactivated = self.pipe

    def disable_sensor(self):
        self.sensor.when_deactivated = None

    def take_pick(self) -> Frame:
        return Frame(self.vs.read())

    def pipe(self):
        self.led.on()
        sleep(0.1)
        passed = self.frame_listener(self.vs.read())
        sleep(0.5)
        self.led.off()
