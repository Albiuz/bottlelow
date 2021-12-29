from threading import Thread, Lock
import cv2

class WebcamVideoStream :
    def __init__(self, src = 0, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        while self.grabbed == False:
            (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.loop, args=(), daemon=True)
        self.thread.start()
        return self

    def loop(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

if __name__ == "__main__" :
    vs = WebcamVideoStream().start()
    frame = vs.read()
    cv2.imshow('webcam', frame)

    while True :
        key = cv2.waitKey(1)

        if key == 27:
            break
        else:
            frame = vs.read()
            cv2.imshow('webcam', frame)

    vs.stop()
    cv2.destroyAllWindows()
