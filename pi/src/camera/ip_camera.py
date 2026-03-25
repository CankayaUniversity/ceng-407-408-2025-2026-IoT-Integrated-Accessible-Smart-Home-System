import cv2


class IPCameraStream:
    def __init__(self, url: str, width: int = 640, height: int = 480):
        self.url = url
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.url)
        return self.cap.isOpened()

    def read(self):
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        frame = cv2.resize(frame, (self.width, self.height))
        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()