import cv2
import time

from threading import Thread
from typing import Sequence


SVGA_RESOLUTION = [800, 600]
HD_RESOLUTION = [1280, 720]
FHD_RESOLUTION = [1920, 1080]
ESC_KEY = 27


class CameraManager:
    is_tracking: bool = False
    frame: cv2.typing.MatLike = []

    @staticmethod
    def get_current_resolution() -> Sequence[int]:
        return HD_RESOLUTION

    def __init__(self):
        self.__capture = cv2.VideoCapture(0)
        self.__capture.set(cv2.CAP_PROP_FRAME_WIDTH, CameraManager.get_current_resolution()[0])
        self.__capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CameraManager.get_current_resolution()[1])
        self.__capture.set(cv2.CAP_PROP_FPS, 24)

        self.__thread = Thread(target=self.update, args=())
        self.__thread.daemon = True
        self.__thread.start()
    
    def __del__(self):
        self.__capture.release()

    def update(self):
        while True:
            self.is_tracking, self.frame = self.__capture.read()
            time.sleep(.01)

    def get_current_frame(self) -> tuple[bool, cv2.typing.MatLike]:
        return self.is_tracking, self.frame
