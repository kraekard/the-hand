import cv2

from typing import Sequence


SVGA_RESOLUTION = [800, 600]
HD_RESOLUTION = [1280, 720]
FHD_RESOLUTION = [1920, 1080]


class CameraManager:
    @staticmethod
    def get_current_resolution() -> Sequence[int]:
        return HD_RESOLUTION

    def __init__(self):
        self.__capture = cv2.VideoCapture(0)
        self.__capture.set(cv2.CAP_PROP_FRAME_WIDTH, CameraManager.get_current_resolution()[0])
        self.__capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CameraManager.get_current_resolution()[1])
        self.__capture.set(cv2.CAP_PROP_FPS, 24)
    
    def __del__(self):
        self.__capture.release()

    def get_current_frame(self) -> tuple[bool, cv2.typing.MatLike]:
        return self.__capture.read()
