import cv2
from typing import Sequence

from core.camera_manager import CameraManager
from core.tracker_data_mapper import TrackerDataMapper
from utils.number_utils import NumberUtils
from utils.color_utils import ColorUtils


COORDS_DEFAULT_SPACING = 16


class HornsDrafter:
    __tracker_data_mapper = TrackerDataMapper()


    def draw_horns_mask(self, base_image: cv2.typing.MatLike, face_landmarks) -> cv2.typing.MatLike:
        output_image = base_image.copy()
        horns_landmarks = self.__tracker_data_mapper.get_horns_coordinates(face_landmarks)
        if horns_landmarks is not None and len(horns_landmarks) > 0:
            current_width, current_height = CameraManager.get_current_resolution()
            left_horn_coords = self.__get_left_horn_coords(horns_landmarks, current_height, current_width)
            right_horn_coords = self.__get_right_horn_coords(horns_landmarks, current_height, current_width)

            color = ColorUtils.generate_random_bgr_code()
            output_image = self.__draw_horn(output_image, left_horn_coords, color)
            output_image = self.__draw_horn(output_image, right_horn_coords, color)
        return output_image


    def __get_left_horn_coords(self, horns_landmarks: Sequence[cv2.typing.Point], current_height: int, current_width: int) -> Sequence[int]:
        horn_landmarks = horns_landmarks[0]
        y = int(horn_landmarks.y * current_height) + NumberUtils.get_random_nonzero_int(-4, 4)
        x = int(horn_landmarks.x * current_width) + NumberUtils.get_random_nonzero_int(-4, 4)
        return [
            (int(x), int(y)),
            (int(x - COORDS_DEFAULT_SPACING), int(y - COORDS_DEFAULT_SPACING * 0.33)),
            (int(x - COORDS_DEFAULT_SPACING * 2), int(y - COORDS_DEFAULT_SPACING * 1.5)),
            (int(x - COORDS_DEFAULT_SPACING * 0.75), int(y - COORDS_DEFAULT_SPACING * 6)),
            (int(x - COORDS_DEFAULT_SPACING * 1.25), int(y - COORDS_DEFAULT_SPACING * 8)),
            (int(x - COORDS_DEFAULT_SPACING * 2), int(y - COORDS_DEFAULT_SPACING * 8.75)),
            (int(x + COORDS_DEFAULT_SPACING * 0.25), int(y - COORDS_DEFAULT_SPACING * 8.25)),
            (int(x + COORDS_DEFAULT_SPACING * 0.75), int(y - COORDS_DEFAULT_SPACING * 7.75)),
            (int(x + COORDS_DEFAULT_SPACING * 1.5), int(y - COORDS_DEFAULT_SPACING * 5.25)),
            (int(x + COORDS_DEFAULT_SPACING), int(y - COORDS_DEFAULT_SPACING)),
            (int(x), int(y)),
        ]


    def __get_right_horn_coords(self, horns_landmarks: Sequence[cv2.typing.Point], current_height: int, current_width: int) -> Sequence[int]:
        horn_landmarks = horns_landmarks[1]
        y = int(horn_landmarks.y * current_height) + NumberUtils.get_random_nonzero_int(-4, 4)
        x = int(horn_landmarks.x * current_width) + NumberUtils.get_random_nonzero_int(-4, 4)
        return [
            (int(x), int(y)),
            (int(x + COORDS_DEFAULT_SPACING), int(y - COORDS_DEFAULT_SPACING * 0.33)),
            (int(x + COORDS_DEFAULT_SPACING * 2), int(y - COORDS_DEFAULT_SPACING * 1.5)),
            (int(x + COORDS_DEFAULT_SPACING * 0.75), int(y - COORDS_DEFAULT_SPACING * 6)),
            (int(x + COORDS_DEFAULT_SPACING * 1.25), int(y - COORDS_DEFAULT_SPACING * 8)),
            (int(x + COORDS_DEFAULT_SPACING * 2), int(y - COORDS_DEFAULT_SPACING * 8.75)),
            (int(x - COORDS_DEFAULT_SPACING * 0.25), int(y - COORDS_DEFAULT_SPACING * 8.25)),
            (int(x - COORDS_DEFAULT_SPACING * 0.75), int(y - COORDS_DEFAULT_SPACING * 7.75)),
            (int(x - COORDS_DEFAULT_SPACING * 1.5), int(y - COORDS_DEFAULT_SPACING * 5.25)),
            (int(x - COORDS_DEFAULT_SPACING), int(y - COORDS_DEFAULT_SPACING)),
            (int(x), int(y)),
        ]


    def __draw_horn(self, base_image: cv2.typing.MatLike, horn_coords: Sequence[int], color: tuple[int, int, int]) -> cv2.typing.MatLike:
        output_image = base_image.copy()
        for index, coords in enumerate(horn_coords):
            if not index:
                continue
            cv2.line(
                output_image,
                horn_coords[index - 1],
                coords,
                color,
                2
            )
        return output_image
