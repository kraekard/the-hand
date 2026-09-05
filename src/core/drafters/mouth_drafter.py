import cv2
from typing import Sequence

from core.camera_manager import CameraManager
from core.tracker_data_mapper import TrackerDataMapper
from utils.number_utils import NumberUtils
from utils.color_utils import ColorUtils


COORDS_DEFAULT_SPACING = 8


class MouthDrafter:
    __tracker_data_mapper = TrackerDataMapper()


    def draw_mouth_mask(self, base_image: cv2.typing.MatLike, face_landmarks) -> cv2.typing.MatLike:
        output_image = base_image.copy()
        mouth_landmarks = self.__tracker_data_mapper.get_mouth_center_coordinates(face_landmarks)
        if mouth_landmarks is not None and len(mouth_landmarks) > 0:
            current_width, current_height = CameraManager.get_current_resolution()
            coords: Sequence[cv2.typing.Point] = []
            for mouth_landmark in mouth_landmarks:
                x = int(mouth_landmark[0] * current_width)
                y = int(mouth_landmark[1] * current_height)
                y_variance = NumberUtils.get_random_nonzero_int(-2, 2)
                coords.extend(
                    [
                        [
                            (int(x + COORDS_DEFAULT_SPACING), y_variance + int(y - COORDS_DEFAULT_SPACING * 1.75)),
                            (int(x + COORDS_DEFAULT_SPACING), y_variance + int(y - COORDS_DEFAULT_SPACING * 1.1))
                        ],
                        [
                            (int(x + COORDS_DEFAULT_SPACING), y_variance + int(y - COORDS_DEFAULT_SPACING * 1.1)),
                            (int(x - COORDS_DEFAULT_SPACING), y_variance + int(y + COORDS_DEFAULT_SPACING * 1.1))
                        ],
                        [
                            (int(x - COORDS_DEFAULT_SPACING), y_variance + int(y + COORDS_DEFAULT_SPACING * 1.1)),
                            (int(x - COORDS_DEFAULT_SPACING), y_variance + int(y + COORDS_DEFAULT_SPACING * 1.75))
                        ],
                        [
                            (int(x + COORDS_DEFAULT_SPACING), y_variance + int(y + COORDS_DEFAULT_SPACING * 1.75)),
                            (int(x + COORDS_DEFAULT_SPACING), y_variance + int(y + COORDS_DEFAULT_SPACING * 1.1))
                        ],
                        [
                            (int(x + COORDS_DEFAULT_SPACING), y_variance + int(y + COORDS_DEFAULT_SPACING * 1.1)),
                            (int(x - COORDS_DEFAULT_SPACING), y_variance + int(y - COORDS_DEFAULT_SPACING * 1.1))
                        ],
                        [
                            (int(x - COORDS_DEFAULT_SPACING), y_variance + int(y - COORDS_DEFAULT_SPACING * 1.1)),
                            (int(x - COORDS_DEFAULT_SPACING), y_variance + int(y - COORDS_DEFAULT_SPACING * 1.75))
                        ]
                    ]
                )

            output_image = self.__draw_mouth_symbol(output_image, coords)
        return output_image


    def __draw_mouth_symbol(self, base_image: cv2.typing.MatLike, coords: Sequence[cv2.typing.Point]) -> cv2.typing.MatLike:
        output_image = base_image.copy()
        color = ColorUtils.generate_random_bgr_code()
        for index, coord in enumerate(coords):
            if index % 6 == 0:
                color = ColorUtils.generate_random_bgr_code()
            start, end = coord
            cv2.line(
                output_image,
                (start[0], start[-1]),
                (end[0], end[-1]),
                color,
                2
            )
        return output_image
