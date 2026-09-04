import cv2
import mediapipe

from typing import Sequence

from core.camera_manager import CameraManager
from core.tracker_data_mapper import TrackerDataMapper
from utils.number_utils import NumberUtils


WHITE_RGB_CODE = (255, 255, 255)
MARKERS_OF_CROSS_SYMBOLS = [
    cv2.MARKER_CROSS,
    cv2.MARKER_TILTED_CROSS
]
MARKERS_OF_OPEN_SYMBOLS = [
    cv2.MARKER_DIAMOND,
    cv2.MARKER_SQUARE,
    cv2.MARKER_TRIANGLE_DOWN,
    cv2.MARKER_TRIANGLE_UP
]
HORN_DEFAULT_DRAWING_STEP = 15


class ImageDrafter:
    __tracker_data_mapper = TrackerDataMapper()

    def draw_fingertips_symbols(self, base_image: cv2.typing.MatLike, hand_landmarks) -> cv2.typing.MatLike:
        output_image = base_image.copy()
        fingertips_landmarks = self.__tracker_data_mapper.get_fingertips_coordinates(hand_landmarks)
        if fingertips_landmarks is not None and len(fingertips_landmarks) > 0:
            for fingertip_landmark in fingertips_landmarks:
                output_image = self.__draw_symbols(
                    output_image,
                    fingertip_landmark
                )
        return output_image


    def __draw_symbols(self, base_image: cv2.typing.MatLike, coordinates) -> cv2.typing.MatLike:
            output_image = base_image.copy()
            if coordinates is not None:
                current_width, current_height = CameraManager.get_current_resolution()
                y, x = int(coordinates.y * current_height), int(coordinates.x * current_width)
                cv2.drawMarker(output_image, (x, y), WHITE_RGB_CODE, cv2.MARKER_DIAMOND, 40)
                cv2.drawMarker(output_image, (x, y), WHITE_RGB_CODE, cv2.MARKER_TILTED_CROSS, 60)

                for _ in range(2):
                    random_y = y + NumberUtils.get_random_nonzero_int(-50, 50)
                    random_x = x + NumberUtils.get_random_nonzero_int(-50, 50)
                    size_markup = NumberUtils.get_random_int(5, 10) * 0.075

                    cv2.drawMarker(
                        output_image,
                        (random_x, random_y),
                        self.__get_random_rgb_color(),
                        MARKERS_OF_OPEN_SYMBOLS[NumberUtils.get_random_int(0, 2)],
                        int(40 * size_markup)
                    )
                    cv2.drawMarker(
                        output_image,
                        (random_x, random_y),
                        self.__get_random_rgb_color(),
                        MARKERS_OF_CROSS_SYMBOLS[NumberUtils.get_random_int(0, 1)],
                        int(60 * size_markup)
                    )
            return output_image


    def __get_random_rgb_color(self):
        return (
            NumberUtils.get_random_int(0, 255),
            NumberUtils.get_random_int(0, 255),
            NumberUtils.get_random_int(0, 255)
        )


    def draw_horns_mask(self, base_image: cv2.typing.MatLike, face_landmarks) -> cv2.typing.MatLike:
        output_image = base_image.copy()
        horns_landmarks = self.__tracker_data_mapper.get_horns_coordinates(face_landmarks)
        if horns_landmarks is not None and len(horns_landmarks) > 0:
            current_width, current_height = CameraManager.get_current_resolution()
            left_horn_coords = self.__get_left_horn_coords(horns_landmarks, current_height, current_width)
            right_horn_coords = self.__get_right_horn_coords(horns_landmarks, current_height, current_width)

            color = self.__get_random_rgb_color()
            output_image = self.__draw_horn(output_image, left_horn_coords, color)
            output_image = self.__draw_horn(output_image, right_horn_coords, color)
        return output_image


    def __get_left_horn_coords(self, horns_landmarks: Sequence[cv2.typing.Point], current_height: int, current_width: int) -> Sequence[int]:
        horn_landmarks = horns_landmarks[0]
        y = int(horn_landmarks.y * current_height) + NumberUtils.get_random_nonzero_int(-4, 4)
        x = int(horn_landmarks.x * current_width) + NumberUtils.get_random_nonzero_int(-4, 4)
        return [
            (int(x), int(y)),
            (int(x + HORN_DEFAULT_DRAWING_STEP), int(y - HORN_DEFAULT_DRAWING_STEP * 0.33)),
            (int(x + HORN_DEFAULT_DRAWING_STEP * 2), int(y - HORN_DEFAULT_DRAWING_STEP * 1.5)),
            (int(x + HORN_DEFAULT_DRAWING_STEP * 0.75), int(y - HORN_DEFAULT_DRAWING_STEP * 6)),
            (int(x + HORN_DEFAULT_DRAWING_STEP * 1.25), int(y - HORN_DEFAULT_DRAWING_STEP * 8)),
            (int(x + HORN_DEFAULT_DRAWING_STEP * 2), int(y - HORN_DEFAULT_DRAWING_STEP * 8.75)),
            (int(x - HORN_DEFAULT_DRAWING_STEP * 0.25), int(y - HORN_DEFAULT_DRAWING_STEP * 8.25)),
            (int(x - HORN_DEFAULT_DRAWING_STEP * 0.75), int(y - HORN_DEFAULT_DRAWING_STEP * 7.75)),
            (int(x - HORN_DEFAULT_DRAWING_STEP * 1.5), int(y - HORN_DEFAULT_DRAWING_STEP * 5.25)),
            (int(x - HORN_DEFAULT_DRAWING_STEP), int(y - HORN_DEFAULT_DRAWING_STEP)),
            (int(x), int(y)),
        ]


    def __get_right_horn_coords(self, horns_landmarks: Sequence[cv2.typing.Point], current_height: int, current_width: int) -> Sequence[int]:
        horn_landmarks = horns_landmarks[1]
        y = int(horn_landmarks.y * current_height) + NumberUtils.get_random_nonzero_int(-4, 4)
        x = int(horn_landmarks.x * current_width) + NumberUtils.get_random_nonzero_int(-4, 4)
        return [
            (int(x), int(y)),
            (int(x - HORN_DEFAULT_DRAWING_STEP), int(y - HORN_DEFAULT_DRAWING_STEP * 0.33)),
            (int(x - HORN_DEFAULT_DRAWING_STEP * 2), int(y - HORN_DEFAULT_DRAWING_STEP * 1.5)),
            (int(x - HORN_DEFAULT_DRAWING_STEP * 0.75), int(y - HORN_DEFAULT_DRAWING_STEP * 6)),
            (int(x - HORN_DEFAULT_DRAWING_STEP * 1.25), int(y - HORN_DEFAULT_DRAWING_STEP * 8)),
            (int(x - HORN_DEFAULT_DRAWING_STEP * 2), int(y - HORN_DEFAULT_DRAWING_STEP * 8.75)),
            (int(x + HORN_DEFAULT_DRAWING_STEP * 0.25), int(y - HORN_DEFAULT_DRAWING_STEP * 8.25)),
            (int(x + HORN_DEFAULT_DRAWING_STEP * 0.75), int(y - HORN_DEFAULT_DRAWING_STEP * 7.75)),
            (int(x + HORN_DEFAULT_DRAWING_STEP * 1.5), int(y - HORN_DEFAULT_DRAWING_STEP * 5.25)),
            (int(x + HORN_DEFAULT_DRAWING_STEP), int(y - HORN_DEFAULT_DRAWING_STEP)),
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


    def draw_eyes_mask(self, base_image: cv2.typing.MatLike, face_landmarks) -> cv2.typing.MatLike:
        output_image = base_image.copy()
        eyes_landmarks = self.__tracker_data_mapper.get_eyes_coordinates(face_landmarks)
        if eyes_landmarks is not None and len(eyes_landmarks) > 0:
            current_width, current_height = CameraManager.get_current_resolution()
            for eye_landmark in eyes_landmarks:
                y = int(eye_landmark.y * current_height) + NumberUtils.get_random_nonzero_int(-2, 2)
                x = int(eye_landmark.x * current_width) + NumberUtils.get_random_nonzero_int(-2, 2)

                tilted_or_regular_cross = NumberUtils.get_random_int(0, 10)
                if tilted_or_regular_cross >= 4:
                    cv2.drawMarker(
                        output_image,
                        (x, y),
                        self.__get_random_rgb_color(),
                        cv2.MARKER_TILTED_CROSS,
                        40,
                        3
                    )
                else:
                    cv2.drawMarker(
                        output_image,
                        (x, y),
                        self.__get_random_rgb_color(),
                        cv2.MARKER_CROSS,
                        45,
                        2
                    )

                circle_or_diamond = NumberUtils.get_random_int(0, 10)
                if circle_or_diamond >= 4:
                    cv2.circle(
                        output_image,
                        (x, y),
                        16,
                        self.__get_random_rgb_color(),
                        2
                    )
                else:
                    cv2.drawMarker(
                        output_image,
                        (x, y),
                        self.__get_random_rgb_color(),
                        cv2.MARKER_DIAMOND,
                        32,
                        2
                    )
        return output_image


    def draw_mouth_mask(self, base_image: cv2.typing.MatLike, face_landmarks) -> cv2.typing.MatLike:
        output_image = base_image.copy()
        mouth_landmarks = self.__tracker_data_mapper.get_mouth_coordinates(face_landmarks)
        if mouth_landmarks is not None and len(mouth_landmarks) > 0:
            current_width, current_height = CameraManager.get_current_resolution()
            y = int(mouth_landmarks[0].y * current_height)
            start_x = int(mouth_landmarks[0].x * current_width)
            end_x = int(mouth_landmarks[-1].x * current_width)
            x_values = set([])
            for value in range(start_x, end_x, 15):
                x_values.add(value)

            for x in x_values:
                cv2.drawMarker(
                    output_image,
                    (x, y + NumberUtils.get_random_nonzero_int(-6, 6)),
                    self.__get_random_rgb_color(),
                    cv2.MARKER_TILTED_CROSS,
                    NumberUtils.get_random_int(20, 25),
                    2
                )
        return output_image
