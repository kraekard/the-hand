import cv2
import mediapipe

from core.camera_manager import CameraManager
from core.tracker_data_mapper import TrackerDataMapper
from utils.number_utils import NumberUtils


WHITE_RGB_CODE = (255, 255, 255)
CROSSES = [
    cv2.MARKER_CROSS,
    cv2.MARKER_TILTED_CROSS
]
OPEN_SHAPES = [
    cv2.MARKER_DIAMOND,
    cv2.MARKER_TRIANGLE_DOWN,
    cv2.MARKER_TRIANGLE_UP
]
SHAPES = [*CROSSES, *OPEN_SHAPES, cv2.MARKER_STAR]


class DrawingSpec:
    def __init__(self, color, thickness, circle_radius = 4):
        self.color = color
        self.thickness = thickness
        self.circle_radius = circle_radius


class ImageDrafter:
    __mp_vision = mediapipe.tasks.vision
    __mp_drawing_utils = __mp_vision.drawing_utils
    __tracker_data_mapper = TrackerDataMapper()

    def draw_base_hands(self, base_image: cv2.typing.MatLike, hand_landmarks) -> cv2.typing.MatLike:
        output_image = base_image.copy()
        if hand_landmarks is not None:
            for hand_landmark in hand_landmarks:
                self.__mp_drawing_utils.draw_landmarks(
                    output_image,
                    hand_landmark,
                    self.__mp_vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    DrawingSpec(
                        color = (0, 255, 0),
                        thickness = 3
                    ),
                    DrawingSpec(
                        color = (155, 80, 150),
                        thickness = 2
                    )
                )
        return output_image


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
                output_image = self.__draw_symbols_2(output_image, (x, y), WHITE_RGB_CODE, 1)

                for _ in range(3):
                    random_y = y + NumberUtils.get_random_nonzero_int(-50, 50)
                    random_x = x + NumberUtils.get_random_nonzero_int(-50, 50)
                    size_markup = NumberUtils.get_random_int(5, 10) * 0.05
                    output_image = self.__draw_symbols_2(
                        output_image,
                        (random_x, random_y),
                        self.__get_random_rgb_color(),
                        1,
                        size_markup
                    )
            return output_image


    def __get_random_rgb_color(self):
        return (
            NumberUtils.get_random_int(0, 255),
            NumberUtils.get_random_int(0, 255),
            NumberUtils.get_random_int(0, 255)
        )


    def __draw_symbols_2(self, base_image: cv2.typing.MatLike, coordinates: cv2.typing.Point, color: cv2.typing.Scalar, thickness: int = 1, size_markup: float = 1) -> cv2.typing.MatLike:
        output_image = base_image.copy()
        cv2.drawMarker(output_image, coordinates, color, cv2.MARKER_DIAMOND, int(40 * size_markup), thickness)
        cv2.drawMarker(output_image, coordinates, color, cv2.MARKER_TILTED_CROSS, int(60 * size_markup), thickness)
        return output_image


    def draw_horns_mask(self, base_image: cv2.typing.MatLike, face_landmarks) -> cv2.typing.MatLike:
        output_image = base_image.copy()
        horns_landmarks = self.__tracker_data_mapper.get_horns_coordinates(face_landmarks)
        if horns_landmarks is not None and len(horns_landmarks) > 0:
            current_width, current_height = CameraManager.get_current_resolution()
            for horn_landmark in horns_landmarks:
                y = int(horn_landmark.y * current_height) + NumberUtils.get_random_nonzero_int(-2, 2)
                x = int(horn_landmark.x * current_width) + NumberUtils.get_random_nonzero_int(-2, 2)
                rgb_color = self.__get_random_rgb_color()
                distance_modifier = 15

                cv2.line(
                    output_image,
                    (x - distance_modifier, y - distance_modifier),
                    (x, y - distance_modifier * 8),
                    rgb_color,
                    2
                )
                cv2.line(
                    output_image,
                    (x + distance_modifier, y - distance_modifier),
                    (x, y - distance_modifier * 8),
                    rgb_color,
                    2
                )
                cv2.line(
                    output_image,
                    (x - distance_modifier, y - distance_modifier),
                    (x, y + distance_modifier),
                    rgb_color,
                    2
                )
                cv2.line(
                    output_image,
                    (x + distance_modifier, y - distance_modifier),
                    (x, y + distance_modifier),
                    rgb_color,
                    2
                )
                # cv2.drawMarker(
                #     output_image,
                #     (x, y),
                #     self.__get_random_rgb_color(),
                #     cv2.MARKER_TRIANGLE_UP,
                #     40,
                #     3
                # )
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
                    (x, y + NumberUtils.get_random_nonzero_int(-4, 4)),
                    self.__get_random_rgb_color(),
                    [cv2.MARKER_DIAMOND, cv2.MARKER_TILTED_CROSS][NumberUtils.get_random_int(0, 1)],
                    NumberUtils.get_random_int(20, 25),
                    2
                )
        return output_image
