import cv2
import mediapipe

from core.camera_manager import CameraManager
from core.hands_data_converter import HandsDataConverter
from utils.number_utils import NumberUtils


WHITE_RGB_CODE = (255, 255, 255)


class DrawingSpec:
    def __init__(self, color, thickness, circle_radius = 4):
        self.color = color
        self.thickness = thickness
        self.circle_radius = circle_radius


class ImageDrafter:
    __mp_vision = mediapipe.tasks.vision
    __mp_drawing_utils = __mp_vision.drawing_utils
    __hands_data_converter = HandsDataConverter()

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
        fingertips_landmarks = self.__hands_data_converter.get_fingertips_coordinates(hand_landmarks)
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
                for _ in range(5):
                    random_y = y + NumberUtils.prevent_zero(NumberUtils.get_random_int(-50, 50))
                    random_x = x + NumberUtils.prevent_zero(NumberUtils.get_random_int(-50, 50))
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
