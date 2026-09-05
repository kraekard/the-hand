import cv2

from core.camera_manager import CameraManager
from core.tracker_data_mapper import TrackerDataMapper
from core.gesture_tracker import GestureTracker
from utils.number_utils import NumberUtils
from utils.color_utils import ColorUtils


WHITE_RGB_CODE = (255, 255, 255)
MARKERS_OF_CROSS_SYMBOLS = [cv2.MARKER_CROSS, cv2.MARKER_TILTED_CROSS]
MARKERS_OF_OPEN_SYMBOLS = [cv2.MARKER_DIAMOND, cv2.MARKER_SQUARE, cv2.MARKER_TRIANGLE_DOWN, cv2.MARKER_TRIANGLE_UP]


class HandsDrafter:
    __tracker_data_mapper = TrackerDataMapper()
    __gesture_tracker = GestureTracker()


    def draw_fingertips_symbols(self, base_image: cv2.typing.MatLike, hand_landmarks) -> cv2.typing.MatLike:
        output_image = base_image.copy()
        if self.__gesture_tracker.is_fist_closed():
            return output_image

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
                        ColorUtils.generate_random_bgr_code(),
                        MARKERS_OF_OPEN_SYMBOLS[NumberUtils.get_random_int(0, 2)],
                        int(40 * size_markup)
                    )
                    cv2.drawMarker(
                        output_image,
                        (random_x, random_y),
                        ColorUtils.generate_random_bgr_code(),
                        MARKERS_OF_CROSS_SYMBOLS[NumberUtils.get_random_int(0, 1)],
                        int(60 * size_markup)
                    )
            return output_image


    def draw_fist_effects(self, base_image: cv2.typing.MatLike, hand_landmarks) -> cv2.typing.MatLike:
        output_image = base_image.copy()
        if not self.__gesture_tracker.is_fist_closed():
            return output_image

        fists_center_coords = self.__tracker_data_mapper.get_fists_center_coordinates(hand_landmarks)
        if fists_center_coords is not None and len(fists_center_coords) > 0:
            current_width, current_height = CameraManager.get_current_resolution()
            for fist_center_coords in fists_center_coords:
                x = int(fist_center_coords[0] * current_width)
                y = int(fist_center_coords[1] * current_height)
                cv2.circle(
                    output_image,
                    (x, y),
                    200,
                    ColorUtils.generate_random_bgr_code(),
                    NumberUtils.get_random_int(2, 6)
                )
        return output_image
