import cv2

from core.camera_manager import CameraManager
from core.tracker_data_mapper import TrackerDataMapper
from utils.number_utils import NumberUtils
from utils.color_utils import ColorUtils


class EyesDrafter:
    __tracker_data_mapper = TrackerDataMapper()


    def draw_eyes_mask(self, base_image: cv2.typing.MatLike, face_landmarks) -> cv2.typing.MatLike:
        output_image = base_image.copy()
        eyes_landmarks = self.__tracker_data_mapper.get_eyes_coordinates(face_landmarks)
        if eyes_landmarks is not None and len(eyes_landmarks) > 0:
            current_width, current_height = CameraManager.get_current_resolution()
            is_double_blink = all([eye_landmark_wrapper[0] for eye_landmark_wrapper in eyes_landmarks])

            if is_double_blink:
                color = ColorUtils.generate_random_bgr_code()
                for eye_landmark_wrapper in eyes_landmarks:
                    _, eye_landmark = eye_landmark_wrapper
                    y = int(eye_landmark.y * current_height) + NumberUtils.get_random_nonzero_int(-2, 2)
                    x = int(eye_landmark.x * current_width) + NumberUtils.get_random_nonzero_int(-2, 2)
                    cv2.line(
                        output_image,
                        (x - 20, y - 10),
                        (x + 20, y + 10),
                        color,
                        4
                    )
                    cv2.line(
                        output_image,
                        (x - 20, y + 10),
                        (x + 20, y - 10),
                        color,
                        4
                    )
            else:
                for eye_landmark_wrapper in eyes_landmarks:
                    is_winking, eye_landmark = eye_landmark_wrapper
                    y = int(eye_landmark.y * current_height) + NumberUtils.get_random_nonzero_int(-2, 2)
                    x = int(eye_landmark.x * current_width) + NumberUtils.get_random_nonzero_int(-2, 2)

                    if is_winking:
                        color = ColorUtils.generate_random_bgr_code()
                        cv2.line(
                            output_image,
                            (x - 20, y - 6),
                            (x, y + 2),
                            color,
                            8
                        )
                        cv2.line(
                            output_image,
                            (x, y + 2),
                            (x + 20, y - 6),
                            color,
                            8
                        )
                    else:
                        cv2.drawMarker(
                            output_image,
                            (x, y),
                            ColorUtils.generate_random_bgr_code(),
                            cv2.MARKER_TILTED_CROSS,
                            40,
                            3
                        )

                        circle_or_diamond = NumberUtils.get_random_int(0, 10)
                        if circle_or_diamond >= 4:
                            cv2.circle(
                                output_image,
                                (x, y),
                                16,
                                ColorUtils.generate_random_bgr_code(),
                                2
                            )
                        else:
                            cv2.drawMarker(
                                output_image,
                                (x, y),
                                ColorUtils.generate_random_bgr_code(),
                                cv2.MARKER_DIAMOND,
                                32,
                                2
                            )
        return output_image
