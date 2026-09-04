import cv2
import time

from core.camera_manager import CameraManager
from core.mediapipe_tracker import MediapipeTracker
from core.image_drafter import ImageDrafter
from core.image_converter import ImageConverter


ESC_KEY = 27


def run():
    camera_manager = CameraManager()
    mediapipe_tracker = MediapipeTracker()
    image_drafter = ImageDrafter()
    image_converter = ImageConverter()
    is_tracking = False
    current_frame_count = 0

    while not is_tracking:
        is_tracking, _ = camera_manager.get_current_frame()
        time.sleep(0.5)
    while is_tracking:
        is_tracking, current_frame_image = camera_manager.get_current_frame()
        output_image = current_frame_image.copy()

        blurred_image = cv2.bilateralFilter(output_image, 5, 50, 25)
        gesture_recognition_result = mediapipe_tracker.track_gestures(blurred_image, current_frame_count)
        face_recognition_result = mediapipe_tracker.track_face(blurred_image, current_frame_count)
        current_frame_count += 1

        if gesture_recognition_result is not None:
            hand_landmarks = gesture_recognition_result.hand_landmarks
            output_image = image_drafter.draw_fingertips_symbols(output_image, hand_landmarks)

        if face_recognition_result is not None:
            face_landmarks = face_recognition_result.face_landmarks
            output_image = image_drafter.draw_horns_mask(output_image, face_landmarks)
            output_image = image_drafter.draw_eyes_mask(output_image, face_landmarks)
            output_image = image_drafter.draw_mouth_mask(output_image, face_landmarks)

        bw_output_image = image_converter.to_bw_image(output_image)
        bw_output_image = cv2.bitwise_not(bw_output_image)

        cv2.imshow("Output", cv2.flip(output_image, 1))
        cv2.imshow("B&W Negative Output", cv2.flip(bw_output_image, 1))

        if cv2.waitKey(24) == ESC_KEY:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
