import cv2
import mediapipe as mp

from pathlib import Path


mp_vision = mp.tasks.vision


class MediapipeTracker:
    def __init__(self):
        root_dirpath = Path(__file__).parents[2]
        root_dirpath = str(root_dirpath).replace('\\', '/')
        self.gesture_tracker = mp_vision.GestureRecognizer.create_from_options(
            mp_vision.GestureRecognizerOptions(
                base_options = mp.tasks.BaseOptions(root_dirpath + "/models/gesture_recognizer.task"),
                running_mode = mp_vision.RunningMode.VIDEO,
                min_hand_detection_confidence = 0.66,
                num_hands = 2
            )
        )
        self.pose_tracker = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options = mp.tasks.BaseOptions(root_dirpath + "/models/pose_landmarker_full.task"),
                running_mode = mp_vision.RunningMode.VIDEO,
                min_pose_detection_confidence = 0.66,
                num_poses = 1
            )
        )


    def track_gestures(self, base_image: cv2.typing.MatLike, current_frame_count: int):
        return self.gesture_tracker.recognize_for_video(
            mp.Image(mp.ImageFormat.SRGB, base_image),
            current_frame_count
        )


    def track_pose(self, base_image: cv2.typing.MatLike, current_frame_count: int):
        return self.pose_tracker.detect_for_video(
            mp.Image(mp.ImageFormat.SRGB, base_image),
            current_frame_count
        )
