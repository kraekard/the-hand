import cv2

from typing import Sequence

from core.wink_tracker import WinkTracker

GESTURE_THUMB_FINGERTIP_INDEX = 4
GESTURE_POINTER_FINGERTIP_INDEX = 8
GESTURE_MIDDLE_FINGERTIP_INDEX = 12
GESTURE_RING_FINGERTIP_INDEX = 16
GESTURE_PINKY_FINGERTIP_INDEX = 20

LEFT_HORN_INDEX = 332
RIGHT_HORN_INDEX = 103

LEFT_EYE_OUTER_INDEX = 263
LEFT_EYE_INNER_INDEX = 362
LEFT_EYE_CENTER_INDEX = 473
RIGHT_EYE_OUTER_INDEX = 33
RIGHT_EYE_INNER_INDEX = 133
RIGHT_EYE_CENTER_INDEX = 468

MOUTH_RIGHT_SIDE_INDEX = 61
MOUTH_LEFT_SIDE_INDEX = 291


class TrackerDataMapper:
    __wink_tracker = WinkTracker()


    def get_fingertips_coordinates(self, hand_landmarks) -> Sequence[cv2.typing.Point]:
        if len(hand_landmarks) > 0:
            coordinates = []
            for hand_landmark in hand_landmarks:
                coordinates.extend(
                    [
                        hand_landmark[GESTURE_THUMB_FINGERTIP_INDEX],
                        hand_landmark[GESTURE_POINTER_FINGERTIP_INDEX],
                        hand_landmark[GESTURE_MIDDLE_FINGERTIP_INDEX],
                        hand_landmark[GESTURE_RING_FINGERTIP_INDEX],
                        hand_landmark[GESTURE_PINKY_FINGERTIP_INDEX]
                    ]
                )
            return coordinates


    def get_horns_coordinates(self, face_landmarks) -> Sequence[cv2.typing.Point]:
        if len(face_landmarks) > 0:
            face_landmark = face_landmarks[0]
            return [
                face_landmark[LEFT_HORN_INDEX],
                face_landmark[RIGHT_HORN_INDEX]
            ]


    def get_eyes_coordinates(self, face_landmarks) -> tuple[bool, cv2.typing.Point]:
        if len(face_landmarks) > 0:
            face_landmark = face_landmarks[0]
            return [
                (self.__wink_tracker.check_left_eye_for_wink(face_landmark), face_landmark[LEFT_EYE_CENTER_INDEX]),
                (self.__wink_tracker.check_right_eye_for_wink(face_landmark), face_landmark[RIGHT_EYE_CENTER_INDEX]),
            ]


    def get_mouth_coordinates(self, face_landmarks) -> Sequence[cv2.typing.Point]:
        if len(face_landmarks) > 0:
            face_landmark = face_landmarks[0]
            return [
                face_landmark[MOUTH_LEFT_SIDE_INDEX],
                face_landmark[MOUTH_RIGHT_SIDE_INDEX]
            ]
