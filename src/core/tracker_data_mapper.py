import cv2

from typing import Sequence

GESTURE_THUMB_FINGERTIP_INDEX = 4
GESTURE_POINTER_FINGERTIP_INDEX = 8
GESTURE_MIDDLE_FINGERTIP_INDEX = 12
GESTURE_RING_FINGERTIP_INDEX = 16
GESTURE_PINKY_FINGERTIP_INDEX = 20

POSE_NOSE_INDEX = 0

POSE_LEFT_EYE_INNER_INDEX = 1
POSE_LEFT_EYE_CENTER_INDEX = 2
POSE_LEFT_EYE_OUTER_INDEX = 3
POSE_RIGHT_EYE_INNER_INDEX = 4
POSE_RIGHT_EYE_CENTER_INDEX = 5
POSE_RIGHT_EYE_OUTER_INDEX = 6

POSE_MOUTH_LEFT_SIDE_INDEX = 9
POSE_MOUTH_RIGHT_SIDE_INDEX = 10


class TrackerDataMapper:
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


    def get_nose_coordinates(self, pose_landmarks) -> Sequence[cv2.typing.Point]:
        if len(pose_landmarks) > 0:
            pose_landmark = pose_landmarks[0]
            return [
                pose_landmark[POSE_NOSE_INDEX]
            ]


    def get_eyes_coordinates(self, pose_landmarks) -> Sequence[cv2.typing.Point]:
        if len(pose_landmarks) > 0:
            pose_landmark = pose_landmarks[0]
            return [
                # pose_landmark[POSE_LEFT_EYE_INNER_INDEX],
                pose_landmark[POSE_LEFT_EYE_CENTER_INDEX],
                # pose_landmark[POSE_LEFT_EYE_OUTER_INDEX],
                # pose_landmark[POSE_RIGHT_EYE_INNER_INDEX],
                pose_landmark[POSE_RIGHT_EYE_CENTER_INDEX],
                # pose_landmark[POSE_RIGHT_EYE_OUTER_INDEX]
            ]


    def get_mouth_coordinates(self, pose_landmarks) -> Sequence[cv2.typing.Point]:
        if len(pose_landmarks) > 0:
            pose_landmark = pose_landmarks[0]
            return [
                pose_landmark[POSE_MOUTH_LEFT_SIDE_INDEX],
                pose_landmark[POSE_MOUTH_RIGHT_SIDE_INDEX]
            ]
