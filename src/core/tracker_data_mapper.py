import cv2
import numpy

from typing import Sequence

from core.wink_tracker import WinkTracker


GESTURE_WRIST_INDEX = 0
GESTURE_THUMB_FINGERTIP_INDEX = 4
GESTURE_POINTER_FINGERTIP_INDEX = 8
GESTURE_MIDDLE_FINGERTIP_INDEX = 12
GESTURE_RING_FINGERTIP_INDEX = 16
GESTURE_PINKY_FINGERTIP_INDEX = 20
GESTURE_THUMB_KNUCKLE_INDEX = 1
GESTURE_POINTER_KNUCKLE_INDEX = 5
GESTURE_MIDDLE_KNUCKLE_INDEX = 9
GESTURE_RING_KNUCKLE_INDEX = 13
GESTURE_PINKY_KNUCKLE_INDEX = 17
KNUCKLES_UPPER_INDEXES = [GESTURE_POINTER_KNUCKLE_INDEX, GESTURE_MIDDLE_KNUCKLE_INDEX, GESTURE_RING_KNUCKLE_INDEX, GESTURE_PINKY_KNUCKLE_INDEX]

LEFT_HORN_INDEX = 332
RIGHT_HORN_INDEX = 103

LEFT_EYE_CENTER_INDEX = 473
RIGHT_EYE_CENTER_INDEX = 468

MOUTH_UPPER_LIP_INDEXES = [61, 40, 39, 37, 0, 267, 269, 270, 291]
MOUTH_LOWER_LIP_INDEXES = [61, 91, 181, 84, 17, 314, 405, 321, 291]


class TrackerDataMapper:
    __wink_tracker = WinkTracker()


    def get_fists_center_coordinates(self, hand_landmarks) -> Sequence[cv2.typing.Point]:
        if len(hand_landmarks) > 0:
            coordinates = []
            for hand_landmark in hand_landmarks:
                hand_landmark = numpy.array(hand_landmark)
                wrist_coords = hand_landmark[GESTURE_WRIST_INDEX]
                upper_coords_x_sum = 0
                upper_coords_y_sum = 0
                upper_coords_z_sum = 0
                for index in KNUCKLES_UPPER_INDEXES:
                    upper_coords_x_sum += hand_landmark[index].x
                    upper_coords_y_sum += hand_landmark[index].y
                    upper_coords_z_sum += hand_landmark[index].z
                coordinates.append(
                    (
                        (upper_coords_x_sum + wrist_coords.x * 4) / 8,
                        (upper_coords_y_sum + wrist_coords.y * 4) / 8,
                        (upper_coords_z_sum + wrist_coords.z * 4) / 8,
                    )
                )
            return coordinates


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


    def get_mouth_center_coordinates(self, face_landmarks) -> Sequence[cv2.typing.Point]:
        coords: cv2.typing.Point = []
        if len(face_landmarks) > 0:
            face_landmarks = numpy.array(face_landmarks[0])
            upper_coords = face_landmarks[numpy.array(MOUTH_UPPER_LIP_INDEXES)]
            lower_coords = face_landmarks[numpy.array(MOUTH_LOWER_LIP_INDEXES)]
            for index in range(len(upper_coords)):
                coords.append(
                    [
                        upper_coords[index].x,
                        (upper_coords[index].y + lower_coords[index].y) / 2,
                        upper_coords[index].z
                    ]
                )
        return coords
