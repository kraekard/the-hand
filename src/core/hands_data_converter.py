import cv2
import numpy

from typing import Sequence

WRIST_CODE = 0
THUMB_FINGERTIP_CODE = 4
INDEX_FINGERTIP_CODE = 8
MIDDLE_FINGERTIP_CODE = 12
RING_FINGERTIP_CODE = 16
PINKY_FINGERTIP_CODE = 20


class HandsDataConverter:
    def get_fingertips_coordinates(self, hand_landmarks) -> Sequence[cv2.typing.Point]:
        if len(hand_landmarks) > 0:
            coordinates = []
            for hand_landmark in hand_landmarks:
                coordinates.extend(
                    [
                        hand_landmark[THUMB_FINGERTIP_CODE],
                        hand_landmark[INDEX_FINGERTIP_CODE],
                        hand_landmark[MIDDLE_FINGERTIP_CODE],
                        hand_landmark[RING_FINGERTIP_CODE],
                        hand_landmark[PINKY_FINGERTIP_CODE]
                    ]
                )
            return coordinates
