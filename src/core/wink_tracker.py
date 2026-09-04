import numpy


RIGHT_EYE_EAR_INDEXES = [33, 159, 158, 133, 153, 145]
LEFT_EYE_EAR_INDEXES = [362, 380, 374, 263, 386, 385]


class WinkTracker():
    def __init__(self, ear_threshold: float = 0.175):
        self.__ear_threshold = ear_threshold


    def check_right_eye_for_wink(self, face_landmarks) -> bool:
        return self.__get_eye_aspect_ratio(RIGHT_EYE_EAR_INDEXES, face_landmarks) < self.__ear_threshold


    def check_left_eye_for_wink(self, face_landmarks) -> bool:
        return self.__get_eye_aspect_ratio(LEFT_EYE_EAR_INDEXES, face_landmarks) < self.__ear_threshold

    
    def __get_eye_aspect_ratio(self, indexes, landmarks):
        if landmarks is None or len(landmarks) < 1:
            return 1

        A = numpy.linalg.norm(self.__extract_coords_only(landmarks[indexes[1]]) - self.__extract_coords_only(landmarks[indexes[5]]))
        B = numpy.linalg.norm(self.__extract_coords_only(landmarks[indexes[2]]) - self.__extract_coords_only(landmarks[indexes[4]]))
        C = numpy.linalg.norm(self.__extract_coords_only(landmarks[indexes[0]]) - self.__extract_coords_only(landmarks[indexes[3]]))
        return (A + B) / (2.0 * C)


    def __extract_coords_only(self, coords):
        return numpy.array([coords.x, coords.y, coords.z])
