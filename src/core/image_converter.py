import cv2

class ImageConverter:
    def to_rgb_image(self, base_image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        return cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)


    def to_bw_image(self, base_image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        grayed_base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(grayed_base_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
