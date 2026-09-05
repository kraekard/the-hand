from utils.number_utils import NumberUtils


class ColorUtils:
    @staticmethod
    def generate_random_bgr_code():
        return (
            NumberUtils.get_random_int(0, 255),
            NumberUtils.get_random_int(0, 255),
            NumberUtils.get_random_int(0, 255)
        )
