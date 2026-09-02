import random


class NumberUtils:
    @staticmethod
    def get_random_int(floor: int, roof: int) -> int:
        return random.randint(floor, roof)


    @staticmethod
    def prevent_zero(value: int) -> int:
        if value == 0:
            return 1
        return value
