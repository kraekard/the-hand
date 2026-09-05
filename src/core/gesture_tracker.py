import global_variables


class GestureTracker():
    def check_for_closed_fist(self, gestures):
        if gestures is not None and len(gestures) > 0:
            if any(gesture.category_name and gesture.category_name.upper() == 'CLOSED_FIST' for gesture in gestures[0]):
                global_variables.is_fist_closed = True
            elif global_variables.is_fist_closed:
                global_variables.is_fist_closed = False


    def is_fist_closed(self) -> bool:
        return global_variables.is_fist_closed
