
class GridLocations:

    def __init__(self, grid_size: int):
        self.__grid_size = grid_size
        self.__actions = ["down", "up", "left", "right"]
        self.__location_ids = []
        self.__location_id_to_coordinates = []
        self.__coordinates_to_location_id = {}
        self.__create_locations()

    def __create_locations(self) -> None:
        state_id = 0
        for coord_i in range(self.__grid_size):
            for coord_j in range(self.__grid_size):
                self.__add_new_location(state_id, coord_i, coord_j)
                state_id += 1

    def __add_new_location(self, state_id: int, coord_i: int, coord_j: int) -> None:
        self.__location_ids.append(state_id)
        self.__location_id_to_coordinates.append((coord_i, coord_j))
        self.__coordinates_to_location_id[(coord_i, coord_j)] = state_id

    def get_all_locations(self) -> list:
        return self.__location_ids

    @property
    def top_left_location(self) -> int:
        return self.__location_ids[0]

    @property
    def bottom_right_location(self) -> int:
        return self.__location_ids[-1]

    def get_actions(self) -> list[str]:
        return self.__actions

    def get_next_location(self, current_location: int, action: str) -> int:
        coord_i, coord_j = self._get_coordinates(current_location)
        if action == "left" and coord_j > 0:
            coord_j -= 1
        if action == "right" and coord_j < self.__grid_size - 1:
            coord_j += 1
        if action == "up" and coord_i > 0:
            coord_i -= 1
        if action == "down" and coord_i < self.__grid_size - 1:
            coord_i += 1
        return self.get_location_id(coord_i, coord_j)

    def _get_coordinates(self, location_id: int) -> (int, int):
        return self.__location_id_to_coordinates[location_id]

    def get_location_id(self, coord_i: int, coord_j: int) -> int:
        return self.__coordinates_to_location_id[(coord_i, coord_j)]
