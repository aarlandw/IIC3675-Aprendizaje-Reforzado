from Problems.AbstractProblem import AbstractProblem
from Problems.GridLocations import GridLocations


class GridProblem(AbstractProblem):

    def __init__(self, grid_size: int = 5):
        self.__grid_size = grid_size
        self.__grid_locations = GridLocations(grid_size)

    @property
    def states(self):
        return self.__grid_locations.get_all_locations()

    def get_initial_state(self) -> int:
        coord_i = self.__grid_size // 2
        coord_j = self.__grid_size // 2
        return self.__grid_locations.get_location_id(coord_i, coord_j)

    def get_available_actions(self, state):
        return self.__grid_locations.get_actions()

    def is_terminal(self, state):
        return state in [self.__grid_locations.top_left_location, self.__grid_locations.bottom_right_location]

    def get_transitions(self, state, action):
        next_state = self.__grid_locations.get_next_location(state, action)
        reward = -1.0
        prob = 1.0
        return [(prob, next_state, reward)]

    def show(self, state):
        location_id = 0
        print((self.__grid_size + 2)*"X")
        for i in range(self.__grid_size):
            print("X", end="")
            for j in range(self.__grid_size):
                if location_id == state:
                    print("A", end="")
                elif self.is_terminal(location_id):
                    print("G", end="")
                else:
                    print(" ", end="")
                location_id += 1
            print("X")
        print((self.__grid_size + 2)*"X")

