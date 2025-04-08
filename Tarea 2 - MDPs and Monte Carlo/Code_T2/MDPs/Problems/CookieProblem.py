from Problems.AbstractProblem import AbstractProblem
from Problems.GridLocations import GridLocations


class CookieProblem(AbstractProblem):

    def __init__(self, grid_size: int = 3):
        self.__grid_size = grid_size
        self.__grid_locations = GridLocations(grid_size)

    @property
    def states(self) -> list[(int, int)]:
        locations = self.__grid_locations.get_all_locations()
        states = []
        for agent_location in locations:
            for cookie_location in locations:
                if agent_location != cookie_location:
                    states.append((agent_location, cookie_location))
        return states

    def get_initial_state(self) -> (int, int):
        agent_location = self.__grid_locations.top_left_location
        cookie_location = self.__grid_locations.bottom_right_location
        return agent_location, cookie_location

    def get_available_actions(self, state: (int, int)) -> list[str]:
        return self.__grid_locations.get_actions()

    def is_terminal(self, state: (int, int)) -> bool:
        return False

    def get_transitions(self, state: (int, int), action: str) -> list[(float, (int, int), float)]:
        agent_location, cookie_location = state
        agent_next_location = self.__grid_locations.get_next_location(agent_location, action)
        if agent_next_location != cookie_location:
            return [(1.0, (agent_next_location, cookie_location), 0.0)]
        return self._get_outcomes_when_agent_reaches_the_cookie(agent_next_location)

    def _get_outcomes_when_agent_reaches_the_cookie(self, agent_location: int) -> list[(float, (int, int), float)]:
        locations = self.__grid_locations.get_all_locations()
        prob = 1.0 / (len(locations) - 1)
        reward = 1.0
        outcomes = []
        for cookie_next_location in locations:
            if agent_location != cookie_next_location:
                outcomes.append((prob, (agent_location, cookie_next_location), reward))
        return outcomes

    def show(self, state):
        agent_location, cookie_location = state
        location_id = 0
        print("X"*(self.__grid_size + 2))
        for i in range(self.__grid_size):
            print("X", end="")
            for j in range(self.__grid_size):
                if location_id == agent_location:
                    print("A", end="")
                elif location_id == cookie_location:
                    print("C", end="")
                else:
                    print(" ", end="")
                location_id += 1
            print("X")
        print("X"*(self.__grid_size + 2))
