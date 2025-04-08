from Environments.AbstractEnv import AbstractEnv
from Environments.CliffVisualizer import CliffVisualizer


class CliffEnv(AbstractEnv):

    def __init__(self, width: int = 6):
        self.__height = 4
        self.__width = width
        self.__initial_state = (0, 0)
        self.__goal_state = (0, self.__width - 1)
        self.__visualizer = CliffVisualizer(self.__height, self.__width)
        self.__state = None
        self.__timestep = None
        self.__max_timestep = 100000

    @property
    def action_space(self):
        return [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def reset(self):
        self.__timestep = 0
        self.__state = self.__initial_state
        return self.__state

    def step(self, action):
        self.__timestep += 1
        self.__update_current_state(action)
        reward = -1
        if self.__is_agent_at_the_cliff():
            reward = -100
            self.__state = self.__initial_state
        done = self.__is_game_over()
        return self.__state, reward, done

    def __update_current_state(self, action):
        next_pos_row = self.__get_next_position_row(action)
        next_pos_col = self.__get_next_position_column(action)
        self.__state = next_pos_row, next_pos_col

    def __get_next_position_row(self, action):
        next_row = self.__state[0] + action[0]
        return self.__move_value_to_valid_range(next_row, 0, self.__height - 1)

    @staticmethod
    def __move_value_to_valid_range(value, min_value, max_value):
        return max(min(value, max_value), min_value)

    def __get_next_position_column(self, action):
        next_column = self.__state[1] + action[1]
        return self.__move_value_to_valid_range(next_column, 0, self.__width - 1)

    def __is_agent_at_the_cliff(self):
        return self.__state[0] == 0 and self.__state[1] not in [0, self.__width - 1]

    def __is_game_over(self):
        return self.__is_agent_at_goal_state() or self.__is_timestep_limit_reached()

    def __is_agent_at_goal_state(self):
        return self.__state == self.__goal_state

    def __is_timestep_limit_reached(self):
        return self.__timestep > self.__max_timestep

    def show(self):
        self.__visualizer.show(self.__state)
