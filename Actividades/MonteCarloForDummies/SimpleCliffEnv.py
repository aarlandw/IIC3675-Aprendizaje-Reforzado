from AbstractEnv import AbstractEnv


class SimpleCliffEnv(AbstractEnv):

    def __init__(self, initial_state=(1, 0)):
        self.__height = 4
        self.__width = 4
        self.__initial_state = initial_state
        self.__goal_state = (1, self.__width - 1)
        self.__state = None

    @property
    def action_space(self):
        return ["down", "up", "right", "left"]
    @property
    def initial_state(self):
        return self.__initial_state
    def reset(self):
        self.__state = self.__initial_state
        return self.__state

    def step(self, action):
        self.__perform_action(action)
        at_goal = self.__state == self.__goal_state
        at_cliff = self.__state[0] == 0
        done = at_goal or at_cliff
        reward = 1.0 if at_goal else 0.0
        return self.__state, reward, done

    def __perform_action(self, action):
        row, col = self.__get_new_raw_position(action)
        self.__update_state(row, col)

    def __get_new_raw_position(self, action):
        row, col = self.__state
        if action == "down":
            row += 1
        if action == "up":
            row -= 1
        if action == "right":
            col += 1
        if action == "left":
            col -= 1
        return row, col

    def __update_state(self, new_row, new_col):
        new_row = self.__adjust_limit(new_row, self.__height - 1)
        new_col = self.__adjust_limit(new_col, self.__width - 1)
        self.__state = (new_row, new_col)

    @staticmethod
    def __adjust_limit(value, max_value):
        return max([min([value, max_value]), 0])

    def show(self):
        print()
        print("X" * (self.__width + 2))
        for i in range(self.__height):
            print("X", end="")
            for j in range(self.__width):
                location = (i, j)
                if location == self.__state:
                    print("A", end="")
                elif location == self.__goal_state:
                    print("G", end="")
                elif location[0] == 0:
                    print("C", end="")
                else:
                    print(" ", end="")
            print("X")
        print("X" * (self.__width + 2))