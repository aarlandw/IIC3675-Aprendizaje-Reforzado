class CliffVisualizer:

    def __init__(self, height, width):
        self.__height = height
        self.__width = width
        self.__state = None

    def show(self, state):
        self.__state = state
        self.__create_empty_map()
        self.__add_cliff()
        self.__add_stating_and_goal_positions()
        self.__add_agent()
        self.__show_map()

    def __create_empty_map(self):
        self.__map = [[" " for _ in range(self.__width)] for _ in range(self.__height)]

    def __add_cliff(self):
        for j in range(self.__width):
            self.__map[0][j] = "C"

    def __add_stating_and_goal_positions(self):
        self.__map[0][0] = "S"
        self.__map[0][self.__width - 1] = "G"

    def __add_agent(self):
        i, j = self.__state
        self.__map[i][j] = "A"

    def __show_map(self):
        print("+" + self.__width * "-" + "+")
        for row in self.__map[::-1]:
            print("|" + "".join(row) + "|")
        print("+" + self.__width * "-" + "+")
