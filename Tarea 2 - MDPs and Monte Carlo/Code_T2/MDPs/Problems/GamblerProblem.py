from Problems.AbstractProblem import AbstractProblem


class GamblerProblem(AbstractProblem):

    def __init__(self, prob_head: float = 0.4):
        self.__min_state = 0
        self.__max_state = 100
        self.__prob_head = prob_head

    @property
    def states(self) -> list[int]:
        return list(range(self.__max_state + 1))

    def get_initial_state(self) -> int:
        return (self.__max_state - self.__min_state) // 2

    def get_available_actions(self, state: int) -> list[int]:
        if self.is_terminal(state):
            return []
        max_stake = min(state, 100 - state)
        return list(range(1, max_stake + 1))

    def is_terminal(self, state: int) -> bool:
        return state in [self.__min_state, self.__max_state]

    def get_transitions(self, state: int, action: int) -> list[(float, int, float)]:
        if self.is_terminal(state):
            return [(1.0, (state, 0.0))]
        head_outcome = self.__get_head_outcome(state, action)
        tail_outcome = self.__get_tail_outcome(state, action)
        return [head_outcome, tail_outcome]

    def __get_head_outcome(self, state: int, action: int) -> (int, float):
        next_state = min(state + action, self.__max_state)
        reward = 1.0 if next_state == self.__max_state else 0.0
        return self.__prob_head, next_state, reward

    def __get_tail_outcome(self, state: int, action: int) -> (int, float):
        next_state = max(state - action, self.__min_state)
        return 1.0 - self.__prob_head, next_state, 0.0

    def show(self, state):
        print(state)
