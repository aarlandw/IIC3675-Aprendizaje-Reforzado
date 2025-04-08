from abc import ABC, abstractmethod


class AbstractProblem(ABC):

    @property
    @abstractmethod
    def states(self):
        """
        :return: a list containing all the states of the problem -- including terminal states.
        """
        pass

    @abstractmethod
    def get_initial_state(self):
        """
        :return: returns the initial states.
        """
        pass

    @abstractmethod
    def get_available_actions(self, state):
        """
        :param state: is a non-terminal state of the problem.
        :return: a list containing the available actions from "state".
        """
        pass

    @abstractmethod
    def is_terminal(self, state):
        """
        :param state: is a state of the problem.
        :return: True iff "state" is a terminal state.
        """
        pass

    @abstractmethod
    def get_transitions(self, state, action):
        """
        Returns the probabilities for the next states and rewards when executing "action" from "state".
        :param state: a state of the problem.
        :param action: a valid action.
        :return: a list of tuples. Each tuple contains three elements: (probability, next state, reward), where
        probability is p(next state, reward | state, action).
        """
        pass

    @abstractmethod
    def show(self, state):
        """
        Shows "state" in console
        """
        pass