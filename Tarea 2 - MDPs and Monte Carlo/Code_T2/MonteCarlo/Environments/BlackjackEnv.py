import random

from Environments.AbstractEnv import AbstractEnv


class BlackjackEnv(AbstractEnv):

    def __init__(self):
        self.__player = None
        self.__dealer = None
        self.__done = None

    def reset(self):
        self.__player = self.__draw_until_points_are_at_least(12)
        self.__dealer = self.__draw_until_points_are_at_least(17)
        self.__done = False
        return self.__get_state()

    def __draw_until_points_are_at_least(self, min_num_of_points: int):
        hand = []
        while self.__count_points(hand) < min_num_of_points:
            hand.append(self.__draw_card())
        return hand

    def __count_points(self, hand):
        points = sum(hand)
        if self.__has_usable_as(hand):
            points += 10
        return points

    @staticmethod
    def __has_usable_as(hand):
        return 1 in hand and sum(hand) < 12

    @staticmethod
    def __draw_card():
        # The "jack", "queen", and "king" are worth 10 points -- which is why 10 appears 3 times in this list.
        return random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10])

    def __get_state(self):
        player_points = self.__count_points(self.__player)
        player_has_usable_as = self.__has_usable_as(self.__player)
        dealer_points = self.__dealer[0]  # We only have to show the dealer's first card
        return player_points, player_has_usable_as, dealer_points

    def step(self, action):
        assert action in self.action_space, f"Invalid action: {action}."
        if action == 'hit':
            self.__player.append(self.__draw_card())
            state = self.__get_state()
            done = self.__count_points(self.__player) > 21
            reward = -1.0 if done else 0.0
        else:
            state = self.__get_terminal_state()
            done = True
            reward = self.__get_reward_end_game()
        self.__done = done
        return state, reward, done

    def __get_terminal_state(self):
        player_points = self.__count_points(self.__player)
        player_has_usable_as = self.__has_usable_as(self.__player)
        dealer_points = self.__count_points(self.__dealer)  # terminal state includes the dealer's points
        return player_points, player_has_usable_as, dealer_points

    def __get_reward_end_game(self) -> float:
        players_points = self.__count_points(self.__player)
        dealers_points = self.__count_points(self.__dealer)
        if players_points == dealers_points:
            return 0.0
        if dealers_points < players_points < 22:
            return 1.0
        if players_points < dealers_points < 22:
            return -1.0
        assert dealers_points > 21, f"If we get here, the dealer should have > 21 points. But he has {dealers_points}"
        return 1.0

    @property
    def action_space(self):
        return ["hit", "stick"]

    def show(self):
        print(f"Player: {self.__player} ({self.__count_points(self.__player)} points)")
        if self.__done:
            print(f"Dealer: {self.__dealer} ({self.__count_points(self.__dealer)} points)")
        else:
            print(f"Dealer: {self.__dealer[0]}")
