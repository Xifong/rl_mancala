from typing import Any, Callable
import numpy as np
import gymnasium as gym
import itertools as it


def random_opponent_policy(seed: int, observation: list[int]) -> int:
    # Depends on the opponent side being the last (but one) six elements of the observation
    opponent_side = observation[-7:-1]
    assert sum(opponent_side) > 0, "Opponent has no valid moves"

    random_action = np.random.randint(0, 6)
    while opponent_side[random_action] == 0:
        random_action = np.random.randint(0, 6)

    return random_action


class Mancala(gym.Env):
    def __init__(
        self, seed: int = None, opponent_policy: Callable = random_opponent_policy
    ):
        self.metadata = {"render_modes": ["None"]}
        self.render_mode = None

        self._history = []
        self._opponent_policy = opponent_policy

        # Initialise env state. If custom seed needed, reset should be called again with that.
        self.reset(seed=seed)

        # TODO: Switch to dict later if it seems better
        self.observation_space = gym.spaces.MultiDiscrete(np.array([49] * 14))

    def _get_obs(self) -> list[int]:
        return (
            self._player_side
            + [self._player_score]
            + self._opponent_side
            + [self._opponent_score]
        )

    def _make_valid_action(
        self,
        action: int,
        entity_side: list[int],
        entity_score: int,
        entity_opponent_side: list[int],
    ) -> tuple[list[int], int, list[int]]:
        gems = entity_side[action]
        entity_side[action] = 0

        first_add_to = action + 1
        # Generate a sequence of consecutive index positions into a
        # [[player_side] (1*6), [player_mancala] (1*1), [opponent_side] (1*6)] list of lists
        sequence = list(
            it.islice(
                zip(
                    it.cycle([0] * 6 + [1] + [2] * 6),
                    it.cycle(list(range(6)) + [0] + list(range(6))),
                ),
                gems + first_add_to,
            )
        )[first_add_to:]

        current_state = [entity_side, [entity_score], entity_opponent_side]

        for i, j in sequence:
            current_state[i][j] += 1

        # Final gem was placed into the entities' Mancala
        plays_again = sequence[-1][0] == 1

        # flatten current_state by extracting the entities score
        return current_state[0], current_state[1][0], current_state[2], plays_again

    def _make_entity_action(self, action: int, is_player: bool) -> bool:
        if is_player:
            self._player_side, self._player_score, self._opponent_side, plays_again = (
                self._make_valid_action(
                    action, self._player_side, self._player_score, self._opponent_side
                )
            )
        else:
            (
                self._opponent_side,
                self._opponent_score,
                self._player_side,
                plays_again,
            ) = self._make_valid_action(
                action, self._opponent_side, self._opponent_score, self._player_side
            )

        self._record()

        return plays_again

    def _is_game_over(self) -> bool:
        if sum(self._player_side) == 0 or sum(self._opponent_side) == 0:
            return True

        return False

    def _get_player_reward(self) -> float:
        if self._player_score > self._opponent_score:
            return 1.0

        if self._player_score == self._opponent_score:
            return 0.5

        return 0

    def _opponent_takes_turn_if_not_game_over(self):
        plays_again = True
        while plays_again and not self._is_game_over():
            plays_again = self._make_entity_action(
                self._opponent_policy(self._seed, self._get_obs()), is_player=False
            )

    def step(self, action: int) -> tuple[list[int], float, bool, bool, Any]:
        assert (
            not self._is_game_over()
        ), "Attempting to step game even though game is over"

        # No state change happens on invalid moves, but a negative reward is received
        # Hopefully this will be enough to learn to produce only valid moves
        if not self._is_action_valid(action):
            return self._get_obs(), -1.0, False, False, None

        player_plays_again = self._make_entity_action(action, is_player=True)

        if not player_plays_again:
            self._opponent_takes_turn_if_not_game_over()

        return (
            self._get_obs(),
            self._get_player_reward() if self._is_game_over() else 0,
            self._is_game_over(),
            False,
            None,
        )

    def _is_action_valid(self, action: int) -> bool:
        return self._player_side[action] > 0

    def _set_seed(self, seed: int):
        self._seed = seed
        super().reset(seed=seed)
        np.random.seed(seed)

    def reset(self, seed: int = None, options: Any = None):
        self._set_seed(seed)

        self._player_side = [4] * 6
        self._opponent_side = [4] * 6
        self._player_score = 0
        self._opponent_score = 0

        self._record()

        # Decide who starts at random
        _does_opponent_start = True if self.np_random.integers(low=0, high=2) else False
        if _does_opponent_start:
            self._make_entity_action(
                self._opponent_policy(seed, self._get_obs()), is_player=False
            )

    def set_opponent_policy(self, policy: Any):
        self._opponent_policy = policy

    def _record(self):
        self._history.append(self._state_str)

    @property
    def _history_str(self) -> str:
        return "\n\t" + "\n\t".join(self._history)

    @property
    def _full_str(self) -> str:
        legend = "[p_side],[p_score],[o_side],[o_score]"
        current_state = f"{self._player_side}, {self._player_score}, {self._opponent_side}, {self._opponent_score}"
        return f"legend: {legend}\ncurrent_state: {current_state}\nhistory: {self._history_str}\n"

    @property
    def _state_str(self) -> str:
        return f"{self._player_side}, {self._player_score}, {self._opponent_side}, {self._opponent_score}"

    def __str__(self) -> str:
        return f"[p_side],[p_score],[o_side],[o_score]: {self._state_str}"


if __name__ == "__main__":
    mancala = Mancala(seed=42)
    mancala.step(5)
    mancala.step(0)
    mancala.step(2)
    mancala.step(3)
    mancala.step(4)
    mancala.step(1)
    mancala.step(5)
    mancala.step(0)
    mancala.step(2)
    mancala.step(3)
    mancala.step(4)
    mancala.step(1)
    mancala.step(5)
    mancala.step(0)
    mancala.step(2)
    mancala.step(3)
    mancala.step(4)
    mancala.step(1)
    mancala.step(5)
    mancala.step(0)
    mancala.step(2)
    mancala.step(3)
    mancala.step(4)
    mancala.step(1)
    mancala.step(5)
    mancala.step(0)
    mancala.step(2)
    mancala.step(3)
    mancala.step(4)
    mancala.step(1)
    mancala.step(5)
    mancala.step(0)
    mancala.step(2)
    mancala.step(3)
    mancala.step(4)
    mancala.step(1)
    mancala.step(5)
    mancala.step(0)
    mancala.step(2)
    mancala.step(3)
    mancala.step(4)
    mancala.step(1)
    print(mancala._full_str)
