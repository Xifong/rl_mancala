from typing import Any
import numpy as np
import gymnasium as gym
import itertools as it


class Mancala(gym.Env):
    def __init__(self):
        self.metadata = {"render_modes": ["None"]}
        self.render_mode = None

        # Initialise env state. If custom seed needed, reset should be called again with that.
        self.reset()

        # TODO: Switch to dict later if it seems better
        self.observation_space = gym.spaces.MultiDiscrete(np.array([49] * 14))

    def _get_obs(self) -> list[int]:
        return (
            self._player_side
            + [self._player_score]
            + self._opponent_side
            + [self._opponent_score]
        )

    def _make_valid_action(self, action: int):
        gems = self._player_side[action]
        self._player_side[action] = 0

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

        add_to = [self._player_side, [self._player_score], self._opponent_side]

        for i, j in sequence:
            add_to[i][j] += 1

        self._player_side, self._player_score, self._opponent_side = add_to

    def step(self, action: int) -> tuple[list[int], float, bool, bool, Any]:
        if not self._is_action_valid(action):
            return self._get_obs(), -1, False, False, None

        self._make_valid_action(action)

        return self._get_obs(), 0, False, False, None

    def _is_action_valid(self, action: int) -> bool:
        return self._player_side[action] > 0

    def reset(self, seed: int = None, options: Any = None):
        super().reset(seed=seed)

        self._player_side = [4] * 6
        self._opponent_side = [4] * 6
        self._player_score = 0
        self._opponent_score = 0
        # Decide who starts at random
        self._is_player_turn = True if self.np_random.integers(low=0, high=2) else False

    def set_opponent_policy(self, policy: Any):
        self._opponent_policy = policy

    def __str__(self) -> str:
        to_play = "player" if self._is_player_turn else "opponent"
        return f"{self._player_side}, {self._player_score}=, {self._opponent_side}, {self._opponent_score=}, {to_play} to play"


if __name__ == "__main__":
    mancala = Mancala()
    print(mancala)
    mancala.step(5)
    print(mancala)
