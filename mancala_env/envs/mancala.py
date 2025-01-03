from typing import Any, Callable
import numpy as np
import gymnasium as gym
import itertools as it
from enum import Enum

from stable_baselines3 import DQN


class GameOutcome(Enum):
    WIN = "win"
    DRAW = "draw"
    LOSE = "lose"


def random_opponent_policy(seed: int, observation: np.array) -> int:
    # Depends on the opponent side being the last (but one) six elements of the observation
    opponent_side = observation[-7:-1]
    assert sum(opponent_side) > 0, "Opponent has no valid moves"

    random_action = np.random.randint(0, 6)
    while opponent_side[random_action] == 0:
        random_action = np.random.randint(0, 6)

    return random_action


def saved_opponent_policy(seed: int, observation: np.array) -> int:
    opponent_side = observation[-7:-1]
    assert sum(opponent_side) > 0, "Opponent has no valid moves"

    model_path = "./saved_models/2024-12-29_09-11-33/"
    # Can probably speed up training by not repeating this loading!
    model = DQN.load(f"{model_path}/best_model")

    action, _ = model.predict(observation, deterministic=True)

    # Use random if opponent model is trying to play invalid move
    while opponent_side[action] == 0:
        action = np.random.randint(0, 6)

    return int(action)


def generate_action_sequence(action: int, total_gems: int) -> list[tuple[int, int]]:
    """
    Generate a sequence of consecutive index positions as a flattened
    [player_side (1*6), player_mancala (1*1), opponent_side (1*6)] list of tuples
    """
    first_add_to = action + 1
    return list(
        it.islice(
            zip(
                it.cycle([0] * 6 + [1] + [2] * 6),
                it.cycle(list(range(6)) + [0] + list(range(6))),
            ),
            first_add_to,  # start
            first_add_to + total_gems,  # stop
        )
    )


def make_valid_action(
    action: int,
    entity_side: list[int],
    entity_score: int,
    entity_opponent_side: list[int],
) -> tuple[list[int], int, list[int]]:
    def make_capture(capture_from_index: int):
        # The sides are arranged going around so to get the opposite spot,
        # you have have to take away from the maximum action index i.e. 5 - action.
        capture_to_index = 5 - capture_from_index
        total = entity_side[capture_from_index] + entity_opponent_side[capture_to_index]
        current_state[0][capture_from_index] = 0
        current_state[2][capture_to_index] = 0
        current_state[1][0] += total

    gems = entity_side[action]
    sequence = generate_action_sequence(action, gems)

    entity_side[action] = 0
    current_state = [entity_side, [entity_score], entity_opponent_side]

    for i, j in sequence:
        current_state[i][j] += 1

    # Final gem was placed into the entities' Mancala
    does_play_again = i == 1

    # Final gem was placed into an empty position on the entities' side
    does_capture = i == 0 and current_state[i][j] == 1
    if does_capture:
        make_capture(capture_from_index=j)

    # flatten current_state by extracting the entities score
    return current_state[0], current_state[1][0], current_state[2], does_play_again


class MancalaEnv(gym.Env):
    def __init__(
        self,
        seed: int = None,
        opponent_policy: Callable = saved_opponent_policy,
    ):
        self.metadata = {"render_modes": ["None"]}
        self.render_mode = None

        self._opponent_policy = opponent_policy

        # Initialise env state. If custom seed needed, reset should be called again with that.
        self.reset(seed=seed)

        # TODO: Switch to dict later if it seems better
        self.observation_space = gym.spaces.MultiDiscrete(np.array([49] * 14))
        self.action_space = gym.spaces.Discrete(6)

    def _get_obs(self) -> np.array:
        return np.array(
            self._player_side
            + [self._player_score]
            + self._opponent_side
            + [self._opponent_score]
        )

    def _get_info(self) -> dict:
        info = {}
        info["is_success"] = self._game_outcome == GameOutcome.WIN
        info["is_draw"] = self._game_outcome == GameOutcome.DRAW
        info["is_loss"] = self._game_outcome == GameOutcome.LOSE

        return info

    def _make_entity_action(self, action: int, is_player: bool) -> bool:
        if is_player:
            self._player_side, self._player_score, self._opponent_side, plays_again = (
                make_valid_action(
                    action, self._player_side, self._player_score, self._opponent_side
                )
            )
        else:
            (
                self._opponent_side,
                self._opponent_score,
                self._player_side,
                plays_again,
            ) = make_valid_action(
                action, self._opponent_side, self._opponent_score, self._player_side
            )

        self._record()

        return plays_again

    def _is_game_over(self) -> bool:
        if sum(self._player_side) == 0 or sum(self._opponent_side) == 0:
            return True

        return False

    @property
    def _game_outcome(self) -> str:
        if self._player_score > self._opponent_score:
            return GameOutcome.WIN

        if self._player_score == self._opponent_score:
            return GameOutcome.DRAW

        return GameOutcome.LOSE

    def _get_player_reward(self) -> float:
        if not self._is_game_over():
            return 1.0 if self._game_outcome == GameOutcome.WIN else 0

        if self._game_outcome == GameOutcome.WIN:
            return 100.0

        if self._game_outcome == GameOutcome.DRAW:
            return 0

        return -100.0

    def _opponent_takes_turn_if_not_game_over(self):
        plays_again = True
        while plays_again and not self._is_game_over():
            plays_again = self._make_entity_action(
                self._opponent_policy(self._seed, self._get_obs()), is_player=False
            )

    def step(self, action: int) -> tuple[list[int], float, bool, bool, dict]:
        assert (
            not self._is_game_over()
        ), "Attempting to step game even though game is over"

        # No state change happens on invalid moves, but a negative reward is received
        # Hopefully this will be enough to learn to produce only valid moves
        if not self._is_action_valid(action):
            return (
                self._get_obs(),
                -1.0,
                False,
                False,
                self._get_info(),
            )

        player_plays_again = self._make_entity_action(action, is_player=True)

        if not player_plays_again:
            self._opponent_takes_turn_if_not_game_over()

        return (
            self._get_obs(),
            self._get_player_reward(),
            self._is_game_over(),
            False,
            self._get_info(),
        )

    def step_in_play_mode(self, action: int):
        assert (
            not self._is_game_over()
        ), "Attempting to step game even though game is over"

        assert self._is_action_valid(
            action
        ), "the action sent to step the env during play is invalid"

        if self._is_player_turn:
            self._is_player_turn = self._make_entity_action(action, is_player=True)
        else:
            # _make_entity_action returns whether the entity that just played gets to play again
            # negation ensures the player will not play again if the is_player=False entity gets another turn
            self._is_player_turn = not self._make_entity_action(action, is_player=False)

    def _is_action_valid(self, action: int) -> bool:
        return self._player_side[action] > 0

    def _set_seed(self, seed: int):
        self._seed = seed
        super().reset(seed=seed)
        np.random.seed(seed)

    def get_serialised_form(self) -> dict:
        return {
            "player_side": self._player_side,
            "opponent_side": self._opponent_side,
            "player_score": self._player_score,
            "opponent_score": self._opponent_score,
            # Always False since the opponent play always assumed to have happened
            "opponent_to_start": False,
        }

    def _deserialise(self, serialised_form: dict) -> bool:
        self._player_side = serialised_form["player_side"]
        self._opponent_side = serialised_form["opponent_side"]
        self._player_score = serialised_form["player_score"]
        self._opponent_score = serialised_form["opponent_score"]
        return serialised_form["opponent_to_start"]

    def _set_board_initial_state(self):
        self._player_side = [4] * 6
        self._opponent_side = [4] * 6
        self._player_score = 0
        self._opponent_score = 0

    def _start_new_history(self):
        self.history = []
        # record initial env state
        self._record()

    def start_in_play_mode_initial(self, is_player_turn: bool):
        self._set_board_initial_state()
        self._start_new_history()
        self._is_player_turn = is_player_turn

    def start_in_play_mode_midgame(self, game_state: dict):
        self._is_player_turn = self._deserialise(game_state)
        self._start_new_history()

    def reset(self, seed: int = None, options: Any = None) -> tuple[list[int], dict]:
        self._set_seed(seed)

        self._set_board_initial_state()
        self._start_new_history()

        # Decide who starts at random
        _does_opponent_start = True if self.np_random.integers(low=0, high=2) else False

        if _does_opponent_start:
            self._make_entity_action(
                self._opponent_policy(seed, self._get_obs()), is_player=False
            )

        self._is_player_turn = True

        return self._get_obs(), {}

    def set_opponent_policy(self, policy: Any):
        self._opponent_policy = policy

    def _record(self):
        self.history.append(
            {
                "player-side": self._player_side,
                "player-score": self._player_score,
                "opponent-side": self._opponent_side,
                "opponent-score": self._opponent_score,
                "as-str": self._state_str,
            }
        )

    @property
    def _history_str(self) -> str:
        return "\n\t" + "\n\t".join(map(lambda x: x["as-str"], self.history))

    @property
    def full_str(self) -> str:
        legend = "[p_side],[p_score],[o_side],[o_score]"
        current_state = f"{self._player_side}, {self._player_score}, {self._opponent_side}, {self._opponent_score}"
        return f"legend: {legend}\ncurrent_state: {current_state}\nhistory: {self._history_str}\n"

    @property
    def _state_str(self) -> str:
        return f"{self._player_side}, {self._player_score}, {self._opponent_side}, {self._opponent_score}"

    def __str__(self) -> str:
        return self.full_str


if __name__ == "__main__":
    mancala = MancalaEnv(seed=42)
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
    print(mancala.full_str)
