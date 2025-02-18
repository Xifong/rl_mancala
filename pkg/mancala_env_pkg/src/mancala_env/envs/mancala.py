from typing import Any, Callable
import numpy as np
import gymnasium as gym
import itertools as it
from enum import Enum
import uuid

from . import env_logging


class GameOutcome(Enum):
    WIN = "win"
    DRAW = "draw"
    LOSE = "lose"


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
    assert (
        len(sequence) > 0
    ), f"could not generate action sequence for action '{action}' on player board side: '{entity_side}'"

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
        opponent_policy: Callable,
        seed: int,
        # TODO: improve interface so that you can directly initialise play mode without calling subsequent methods
        is_play_mode: bool,
    ):
        self.metadata = {"render_modes": ["None"]}
        self.render_mode = None

        self._opponent_policy = opponent_policy

        if is_play_mode:
            # TODO: Not actually used because inference server will run this again to supply correct initial player
            self.start_in_play_mode_initial(True)
        else:
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

    def _get_opponent_obs(self) -> np.array:
        return np.array(
            self._opponent_side
            + [self._opponent_score]
            + self._player_side
            + [self._player_score]
        )

    def _get_info(self) -> dict:
        info = {}
        info["is_success"] = self._current_game_outcome == GameOutcome.WIN
        info["is_draw"] = self._current_game_outcome == GameOutcome.DRAW
        info["is_loss"] = self._current_game_outcome == GameOutcome.LOSE

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
    def _current_game_outcome(self) -> str:
        if self._player_score > self._opponent_score:
            return GameOutcome.WIN

        if self._player_score == self._opponent_score:
            return GameOutcome.DRAW

        return GameOutcome.LOSE

    def _get_player_reward(self) -> float:
        if not self._is_game_over():
            return max(self._player_score - self._opponent_score, -1.0)

        if self._current_game_outcome == GameOutcome.WIN:
            return 100.0

        if self._current_game_outcome == GameOutcome.DRAW:
            return 0

        return -100.0

    def _opponent_takes_turn_if_not_game_over(self):
        plays_again = True
        while plays_again and not self._is_game_over():
            opponent_action = self._opponent_policy(
                self._seed, self._get_opponent_obs()
            )
            assert self._is_opponent_action_valid(
                opponent_action
            ), f"opponent action '{opponent_action}' was not valid on opponent board side: '{self._opponent_side}'"
            plays_again = self._make_entity_action(opponent_action, is_player=False)
        self._is_player_turn = True

    def step(self, action: int) -> tuple[list[int], float, bool, bool, dict]:
        assert (
            not self._is_game_over()
        ), "attempting to step game even though game is over"

        # No state change happens on invalid moves, but a negative reward is received
        # Truncate after 10 consecutive invalid actions
        if not self._is_player_action_valid(action):
            self.logger.debug(f"player attempted invalid action: '{action}'")
            self._invalid_count += 1
            return (
                self._get_obs(),
                -1.0,
                False,
                self._invalid_count >= 10,
                self._get_info(),
            )
        self._invalid_count = 0

        player_plays_again = self._make_entity_action(action, is_player=True)

        if not player_plays_again:
            self._is_player_turn = False
            self._opponent_takes_turn_if_not_game_over()
        else:
            self.logger.debug(
                "only player will play on this step since they get to take an extra turn"
            )

        self.logger.debug(self.get_serialised_form())
        if self._is_game_over():
            self.logger.debug("finished a game")

        self.logger.debug(f"step '{self._valid_step_count}' complete")
        self._valid_step_count += 1
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

        assert self._is_player_action_valid(
            action
        ), "the action sent to step the env during play is invalid"

        if self._is_player_turn:
            self._is_player_turn = self._make_entity_action(action, is_player=True)
        else:
            # _make_entity_action returns whether the entity that just played gets to play again
            # negation ensures the player will not play again if the is_player=False entity gets another turn
            self._is_player_turn = not self._make_entity_action(action, is_player=False)

    def _is_player_action_valid(self, action: int) -> bool:
        return self._player_side[action] > 0

    def _is_opponent_action_valid(self, action: int) -> bool:
        return self._opponent_side[action] > 0

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
            "opponent_to_start": not self._is_player_turn,
        }

    def _deserialise(self, serialised_form: dict) -> bool:
        self._player_side = serialised_form.player_side
        self._opponent_side = serialised_form.opponent_side
        self._player_score = serialised_form.player_score
        self._opponent_score = serialised_form.opponent_score
        return not serialised_form.opponent_to_start

    def _set_board_initial_state(self):
        self._player_side = [4] * 6
        self._opponent_side = [4] * 6
        self._player_score = 0
        self._opponent_score = 0

    def _start_new_history(self):
        self._valid_step_count = 0
        self._invalid_count = 0
        self.history = []
        # record initial env state
        self._record()

    def start_in_play_mode_initial(self, is_player_turn: bool):
        self.logger = env_logging.setup_env_logger()({"game_id": uuid.uuid4()})
        self._set_board_initial_state()
        self._start_new_history()
        self._is_player_turn = is_player_turn
        self.logger.debug(f"Initial state: '{self.get_serialised_form()}'")

    def start_in_play_mode_midgame(self, game_state: dict):
        self.logger = env_logging.setup_env_logger()({"game_id": uuid.uuid4()})
        self._is_player_turn = self._deserialise(game_state)
        self._start_new_history()
        self.logger.debug(f"Initial state: '{self.get_serialised_form()}'")

    def reset(self, seed: int = None, options: Any = None) -> tuple[list[int], dict]:
        # Set a new logger uuid
        self.logger = env_logging.setup_env_logger()({"game_id": uuid.uuid4()})
        self._set_seed(seed)

        self._set_board_initial_state()
        self._start_new_history()

        # Decide who starts at random
        self._is_player_turn = True if self.np_random.integers(low=0, high=2) else False

        if not self._is_player_turn:
            self._opponent_takes_turn_if_not_game_over()

        self.logger.debug(f"Initial state: '{self.get_serialised_form()}'")
        return self._get_obs(), {}

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
