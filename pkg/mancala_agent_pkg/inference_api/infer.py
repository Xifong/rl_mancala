import gymnasium as gym
import mancala_env

from pkg.mancala_agent_pkg.inference_api.types import BoardState
from dataclasses import dataclass
from pkg.mancala_agent_pkg.model.infer import infer_from_observation


@dataclass
class ActionPlayed:
    action: int
    was_opponent_move: bool


def get_gym_env() -> mancala_env.MancalaEnv:
    return gym.make(
        "Mancala-v0",
        max_episode_steps=100,
        opponent_policy=lambda _: None,
        is_play_mode=True,
    ).unwrapped


def get_fresh_env(is_player_turn: bool) -> mancala_env.MancalaEnv:
    base_env = get_gym_env()
    base_env.start_in_play_mode_initial(is_player_turn=is_player_turn)
    return base_env


def get_env_from(boardstate: BoardState) -> mancala_env.MancalaEnv:
    base_env = get_gym_env()
    base_env.start_in_play_mode_midgame(boardstate.copy(deep=True))
    return base_env


def get_player_perspective_env(board_state: BoardState) -> mancala_env.MancalaEnv:
    rotated_state = board_state

    if board_state.opponent_to_start:
        rotated_state = BoardState(
            player_score=board_state.opponent_score,
            player_side=board_state.opponent_side,
            opponent_score=board_state.player_score,
            opponent_side=board_state.player_side,
            opponent_to_start=not board_state.opponent_side,
        )

    return get_env_from(rotated_state)


def get_action_to_play_from(board_state: BoardState) -> ActionPlayed:
    env = get_player_perspective_env(board_state)

    if env.get_serialised_form()["is_game_over"]:
        raise ValueError("Cannot make inference when game is over")

    return ActionPlayed(
        action=infer_from_observation(env._get_obs()),
        was_opponent_move=board_state.opponent_to_start,
    )
