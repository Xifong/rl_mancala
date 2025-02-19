import gymnasium as gym
import mancala_env

from pkg.mancala_agent_pkg.inference_api.types import BoardState
from dataclasses import dataclass
from pkg.mancala_agent_pkg.model.infer import infer_from_observation


@dataclass
class ActionPlayed:
    action: int
    was_opponent_move: bool


def get_action_to_play(boardState: BoardState) -> ActionPlayed:
    if boardState.opponent_to_start:
        raise ValueError("Cannot make inference from opponent perspective")

    env = get_env_from(boardState)

    if env.get_serialised_form()["is_game_over"]:
        raise ValueError("Cannot make inference when game is over")

    return ActionPlayed(infer_from_observation(env._get_obs()), False)


def get_fresh_env(is_player_turn: bool) -> mancala_env.MancalaEnv:
    base_env = get_gym_env()
    base_env.start_in_play_mode_initial(is_player_turn=is_player_turn)
    return base_env


def get_env_from(boardstate: BoardState) -> mancala_env.MancalaEnv:
    base_env = get_gym_env()
    base_env.start_in_play_mode_midgame(boardstate)
    return base_env


def get_gym_env() -> mancala_env.MancalaEnv:
    return gym.make(
        "Mancala-v0",
        max_episode_steps=100,
        opponent_policy=lambda _: None,
        is_play_mode=True,
    ).unwrapped
