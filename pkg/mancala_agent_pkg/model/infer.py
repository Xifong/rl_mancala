import os
import gymnasium as gym
import numpy as np

import mancala_env  # noqa: F401 is used
from stable_baselines3 import DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from dataclasses import dataclass


def load_model(model: str) -> BaseAlgorithm:
    model_path = f"./saved_models/{model}/best_model"
    assert os.path.isfile(
        f"{model_path}.zip"
    ), f"{model_path}.zip must exist if using saved opponent policy"

    return DQN.load(model_path)


def infer_from_observation(observation: np.array) -> int:
    fresh_model = load_model("prod")
    action, _ = fresh_model.predict(observation, deterministic=False)
    return int(action)


@dataclass
class ActionPlayed:
    action: int
    was_opponent_move: bool


def get_action_to_play(env: mancala_env.MancalaEnv) -> ActionPlayed:
    env_details = env.get_serialised_form()
    if env_details["opponent_to_start"]:
        raise ValueError("Cannot make inference from opponent perspective")

    if env_details["is_game_over"]:
        raise ValueError("Cannot make inference when game is over")

    return ActionPlayed(infer_from_observation(env._get_obs()), False)


if __name__ == "__main__":
    env = gym.make("Mancala-v0", max_episode_steps=100)
    model = load_model("prod")

    obs, info = env.reset()
    print("Running Mancala env")
    counter = 0
    while True:
        print(f"step {counter=}")
        action, states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

        counter += 1

    print(env)
