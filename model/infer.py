import gymnasium as gym
import numpy as np

import mancala_env
from stable_baselines3 import DQN

infer_using = "./saved_models/2024-12-29_12-30-15/"


def infer_from_observation(observation: np.array) -> int:
    # TODO: change the model
    fresh_model = DQN.load(f"{infer_using}/best_model")
    action, _ = fresh_model.predict(observation, deterministic=False)
    return int(action)


if __name__ == "__main__":
    env = gym.make("Mancala-v0", max_episode_steps=100)
    model = DQN.load(f"{infer_using}/best_model")

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
