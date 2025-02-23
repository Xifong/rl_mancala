import gymnasium as gym
import numpy as np

import mancala_env  # noqa: F401 is used
from pkg.mancala_agent_pkg.model.opponent_policy import get_saved_opponent_policy
from pkg.mancala_agent_pkg.model.load_model import load_model


def infer_from_observation(observation: np.array) -> int:
    opponent_policy = get_saved_opponent_policy("prod", deterministic=False)
    action = opponent_policy(seed=None, observation=observation)
    return int(action)


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
