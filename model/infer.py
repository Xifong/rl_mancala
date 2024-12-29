import gymnasium as gym
from numpy import np

import mancala_env
from stable_baselines3 import DQN
import save

env = gym.make("Mancala-v0", max_episode_steps=100)

model = DQN.load(f"{save.last_run_path}/best_model")


def infer_from_observation(observation: np.array) -> int:
    # TODO: change the model
    fresh_model = DQN.load(f"{save.last_run_path}/best_model")
    action, _ = fresh_model.predict(obs, deterministic=False)
    return int(action)


if __name__ == "__main__":
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
