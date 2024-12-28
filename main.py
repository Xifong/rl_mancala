import gymnasium as gym

import mancala_env
from stable_baselines3 import DQN

env = gym.make("Mancala-v0", max_episode_steps=100)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20_000, log_interval=4)
model.save("dqn_mancala")

obs, info = env.reset()

print("Running Mancala env")
counter = 0
while True:
    print(f"step {counter=}")
    action, states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

    counter += 1

print(env)
