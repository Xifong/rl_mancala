import gymnasium as gym

import mancala_env
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

env = gym.make("Mancala-v0", max_episode_steps=100)
check_env(env)

eval_env = gym.make("Mancala-v0", max_episode_steps=100)


eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=500,
    deterministic=True,
    render=False,
)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000, log_interval=4, callback=eval_callback)
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
