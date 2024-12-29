import gymnasium as gym

import mancala_env
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import view_logs

env = gym.make("Mancala-v0", max_episode_steps=100)
check_env(env)

eval_env = Monitor(gym.make("Mancala-v0", max_episode_steps=100))


eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=1500,
    n_eval_episodes=80,
    deterministic=True,
    render=False,
)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000, log_interval=4, callback=eval_callback)

view_logs.generate_plots()
view_logs.save_model()
