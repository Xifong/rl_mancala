import gymnasium as gym

import mancala_env
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import pkg.mancala_agent_pkg.model.opponent_policy as op
import pkg.mancala_agent_pkg.model.save as save

OPPONENT_MODEL_NAME = "opponent"

env = gym.make(
    "Mancala-v0",
    max_episode_steps=100,
    opponent_policy=op.get_saved_opponent_policy(OPPONENT_MODEL_NAME),
)
check_env(env)

eval_env = Monitor(gym.make("Mancala-v0", max_episode_steps=100))


eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=save.last_run_path,
    log_path=save.last_run_path,
    eval_freq=5000,
    n_eval_episodes=50,
    deterministic=True,
    render=False,
)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000, log_interval=4, callback=eval_callback)

# Assumes that an EvalCallback has been used
save.save_run()
