import logging
import gymnasium as gym

import mancala_env  # noqa: F401 (mancala_env is in fact used)
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import pkg.mancala_agent_pkg.model.opponent_policy as op
import pkg.mancala_agent_pkg.model.save as save
import pkg.mancala_agent_pkg.model.infer as infer

logging.basicConfig(
    filename=f"./{save.get_last_run_path()}/env.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


OPPONENT_MODEL_NAME = "opponent"

# I believe that even in evaluation the opponent itself should not be deterministic
# opponent_policy = op.get_saved_opponent_policy(OPPONENT_MODEL_NAME, deterministic=False)
opponent_policy = op.random_opponent_policy


env = gym.make(
    "Mancala-v0",
    max_episode_steps=100,
    opponent_policy=opponent_policy,
)
check_env(env)


eval_env = Monitor(
    gym.make(
        "Mancala-v0",
        max_episode_steps=100,
        opponent_policy=opponent_policy,
    ),
)


eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=save.get_last_run_path(),
    log_path=save.get_last_run_path(),
    eval_freq=5000,
    n_eval_episodes=80,
    deterministic=True,
    render=False,
)

policy_kwargs = dict(net_arch=[256, 256])

# model = infer.load_model("train_from")
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    #     learning_rate=0.0017660683439426617,
    #     batch_size=100,
    #     buffer_size=10000,
    #     learning_starts=1000,
    #     gamma=0.98,
    #     target_update_interval=5000,
    #     train_freq=256,
    #     exploration_fraction=0.15885316212408052,
    #     exploration_final_eps=0.1533784902718015,
    #     policy_kwargs=policy_kwargs,
)
model.set_env(env, force_reset=True)
model.learn(
    total_timesteps=50_000,
    log_interval=4,
    callback=eval_callback,
)


# Assumes that an EvalCallback has been used
save.save_run()
