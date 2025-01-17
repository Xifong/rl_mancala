import argparse
import logging
import os

from rl_zoo3 import train as rl_zoo3
from gymnasium.envs.registration import register

import mancala_env
import pkg.mancala_agent_pkg.model.opponent_policy as op

OPPONENT_MODEL_NAME = "opponent"

opponent_policy = op.get_saved_opponent_policy(OPPONENT_MODEL_NAME, deterministic=False)

# Don't have control over the training entrypoint used by rlzoo3, so must re-register to set custom env attributes
# In this cas to use the correct opponent_policy
register(
    id="Mancala-v0",
    kwargs={"seed": None, "opponent_policy": opponent_policy, "is_play_mode": False},
    entry_point="mancala_env.envs:MancalaEnv",
)


# LLM generated shim for rl_zoo3's training module
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="dqn", help="Algorithm to use")
    parser.add_argument("--env", type=str, default="Mancala-v0", help="Environment ID")
    parser.add_argument(
        "-n", "--n-timesteps", type=int, default=10_000, help="Number of timesteps"
    )
    parser.add_argument(
        "-optimize",
        "--optimize",
        action="store_true",
        default=False,
        help="Enable hyperparameter optimization",
    )
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of optimization trials"
    )
    parser.add_argument(
        "--conf-file",
        type=str,
        default="pkg/mancala_agent_pkg/model/hyperparameter_tuning/dqn_mancala_config.yml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--log-folder",
        type=str,
        default="last_rl_zoo3_run/",
        help="Root directory for logs",
    )

    args = parser.parse_known_args()[0]

    log_path = os.path.join(args.log_folder, args.algo)
    # Adding the run count on the end
    log_path = os.path.join(log_path, f"{args.env}-0")
    os.makedirs(log_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_path, "env.log"),
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting training")

    rl_zoo3.train()

    logger.info("Finished training")


if __name__ == "__main__":
    main()
