import os
import shutil
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

matplotlib.use("Agg")


def mkdir_r_p(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)


cleared = False


# Using an lru cache to ensure the side-effect of deleting and creating the directory only happens once
@lru_cache(maxsize=1)
def get_last_run_path(last_run_path="./last_run"):
    mkdir_r_p(last_run_path)
    return last_run_path


def generate_plots():
    last_run_path = get_last_run_path()

    evauluations_path = f"{last_run_path}/evaluations.npz"
    if not os.path.isfile(evauluations_path):
        logger.warning(
            f"No evalutions data found for the last run at '{evauluations_path}'"
        )
        return

    with np.load(f"{last_run_path}/evaluations.npz") as data:
        fig, axes = plt.subplots(3)
        timesteps, results, ep_lengths, successes = (
            data["timesteps"],
            data["results"].mean(axis=1),
            data["ep_lengths"].mean(axis=1),
            data["successes"].mean(axis=1),
        )
        axes[0].plot(timesteps, results)
        axes[0].set_ylabel("reward")
        axes[0].set_ylim(-100, 130)

        axes[1].plot(timesteps, ep_lengths)
        axes[1].set_ylabel("episode length")

        axes[2].plot(timesteps, successes)
        axes[2].set_ylabel("success ratio")
        axes[2].set_ylim(0.0, 1.0)

        axes[2].set_xlabel("timesteps")
        plt.savefig(f"{last_run_path}/plots.png")


def save_files():
    last_run_path = get_last_run_path()
    now = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    mkdir_r_p(f"./saved_models/{now}/")
    for file in os.listdir(last_run_path):
        filename = os.path.basename(file)
        shutil.copy(
            f"{last_run_path}/{filename}",
            f"./saved_models/{now}/{filename}",
        )


def save_run():
    generate_plots()
    save_files()


if __name__ == "__main__":
    save_run()
