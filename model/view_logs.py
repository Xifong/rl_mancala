import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def generate_plots():
    with np.load("./logs/evaluations.npz") as data:
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
        plt.savefig("./logs/plots.png")


if __name__ == "__main__":
    generate_plots()
