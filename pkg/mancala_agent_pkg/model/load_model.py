import os
from stable_baselines3 import DQN
from stable_baselines3.common.base_class import BaseAlgorithm


def load_model(model: str) -> BaseAlgorithm:
    model_path = f"./saved_models/{model}/best_model"
    assert os.path.isfile(
        f"{model_path}.zip"
    ), f"{model_path}.zip must exist if using saved opponent policy"

    return DQN.load(model_path)
