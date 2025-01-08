import numpy as np
import os
from stable_baselines3 import DQN

from pkg.mancala_agent_pkg.model.infer import load_model


def random_opponent_policy(seed: int, observation: np.array) -> int:
    # Depends on the opponent side being the last (but one) six elements of the observation
    opponent_side = observation[-7:-1]
    assert sum(opponent_side) > 0, "Opponent has no valid moves"

    random_action = np.random.randint(0, 6)
    while opponent_side[random_action] == 0:
        random_action = np.random.randint(0, 6)

    return random_action


# change to pathlib.Path
def get_saved_opponent_policy(model_name: str):
    model = load_model(model_name)

    def saved_opponent_policy(seed: int, observation: np.array) -> int:
        nonlocal model

        opponent_side = observation[-7:-1]
        assert sum(opponent_side) > 0, "Opponent has no valid moves"

        action, _ = model.predict(observation, deterministic=True)

        # Use random if opponent model is trying to play invalid move
        while opponent_side[action] == 0:
            # TODO: log when these fallbacks are happening
            action = np.random.randint(0, 6)

        return int(action)

    return saved_opponent_policy
