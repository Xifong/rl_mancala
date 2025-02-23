import numpy as np

from pkg.mancala_agent_pkg.model.load_model import load_model


def get_random_valid_move(opponent_side: np.array) -> int:
    assert sum(opponent_side) > 0, "Opponent has no valid moves"

    random_action = np.random.randint(0, 6)
    while opponent_side[random_action] == 0:
        random_action = np.random.randint(0, 6)

    return random_action


# Observation from the perspective of the opponent
def random_opponent_policy(seed: int, observation: np.array) -> int:
    # Extract opponent side
    return get_random_valid_move(observation[:6])


def get_saved_opponent_policy(model_name: str, deterministic: bool = False):
    model = load_model(model_name)

    # Observation from the perspective of the opponent
    def saved_opponent_policy(seed: int, observation: np.array) -> int:
        nonlocal model

        # TODO: extract these using a more sensible shared interface
        # Extract opponent side
        opponent_side = observation[:6]
        assert sum(opponent_side) > 0, "Opponent has no valid moves"

        action, _ = model.predict(observation, deterministic=deterministic)
        assert action >= 0 and action <= 5, f"Action {action} was not in correct range"

        if opponent_side[action] == 0:
            action = get_random_valid_move(opponent_side)

        return int(action)

    return saved_opponent_policy
