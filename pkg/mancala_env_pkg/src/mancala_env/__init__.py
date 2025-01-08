from typing import Any
from gymnasium.envs.registration import register


# TODO: find a way to force clients to supply an opponent_policy without befouling
# the env checker
def placeholder_policy(seed: Any, _: Any):
    return 0


register(
    id="Mancala-v0",
    kwargs={"seed": None, "opponent_policy": placeholder_policy, "is_play_mode": False},
    entry_point="mancala_env.envs:MancalaEnv",
)
