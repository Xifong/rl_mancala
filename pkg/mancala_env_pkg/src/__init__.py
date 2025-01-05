from gymnasium.envs.registration import register

register(
    id="Mancala-v0",
    entry_point="src.envs:MancalaEnv",
)
