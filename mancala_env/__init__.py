from gymnasium.envs.registration import register

register(
    id="Mancala-v0",
    entry_point="mancala_env.envs:MancalaEnv",
)
