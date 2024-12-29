import os
import gymnasium as gym

import mancala_env
from typing import Any

import model.infer as model

from flask import Flask

app = Flask(__name__)

logger = app.logger


@app.get("/api/get_next_state", methods=["POST"])
def get_next_env_state():
    env = deserialise_env(serialised_form)
    action = get_action()
    obs, reward, terminated, truncated, info = env.step(action)
    # Problem is it will play an opponent move here!
    return serialise_env(env)


@app.get("/api/get_next_move", methods=["POST"])
def get_next_move():
    env = deserialise_env(serialised_form)
    # TODO: not use internal method
    action = model.infer_from_observation(env._get_obs())
    return action


def serialise_env(env: gym.Env) -> dict:
    base_env = env.unwrapped
    return base_env.serialise()


def deserialise_env(serialised_form: Any) -> gym.Env:
    env = gym.make("Mancala-v0", max_episode_steps=100)
    base_env = env.unwrapped
    return base_env.deserialse_into_midgame(serialised_form)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
