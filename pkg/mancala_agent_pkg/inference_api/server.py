import os
import json
from typing import Any
from flask import Flask, request, abort

import gymnasium as gym
import mancala_env

from pkg.mancala_agent_pkg.model import infer as model

app = Flask(__name__)

logger = app.logger


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/api/initial_state", methods=["POST"])
def get_initial_state():
    if not request.data:
        abort(400)

    data = json.loads(request.data)
    is_player_turn = data["is-agent-turn"]
    env = gym.make("Mancala-v0", max_episode_steps=100).unwrapped
    env.start_in_play_mode_initial(is_player_turn)
    return serialise_env(env)


@app.route("/api/next_state", methods=["POST"])
def get_next_env_state():
    if not request.data:
        abort(400)

    data = json.loads(request.data)
    serialised_env, action_to_play = data["current-state"], data["action"]
    env = deserialise_env(serialised_env)
    env.step_in_play_mode(action_to_play)
    return serialise_env(env)


@app.route("/api/next_move", methods=["POST"])
def get_next_move():
    if not request.data:
        abort(400)

    data = json.loads(request.data)
    serialised_env = data["current-state"]
    env = deserialise_env(serialised_env)
    # TODO: not use internal method
    action = model.infer_from_observation(env._get_obs())
    return str(action)


def serialise_env(env: gym.Env) -> dict:
    base_env = env.unwrapped
    return base_env.get_serialised_form()


def deserialise_env(serialised_form: Any) -> gym.Env:
    env = gym.make("Mancala-v0", max_episode_steps=100)
    base_env = env.unwrapped
    base_env.start_in_play_mode_initial(serialised_form)
    return base_env


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
