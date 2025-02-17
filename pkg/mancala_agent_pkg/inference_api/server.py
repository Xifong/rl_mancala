import logging
import os
import json
from typing import Any
from flask import Flask, request, abort

import gymnasium as gym
import mancala_env  # noqa: F401 (mancala_env is in fact used)

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
    env = get_gym_env()
    env.start_in_play_mode_initial(is_player_turn)
    return env.get_serialised_form()


@app.route("/api/next_state", methods=["POST"])
def get_next_env_state():
    if not request.data:
        abort(400)

    data = json.loads(request.data)
    serialised_env, action_to_play = data["current-state"], data["action"]
    env = deserialise_env(serialised_env)
    env.step_in_play_mode(action_to_play)
    return env.get_serialised_form()


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


def deserialise_env(serialised_form: Any) -> gym.Env:
    base_env = get_gym_env()
    base_env.start_in_play_mode_midgame(serialised_form)
    return base_env


def get_gym_env() -> gym.Env:
    return gym.make(
        "Mancala-v0",
        max_episode_steps=100,
        opponent_policy=lambda _: None,
        is_play_mode=True,
    ).unwrapped


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
