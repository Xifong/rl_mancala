from pydantic import BaseModel, NonNegativeInt, ValidationError, Field
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


@app.route("/api/initial_state", methods=["GET"])
def get_initial_state():
    args = request.args

    query_param = "is-agent-turn"

    if len(args) <= 0:
        return f"must supply an '{query_param}' query param", 400

    if len(args) > 1:
        return f"must only supply an '{query_param}' query param", 400

    query_value = args.get(query_param, None)
    if query_value is None:
        return f"must supply '{query_param}' query param", 400

    if query_value == "true":
        is_player_turn = True
    elif query_value == "false":
        is_player_turn = False
    else:
        return (
            f"'{query_param}' must be value in {{true, false}}, not '{query_value}'",
            400,
        )

    env = get_gym_env()
    env.start_in_play_mode_initial(is_player_turn=is_player_turn)
    return env.get_serialised_form()


class BoardState(BaseModel):
    player_score: NonNegativeInt
    player_side: list[NonNegativeInt]
    opponent_score: NonNegativeInt
    opponent_side: list[NonNegativeInt]
    opponent_to_start: bool


class NextStateRequest(BaseModel):
    current_state: BoardState = Field(alias="current-state")
    action: NonNegativeInt


@app.route("/api/next_state", methods=["PUT"])
def get_next_env_state():
    if not request.data:
        return "body must contain data", 400

    try:
        data = json.loads(request.data)
    except json.JSONDecodeError as e:
        return f"could not load body data '{e.doc[:15]}', error '{e}'", 400

    try:
        stateRequest = NextStateRequest(**data)
    except ValidationError as e:
        return f"could not unmarshal body data: '{e.errors()}'", 400

    env = deserialise_env(stateRequest.current_state)
    env.step_in_play_mode(stateRequest.action)
    return env.get_serialised_form()


@app.route("/api/next_move", methods=["PUT"])
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
