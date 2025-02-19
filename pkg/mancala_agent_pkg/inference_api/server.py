import os
import json
from dataclasses import asdict

from flask import Flask, request, jsonify
from pydantic import BaseModel, NonNegativeInt, ValidationError, Field, ConfigDict

from pkg.mancala_agent_pkg.inference_api.infer import (
    get_action_to_play,
    get_fresh_env,
    get_env_from,
)
from pkg.mancala_agent_pkg.inference_api.types import BoardState, PlayMetadata

app = Flask(__name__)

logger = app.logger


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


class GetInitialRequest(BaseModel):
    # Allowing type coercion from "true" -> True etc
    is_player_turn: bool = Field(alias="is-agent-turn")


class BoardStateResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    current_state: BoardState
    metadata: PlayMetadata


@app.route("/api/initial_state", methods=["GET"])
def get_initial_state() -> str:
    args = request.args

    try:
        initial_request = GetInitialRequest(**args)
    except ValidationError as e:
        return f"could not unmarshal request args: '{e.errors()}'", 400

    env = get_fresh_env(is_player_turn=initial_request.is_player_turn)

    return jsonify(
        BoardStateResponse(
            current_state=env.get_serialised_form(),
            metadata={"allowed_moves": env.get_allowed_moves()},
        ).model_dump()
    )


class NextStateRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    current_state: BoardState = Field(alias="current-state")
    action: NonNegativeInt


@app.route("/api/next_state", methods=["PUT"])
def get_next_env_state() -> str:
    if not request.data:
        return "body must contain data", 400

    try:
        data = json.loads(request.data)
    except json.JSONDecodeError as e:
        return f"could not load body data '{e.doc[:15]}', error '{e}'", 400

    try:
        state_request = NextStateRequest(**data)
    except ValidationError as e:
        return f"could not unmarshal body data: '{e.errors()}'", 400

    env = get_env_from(state_request.current_state)
    env.step_in_play_mode(state_request.action)

    return jsonify(
        BoardStateResponse(
            current_state=env.get_serialised_form(),
            metadata={"allowed_moves": env.get_allowed_moves()},
        ).model_dump()
    )


class ActionRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    current_state: BoardState = Field(alias="current-state")


@app.route("/api/next_move", methods=["PUT"])
def get_next_move() -> str:
    if not request.data:
        return "body must contain data", 400

    try:
        data = json.loads(request.data)
    except json.JSONDecodeError as e:
        return f"could not load body data '{e.doc[:15]}', error {e}", 400

    try:
        move_request = ActionRequest(**data)
    except ValidationError as e:
        return f"could not unmarshal body data: {e.errors()}", 400

    try:
        action = get_action_to_play(move_request.current_state)
    except ValueError as e:
        return f"could not get action to play: {e}", 400
    except Exception as e:
        return f"could not get action to play: {e}", 500

    return jsonify(asdict(action))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
