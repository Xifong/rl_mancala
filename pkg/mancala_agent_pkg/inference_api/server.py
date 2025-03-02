import os
import sys
import logging
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute

from pydantic import BaseModel, NonNegativeInt, Field, ConfigDict

from pkg.mancala_agent_pkg.inference_api.infer import (
    get_action_to_play_from,
    get_fresh_env,
    get_env_from,
)
from pkg.mancala_agent_pkg.inference_api.types import (
    BoardState,
    PlayMetadata,
    History,
)

from mancala_env import get_game_information_message_format

REDIRECT_PREFIX = "/mancala"
open_api_schema_path = "api/v1/openapi.json"

app = FastAPI(
    title="Mancala API",
    description="API for playing the game of Mancala",
    version="1.0.0",
    openapi_url=f"/{open_api_schema_path}",
    docs_url="/api",
    redoc_url="/api/redoc_ui",
)

uvicorn_logger = logging.getLogger("uvicorn.error")

mancala_env_logger = logging.getLogger("mancala_env.envs.env_logging")
mancala_env_logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(processName)s: %(process)d] "
    "[%(threadName)s: %(thread)d] [%(levelname)s] "
    f"[%(name)s] [{get_game_information_message_format()}]: "
    "%(message)s"
)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
mancala_env_logger.addHandler(stream_handler)

headers = {
    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
    "Pragme": "no-cache",
    "Expires": "0",
}


open_api_schemas = {}


def ensure_open_api_schemas(app: FastAPI):
    global open_api_schemas
    if not open_api_schemas:
        uvicorn_logger.debug("creating open api schemas")
        open_api_schemas["non-redirected"] = get_openapi(
            title="Mancala API",
            version="1.0.0",
            description="API for playing the game of Mancala",
            routes=app.routes,
        )

        redirected_routes = []
        for route in app.routes:
            if isinstance(route, APIRoute):
                redirected_routes.append(
                    APIRoute(
                        path=f"{REDIRECT_PREFIX}{route.path}",
                        name=route.name,
                        methods=route.methods,
                        endpoint=route.endpoint,
                    )
                )

        open_api_schemas["redirected"] = get_openapi(
            title="Mancala API",
            version="1.0.0",
            description="API for playing the game of Mancala",
            routes=redirected_routes,
        )


def set_openapi_schema(request: Request, app: FastAPI):
    ensure_open_api_schemas(app)

    if "X-Cloudflare-Redirect" in request.headers:
        uvicorn_logger.info("setting api schema and url for redirection")
        request.app.openapi_url = f"{REDIRECT_PREFIX}/{open_api_schema_path}"
        request.app.openapi_schema = open_api_schemas["redirected"]
    else:
        # Must set these back, since they are not isolated between requests
        request.app.openapi_url = f"/{open_api_schema_path}"
        request.app.openapi_schema = open_api_schemas["non-redirected"]


# this middleware checks if the server is being run behind a redirect and adjusts the generated
# openapi specs and urls to account for it
@app.middleware("http")
async def dispatch(request: Request, call_next):
    set_openapi_schema(request, request.app)

    uvicorn_logger.info(f"path: {request.scope['path']}")
    uvicorn_logger.debug(f"open_api_url: {request.app.openapi_url}")

    response = await call_next(request)
    return response


class BoardStateResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    current_state: BoardState
    metadata: PlayMetadata


class ActionResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    action: NonNegativeInt
    was_opponent_move: bool


class ActionNextStateRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    current_state: BoardState = Field(alias="current-state")
    action: NonNegativeInt


class ActionRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    current_state: BoardState = Field(alias="current-state")


@app.get("/api/initial_state", tags=["atomic-action"])
async def get_initial_state(is_agent_turn: bool) -> BoardStateResponse:
    """
    Get the initial state of a new game.
    """
    env = get_fresh_env(is_player_turn=is_agent_turn)

    return JSONResponse(
        content=BoardStateResponse(
            current_state=env.get_serialised_form(),
            metadata={"allowed_moves": env.get_allowed_moves()},
        ).model_dump(),
        headers=headers,
    )


@app.post("/api/next_state", tags=["atomic-action"])
async def get_next_env_state(body: ActionNextStateRequest) -> BoardStateResponse:
    """
    Given an existing game state and an action to play, return the next game state.
    Actions can be played from either player or opponent perspectives, with the current
    value of `opponent_to_start` determining which.
    """
    history = History()
    history.record_start(state=body.current_state, action=body.action)

    env = get_env_from(body.current_state)
    env.step_in_play_mode(body.action)

    final_state = env.get_serialised_form()
    history.end(last_state=BoardState(**final_state))

    return JSONResponse(
        content=BoardStateResponse(
            current_state=final_state,
            metadata={"allowed_moves": env.get_allowed_moves(), "history": history},
        ).model_dump(),
        headers=headers,
    )


@app.post("/api/next_move", tags=["atomic-action"])
async def get_next_move(body: ActionRequest) -> ActionResponse:
    """
    Given an existing game state, return an action to play using the current deployed RL
    model. (Skill not currently guaranteed!)
    """
    try:
        action = get_action_to_play_from(body.current_state)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"could not get action to play: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"could not get action to play: {e}"
        )

    return JSONResponse(
        content=asdict(action),
        headers=headers,
    )


@app.post("/api/play_move", tags=["compound-action"])
async def play_move(body: ActionNextStateRequest) -> BoardStateResponse:
    """
    Given an existing game state where the player is next to play, and an action for them
    to play:
        1. update the game state
        2. play opponent actions until it's the players turn again
        3. return a full history
    """
    if body.current_state.opponent_to_start:
        raise HTTPException(status_code=400, detail="must be player's turn to play")

    history = History()
    history.record_start(state=body.current_state, action=body.action)

    env = get_env_from(body.current_state)
    env.step_in_play_mode(body.action)

    latest_state = BoardState(**env.get_serialised_form())

    if not latest_state.opponent_to_start:
        history.end(latest_state)
        return JSONResponse(
            content=BoardStateResponse(
                current_state=latest_state,
                metadata={
                    "allowed_moves": env.get_allowed_moves(),
                    "history": history,
                },
            ).model_dump(),
            headers=headers,
        )

    while latest_state.opponent_to_start:
        try:
            opponent_action = get_action_to_play_from(latest_state)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"could not get action to play: {e}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"could not get action to play: {e}"
            )

        history.record(latest_state, opponent_action.action)

        env.step_in_play_mode(opponent_action.action)
        latest_state = BoardState(**env.get_serialised_form())

    history.end(latest_state)

    return JSONResponse(
        content=BoardStateResponse(
            current_state=env.get_serialised_form(),
            metadata={
                "allowed_moves": env.get_allowed_moves(),
                "history": history,
            },
        ).model_dump(),
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn

    # setting log level here because this overwrites log level set directly on the uvicorn logger
    uvicorn.run(
        app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), log_level="debug"
    )
