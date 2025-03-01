import os
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

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

open_api_schema_path = "api/v1/openapi.json"

app = FastAPI(
    title="Mancala API",
    description="API for playing the game of Mancala",
    version="1.0.0",
    openapi_url=f"/{open_api_schema_path}",
    docs_url="/api",
    redoc_url="/api/redoc_ui",
)

headers = {
    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
    "Pragme": "no-cache",
    "Expires": "0",
}


# This middleware checks if the server is being run in prod and accounts for the redirect
# that I'm using to keep everything under my personal domain.
@app.middleware("http")
async def dispatch(request: Request, call_next):
    if os.environ.get("IS_PROD") == "true":
        request.app.openapi_url = f"/mancala/{open_api_schema_path}"

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

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
