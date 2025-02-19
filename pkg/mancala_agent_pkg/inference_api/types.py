from typing import Optional
from pydantic import BaseModel, NonNegativeInt, ConfigDict, ValidationError, validator


class BoardState(BaseModel):
    model_config = ConfigDict(strict=True)
    player_score: NonNegativeInt
    player_side: list[NonNegativeInt]
    opponent_score: NonNegativeInt
    opponent_side: list[NonNegativeInt]
    opponent_to_start: bool


class PlayMetadata(BaseModel):
    model_config = ConfigDict(strict=True)
    intermediate_state: Optional[BoardState] = None
    opponent_played: Optional[NonNegativeInt] = None
    allowed_moves: list[NonNegativeInt]

    @validator("intermediate_state")
    def validate_opponent_played_present(cls, value, values):
        if value is not None and values.get("opponent_played") is None:
            raise ValueError(
                "if 'intermediate_state' is set, 'opponent_played' must also be set"
            )
        return value

    @validator("opponent_played")
    def validate_intermediate_state_present(cls, value, values):
        if value is not None and values.get("intermediate_state") is None:
            raise ValueError(
                "if 'opponent_played' is set, 'intermediate_state' must also be set"
            )
        return value
