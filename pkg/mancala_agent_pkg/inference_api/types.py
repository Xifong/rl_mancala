from pydantic import BaseModel, NonNegativeInt, ConfigDict


class BoardState(BaseModel):
    model_config = ConfigDict(strict=True)
    player_score: NonNegativeInt
    player_side: list[NonNegativeInt]
    opponent_score: NonNegativeInt
    opponent_side: list[NonNegativeInt]
    opponent_to_start: bool


class PlayMetadata(BaseModel):
    model_config = ConfigDict(strict=True)
    allowed_moves: list[NonNegativeInt]
