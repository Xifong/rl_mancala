from typing import Optional, Self
from pydantic import (
    BaseModel,
    NonNegativeInt,
    ConfigDict,
    model_validator,
)


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

    # @model_validator(mode="after")
    # def validate_opponent_played_present(self) -> Self:
    #     if self.intermediate_state is not None and self.opponent_played is None:
    #         raise ValueError(
    #             "if 'intermediate_state' is set, 'opponent_played' must also be set"
    #         )
    #
    #     if self.intermediate_state is not None and self.opponent_played is None:
    #         raise ValueError(
    #             "if 'opponent_played' is set, 'intermediate_state' must also be set"
    #         )
    #     return self
