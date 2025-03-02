from typing import Optional, Self
from pydantic import BaseModel, NonNegativeInt, ConfigDict, model_validator


class BoardState(BaseModel):
    model_config = {
        "strict": True,
        "json_schema_extra": {
            "examples": [
                {
                    "opponent_score": 0,
                    "opponent_side": [4, 4, 4, 4, 4, 4],
                    "opponent_to_start": True,
                    "player_score": 0,
                    "player_side": [4, 4, 4, 4, 4, 4],
                },
            ]
        },
    }
    player_score: NonNegativeInt
    player_side: list[NonNegativeInt]
    opponent_score: NonNegativeInt
    opponent_side: list[NonNegativeInt]
    opponent_to_start: bool

    @model_validator(mode="after")
    def check_game_state_valid(self) -> Self:
        gems_total = (
            self.player_score
            + sum(self.player_side)
            + self.opponent_score
            + sum(self.opponent_side)
        )
        if gems_total != 48:
            raise ValueError(
                f"must always be exactly 48 gems, but board state contained only '{gems_total}'"
            )
        return self


class HistoryEntry(BaseModel):
    model_config = ConfigDict(strict=True)
    pre_action: Optional[NonNegativeInt] = None
    state: BoardState
    post_action: Optional[NonNegativeInt] = None


class History(BaseModel):
    model_config = ConfigDict(strict=True)
    entries: Optional[list[HistoryEntry]] = None
    _current_action: int

    def record_start(self, state: BoardState, action: int):
        assert self.entries is None, "cannot record game start if there already history"
        self.entries = [HistoryEntry(state=state.copy(deep=True), post_action=action)]
        self._current_action = action

    def record(self, new_state: BoardState, action: int):
        assert (
            len(self.entries) > 0
        ), "cannot record next game state, if there is no previous"
        self.entries.append(
            HistoryEntry(
                pre_action=self._current_action,
                state=new_state.copy(deep=True),
                post_action=action,
            )
        )
        self._current_action = action

    def end(self, last_state: BoardState):
        assert (
            len(self.entries) > 0
        ), "cannot record end game state, if there is no previous state"
        self.entries.append(
            HistoryEntry(
                pre_action=self._current_action, state=last_state.copy(deep=True)
            )
        )


class PlayMetadata(BaseModel):
    model_config = ConfigDict(strict=True)
    allowed_moves: list[NonNegativeInt]
    history: Optional[History] = None
