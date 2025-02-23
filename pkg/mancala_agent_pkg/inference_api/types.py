from typing import Optional
from pydantic import (
    BaseModel,
    NonNegativeInt,
    ConfigDict,
)


class BoardState(BaseModel):
    model_config = ConfigDict(strict=True)
    player_score: NonNegativeInt
    player_side: list[NonNegativeInt]
    opponent_score: NonNegativeInt
    opponent_side: list[NonNegativeInt]
    opponent_to_start: bool


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
