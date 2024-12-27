import mancala


def test_starting_state():
    game = mancala.Mancala(seed=42)
    assert len(game.history) == 1
    assert game.history[-1]["player-side"] == [4] * 6
    assert game.history[-1]["player-score"] == 0
    assert game.history[-1]["opponent-side"] == [4] * 6
    assert game.history[-1]["opponent-score"] == 0
