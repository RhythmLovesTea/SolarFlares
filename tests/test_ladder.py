from src.ladder import InstabilityLadder, InstabilityLadderState, assign_ladder_state


def test_green_state_quiet_sun():
    assert assign_ladder_state(0.1, 0.5, 0) == InstabilityLadderState.GREEN


def test_critical_requires_both_fai_and_nri():
    assert assign_ladder_state(0.8, 1.0, 0.2) != InstabilityLadderState.CRITICAL
    assert assign_ladder_state(0.4, 3.5, 0.2) != InstabilityLadderState.CRITICAL


def test_ladder_cannot_skip_states():
    ladder = InstabilityLadder()
    first = ladder.update(0.9, 4.5, 1.0)
    second = ladder.update(0.9, 4.5, 1.0)
    assert first.state == InstabilityLadderState.YELLOW
    assert second.state == InstabilityLadderState.ORANGE
