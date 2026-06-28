from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InstabilityLadderState(Enum):
    QUIET = 0
    PREHEATING = 1
    THERMAL_INSTABILITY = 2
    IGNITION = 3
    CRITICAL = 4


ALERT_MESSAGES = {
    InstabilityLadderState.QUIET: "Quiet – Quiet Sun – No significant instability.",
    InstabilityLadderState.PREHEATING: "Pre-Heating – Thermal Instability Building – FAI > 0.3",
    InstabilityLadderState.THERMAL_INSTABILITY: "Thermal Instability – Sustained Emission Measure Rise – High energy storage.",
    InstabilityLadderState.IGNITION: "Ignition – Non-Thermal Ignition Signatures – HXR microbursts detected.",
    InstabilityLadderState.CRITICAL: "Critical – Thermal + Non-Thermal Confirmed – Eruption likely < 15 min.",
}

STATE_COLORS = {
    InstabilityLadderState.QUIET: "#00C851",
    InstabilityLadderState.PREHEATING: "#FFD700",
    InstabilityLadderState.THERMAL_INSTABILITY: "#FF8C00",
    InstabilityLadderState.IGNITION: "#FF3D00",
    InstabilityLadderState.CRITICAL: "#FF0000",
}


def assign_ladder_state(fai: float, nri_sigma: float, nri_slope: float) -> InstabilityLadderState:
    """
    Sequential Solar Instability Ladder -- 5-stage physics state machine.

    States and transitions:
    QUIET:               FAI < 0.3  AND  NRI < 1sigma
    PREHEATING:          FAI >= 0.3 AND < 0.6  AND  NRI < 1sigma
    THERMAL_INSTABILITY: FAI >= 0.6  AND  NRI < 2sigma
    IGNITION:            NRI >= 2sigma AND FAI >= 0.5  OR  NRI spikes > 4sigma
    CRITICAL:            FAI >= 0.7 AND NRI >= 3sigma for >= 1 min AND positive NRI slope

    Citation: Coronalytics proposal v2.0, Team Chromium, BAH 2026
              Physical basis: Fletcher et al. (2011), Space Science Reviews 159, 19
    """
    if fai >= 0.7 and nri_sigma >= 3 and nri_slope > 0:
        return InstabilityLadderState.CRITICAL
    if (nri_sigma >= 2 and fai >= 0.5) or nri_sigma > 4:
        return InstabilityLadderState.IGNITION
    if fai >= 0.6 and nri_sigma < 2:
        return InstabilityLadderState.THERMAL_INSTABILITY
    if 0.3 <= fai < 0.6 and nri_sigma < 1:
        return InstabilityLadderState.PREHEATING
    return InstabilityLadderState.QUIET


@dataclass
class LadderReading:
    state: InstabilityLadderState
    alert: str
    duration_minutes: int


class InstabilityLadder:
    """Duration-aware state machine for replaying one-minute solar instability readings."""

    def __init__(self) -> None:
        self.state = InstabilityLadderState.QUIET
        self.duration_minutes = 0
        self._critical_candidate_minutes = 0

    def update(self, fai: float, nri_sigma: float, nri_slope: float) -> LadderReading:
        target = assign_ladder_state(fai, nri_sigma, nri_slope)
        critical_ready = fai >= 0.7 and nri_sigma >= 3 and nri_slope > 0
        self._critical_candidate_minutes = self._critical_candidate_minutes + 1 if critical_ready else 0

        if target is InstabilityLadderState.CRITICAL and self._critical_candidate_minutes < 1:
            target = InstabilityLadderState.IGNITION

        if target.value > self.state.value + 1:
            target = InstabilityLadderState(self.state.value + 1)

        if target == self.state:
            self.duration_minutes += 1
        else:
            self.state = target
            self.duration_minutes = 1

        return LadderReading(
            state=self.state,
            alert=ALERT_MESSAGES[self.state],
            duration_minutes=self.duration_minutes,
        )


def replay_ladder(fai_series, nri_sigma_series):
    ladder = InstabilityLadder()
    readings = []
    previous_nri = None
    for fai, nri_value in zip(fai_series, nri_sigma_series):
        slope = 0.0 if previous_nri is None else float(nri_value - previous_nri)
        readings.append(ladder.update(float(fai), float(nri_value), slope))
        previous_nri = nri_value
    return readings
