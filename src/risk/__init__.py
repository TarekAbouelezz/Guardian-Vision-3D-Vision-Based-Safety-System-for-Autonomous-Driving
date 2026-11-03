"""Distance-zone classification and vehicle action policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZoneThresholds:
    """Critical and warning distance boundaries in metres."""

    critical: float = 7.0
    warning: float = 15.0

    def __post_init__(self) -> None:
        if self.critical < 0 or self.warning <= self.critical:
            raise ValueError("Require 0 <= critical < warning")


@dataclass(frozen=True)
class ZoneResult:
    zone: str
    action: str
    distance: float
    color_bgr: tuple[int, int, int]


def stopping_distance(speed_kmh: float, reaction_time_s: float = 1.0, deceleration_mps2: float = 6.0) -> float:
    """Return ``v*t_react + v²/(2a)`` using speed in km/h."""
    if speed_kmh < 0 or reaction_time_s < 0 or deceleration_mps2 <= 0:
        raise ValueError("Speed/reaction time must be non-negative and deceleration must be positive")
    speed_mps = speed_kmh / 3.6
    return speed_mps * reaction_time_s + speed_mps**2 / (2.0 * deceleration_mps2)


def thresholds_from_speed(
    speed_kmh: float,
    reaction_time_s: float = 1.0,
    deceleration_mps2: float = 6.0,
    warning_buffer_m: float = 5.0,
) -> ZoneThresholds:
    """Derive critical and warning zones from vehicle stopping distance."""
    critical = stopping_distance(speed_kmh, reaction_time_s, deceleration_mps2)
    return ZoneThresholds(critical=max(critical, 0.1), warning=critical + max(warning_buffer_m, 0.0))


def classify_zone(distance: float, thresholds: ZoneThresholds | None = None) -> ZoneResult:
    """Classify a metric distance into monitoring, warning, or critical danger."""
    if distance < 0:
        raise ValueError("Distance must be non-negative")
    limits = thresholds or ZoneThresholds()
    if distance < limits.critical:
        return ZoneResult("critical_danger", "EMERGENCY BRAKE", distance, (0, 0, 255))
    if distance < limits.warning:
        return ZoneResult("warning", "REDUCE SPEED", distance, (0, 165, 255))
    return ZoneResult("monitoring", "ALARM", distance, (0, 255, 0))


def get_system_action(detections: list[ZoneResult]) -> str:
    """Return the action with the highest safety priority."""
    priority = {"EMERGENCY BRAKE": 3, "REDUCE SPEED": 2, "ALARM": 1, "NORMAL": 0}
    return max((item.action for item in detections), key=priority.get, default="NORMAL")
