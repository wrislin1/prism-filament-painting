"""Profile-specific print time from a user's plate measurements."""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path


# Optional default supplied by the embedding application.
DEFAULT_CALIBRATION: Path | None = None


@dataclass(frozen=True)
class TimeCalibration:
    start_minutes: float
    area_rate: float  # minutes per square mm for reference_layers layers
    reference_layers: int
    change_minutes: float  # minutes for reference_changes tool changes
    reference_changes: int
    flush_rate: float  # minutes per cubic mm

    tower_start_g: float | None = None
    tower_change_g: float | None = None

    min_margin_minutes: float | None = None
    relative_margin: float | None = None

    def __post_init__(self):
        for count in (self.reference_layers, self.reference_changes):
            if type(count) is not int or count < 1:
                raise ValueError("reference counts must be positive integers")
        rates = [self.start_minutes, self.area_rate, self.change_minutes, self.flush_rate]
        for first, second in ((self.tower_start_g, self.tower_change_g),
                              (self.min_margin_minutes, self.relative_margin)):
            if (first is None) != (second is None):
                raise ValueError("supply both terms for tower mass or time uncertainty")
            if first is not None:
                rates.extend((first, second))
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in rates):
            raise ValueError("time calibration requires finite non-negative rates")


def load_calibration(profile):
    """PRISM_TIME_CALIBRATION names a JSON object keyed by printer profile."""
    path = os.environ.get("PRISM_TIME_CALIBRATION", DEFAULT_CALIBRATION)
    if not path:
        return None
    try:
        profiles = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(profiles, dict):
            raise ValueError("expected a mapping of profiles")
        return TimeCalibration(**profiles[profile]) if profile in profiles else None
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid time calibration: {exc}") from exc


def calibrated_time(calibration, layer_area_mm2, changes, flush_mm3=0):
    if calibration is None:
        return dict(minutes=None, deposition_min=None, time_kind="no estimate available", minutes_range=None)
    deposition = layer_area_mm2 * calibration.area_rate / calibration.reference_layers
    minutes = (calibration.start_minutes + deposition
               + changes * calibration.change_minutes / calibration.reference_changes
               + flush_mm3 * calibration.flush_rate)
    if not math.isfinite(minutes):
        raise ValueError("time estimate exceeds numeric range")
    interval = None
    if calibration.relative_margin is not None:
        margin = max(calibration.min_margin_minutes, calibration.relative_margin * minutes)
        upper = minutes + margin
        if not math.isfinite(upper):
            raise ValueError("time uncertainty exceeds numeric range")
        interval = [max(0., minutes - margin), upper]
    return dict(minutes=minutes, deposition_min=deposition, time_kind="calibrated estimate",
                minutes_range=interval)
