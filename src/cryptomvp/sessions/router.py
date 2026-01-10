"""Session routing for market time buckets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover
    ZoneInfo = None
    ZoneInfoNotFoundError = Exception


OFF_SESSION = "OFF_SESSION"


@dataclass(frozen=True)
class SessionDefinition:
    name: str
    start: str
    end: str
    tz: Optional[str] = None


DEFAULT_FIXED = {
    "ASIA": SessionDefinition("ASIA", "00:00", "08:00"),
    "EUROPE": SessionDefinition("EUROPE", "08:00", "16:00"),
    "US": SessionDefinition("US", "16:00", "24:00"),
}

DEFAULT_CLASSIC = {
    "ASIA": SessionDefinition("ASIA", "09:00", "18:00", "Asia/Tokyo"),
    "EUROPE": SessionDefinition("EUROPE", "08:00", "17:00", "Europe/London"),
    "US": SessionDefinition("US", "09:30", "16:00", "America/New_York"),
}


def _parse_time(value: str) -> time:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {value}")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour == 24 and minute == 0:
        return time(23, 59, 59, 999999)
    return time(hour, minute)


def _time_to_minutes(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {value}")
    return int(parts[0]) * 60 + int(parts[1])


def _in_window(t: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= t < end
    return t >= start or t < end


class SessionRouter:
    """Assign session ids based on timestamp and session windows."""

    def __init__(
        self,
        mode: str = "fixed_utc_partitions",
        overlap_policy: str = "priority",
        priority_order: Optional[Iterable[str]] = None,
        sessions: Optional[Dict[str, SessionDefinition]] = None,
    ) -> None:
        self.mode = mode
        self.overlap_policy = overlap_policy
        self.priority_order = list(priority_order or ["US", "EUROPE", "ASIA"])
        if sessions is None:
            if mode == "classic_tz":
                sessions = DEFAULT_CLASSIC
            else:
                sessions = DEFAULT_FIXED
        self.sessions = sessions

    def session_for_timestamp(self, ts_utc: pd.Timestamp) -> str:
        ts = ts_utc
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        matches: List[str] = []
        for name, definition in self.sessions.items():
            start_t = _parse_time(definition.start)
            end_t = _parse_time(definition.end)
            if self.mode == "classic_tz":
                if ZoneInfo is None:
                    raise RuntimeError("zoneinfo unavailable; install tzdata for classic_tz mode.")
                if not definition.tz:
                    raise ValueError(f"Missing tz for session {name}.")
                try:
                    local_ts = ts.tz_convert(ZoneInfo(definition.tz))
                except ZoneInfoNotFoundError as exc:
                    raise RuntimeError(
                        f"Timezone {definition.tz} not found; install tzdata."
                    ) from exc
                local_time = local_ts.timetz()
            else:
                local_time = ts.timetz()

            if _in_window(local_time, start_t, end_t):
                matches.append(name)

        if not matches:
            return OFF_SESSION
        if len(matches) == 1:
            return matches[0]
        if self.overlap_policy == "split":
            ordered = [s for s in self.priority_order if s in matches]
            if not ordered:
                ordered = sorted(matches)
            return "_".join(ordered) + "_OVERLAP"
        # priority
        for name in self.priority_order:
            if name in matches:
                return name
        return matches[0]


def assign_sessions(
    df: pd.DataFrame, router: SessionRouter, ts_col: str = "open_time_ms"
) -> pd.DataFrame:
    """Return a copy of df with session_id derived from UTC timestamps."""
    out = df.copy()
    times = pd.to_datetime(out[ts_col], unit="ms", utc=True)
    out["session_id"] = [router.session_for_timestamp(ts) for ts in times]
    return out


def assign_session_features(
    df: pd.DataFrame, router: SessionRouter, ts_col: str = "open_time_ms"
) -> pd.DataFrame:
    """Annotate session_id and session timing features."""
    out = df.copy()
    times = pd.to_datetime(out[ts_col], unit="ms", utc=True)
    session_ids: List[str] = []
    overlap_flags: List[int] = []
    start_minutes: List[float] = []
    end_minutes: List[float] = []
    minutes_since: List[float] = []
    minutes_to: List[float] = []
    local_minutes: List[float] = []

    for ts in times:
        session_id = router.session_for_timestamp(ts)
        session_ids.append(session_id)
        overlap = int(session_id.endswith("_OVERLAP"))
        overlap_flags.append(overlap)

        if session_id == OFF_SESSION:
            start_minutes.append(float("nan"))
            end_minutes.append(float("nan"))
            minutes_since.append(float("nan"))
            minutes_to.append(float("nan"))
            local_minutes.append(float("nan"))
            continue

        base_id = session_id.replace("_OVERLAP", "").split("_")[0]
        definition = router.sessions.get(base_id)
        if definition is None:
            start_minutes.append(float("nan"))
            end_minutes.append(float("nan"))
            minutes_since.append(float("nan"))
            minutes_to.append(float("nan"))
            local_minutes.append(float("nan"))
            continue

        start_min = _time_to_minutes(definition.start)
        end_min = _time_to_minutes(definition.end)
        start_minutes.append(float(start_min))
        end_minutes.append(float(end_min))

        if router.mode == "classic_tz":
            if ZoneInfo is None:
                raise RuntimeError("zoneinfo unavailable; install tzdata for classic_tz mode.")
            if not definition.tz:
                raise ValueError(f"Missing tz for session {base_id}.")
            try:
                local_ts = ts.tz_convert(ZoneInfo(definition.tz))
            except ZoneInfoNotFoundError as exc:
                raise RuntimeError(
                    f"Timezone {definition.tz} not found; install tzdata."
                ) from exc
        else:
            local_ts = ts

        minute_of_day = local_ts.hour * 60 + local_ts.minute
        local_minutes.append(float(minute_of_day))

        if start_min <= end_min:
            since = minute_of_day - start_min
            to_end = end_min - minute_of_day
        else:
            if minute_of_day >= start_min:
                since = minute_of_day - start_min
                to_end = (24 * 60 - minute_of_day) + end_min
            else:
                since = (24 * 60 - start_min) + minute_of_day
                to_end = end_min - minute_of_day

        minutes_since.append(float(since))
        minutes_to.append(float(to_end))

    out["session_id"] = session_ids
    out["session_overlap"] = overlap_flags
    out["session_start_minute"] = start_minutes
    out["session_end_minute"] = end_minutes
    out["session_minute_of_day"] = local_minutes
    out["minutes_since_session_start"] = minutes_since
    out["minutes_to_session_end"] = minutes_to
    return out
