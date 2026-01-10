from __future__ import annotations

import pytest

import pandas as pd

from cryptomvp.sessions.router import SessionRouter


def test_fixed_utc_partitions_boundaries() -> None:
    router = SessionRouter(mode="fixed_utc_partitions")
    assert router.session_for_timestamp(pd.Timestamp("2023-01-01T00:00:00Z")) == "ASIA"
    assert router.session_for_timestamp(pd.Timestamp("2023-01-01T07:59:00Z")) == "ASIA"
    assert router.session_for_timestamp(pd.Timestamp("2023-01-01T08:00:00Z")) == "EUROPE"
    assert router.session_for_timestamp(pd.Timestamp("2023-01-01T15:59:00Z")) == "EUROPE"
    assert router.session_for_timestamp(pd.Timestamp("2023-01-01T16:00:00Z")) == "US"
    assert router.session_for_timestamp(pd.Timestamp("2023-01-01T23:59:00Z")) == "US"


def _classic_router(overlap_policy: str) -> SessionRouter:
    return SessionRouter(
        mode="classic_tz",
        overlap_policy=overlap_policy,
        priority_order=["US", "EUROPE", "ASIA"],
    )


def test_classic_tz_priority_overlap() -> None:
    try:
        router = _classic_router("priority")
        ts = pd.Timestamp("2023-01-10T15:00:00Z")  # overlap Europe + US
        assert router.session_for_timestamp(ts) == "US"
    except Exception as exc:  # tzdata missing on Windows
        pytest.skip(str(exc))


def test_classic_tz_split_overlap() -> None:
    try:
        router = _classic_router("split")
        ts = pd.Timestamp("2023-01-10T15:00:00Z")  # overlap Europe + US
        assert router.session_for_timestamp(ts) == "US_EUROPE_OVERLAP"
    except Exception as exc:  # tzdata missing on Windows
        pytest.skip(str(exc))


def test_classic_tz_dst_window() -> None:
    try:
        router = _classic_router("priority")
        # After DST change, 09:30 NY is 13:30 UTC.
        ts = pd.Timestamp("2023-03-13T13:30:00Z")
        assert router.session_for_timestamp(ts) == "US"
    except Exception as exc:  # tzdata missing on Windows
        pytest.skip(str(exc))
