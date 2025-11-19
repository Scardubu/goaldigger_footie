"""Shared normalization utilities for leagues, teams, dates, and match status.

This module centralizes canonical mapping logic so that the real data integrator
and ETL pipeline use identical transformations, improving idempotency and
preventing diverging logic across ingestion paths.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

# Canonical league name -> (id, display_name)
_LEAGUE_MAP = {
    'premier league': ('EPL', 'Premier League'),
    'pl': ('EPL', 'Premier League'),
    'england - premier league': ('EPL', 'Premier League'),
    'la liga': ('LLA', 'La Liga'),
    'laliga': ('LLA', 'La Liga'),
    'bundesliga': ('BL1', 'Bundesliga'),
    'serie a': ('SA', 'Serie A'),
    'ligue 1': ('FL1', 'Ligue 1'),
}

_STATUS_MAP = {
    'scheduled': 'scheduled',
    'timed': 'scheduled',
    'not_started': 'scheduled',
    'in_play': 'in_play',
    'live': 'in_play',
    'paused': 'in_play',
    'finished': 'finished',
    'ft': 'finished',
    'full_time': 'finished',
    'postponed': 'postponed',
    'canceled': 'cancelled',
    'cancelled': 'cancelled'
}

_TEAM_CLEAN_RE = re.compile(r"\s+")


def normalize_league(raw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not raw:
        return None, None
    key = raw.strip().lower()
    return _LEAGUE_MAP.get(key, (None, raw.strip()))


def normalize_status(raw: Optional[str]) -> str:
    if not raw:
        return 'scheduled'
    return _STATUS_MAP.get(raw.strip().lower(), raw.strip().lower())


def parse_match_datetime(raw) -> datetime:
    from datetime import timezone
    if isinstance(raw, datetime):
        return raw
    if not raw:
        return datetime.now(timezone.utc)
    # Remove trailing Z if present
    raw = str(raw).replace('Z', '')
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        # Fallback: just return now
        return datetime.now(timezone.utc)


def normalize_team_name(name: str) -> str:
    return _TEAM_CLEAN_RE.sub(' ', name).strip() if name else name


def canonical_match_id(league_id: Optional[str], home_id: str, away_id: str, dt: datetime) -> str:
    lid = league_id or 'UNK'
    return f"{lid}_{home_id}_{away_id}_{dt.date().isoformat()}"[:50]

__all__ = [
    'normalize_league', 'normalize_status', 'parse_match_datetime', 'normalize_team_name', 'canonical_match_id'
]
