"""
Resilient ETL pipeline to ingest fixtures, results, teams, and leagues
from multiple sources into the normalized SQLAlchemy schema.

Features:
- Multi-source ingestion with graceful fallbacks
- Idempotent upserts and deduplication
- Basic data quality validation hooks
- Retry with exponential backoff for transient errors
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from database.db_manager import DatabaseManager
from database.schema import League, Match, Team
from utils.normalization import (canonical_match_id, normalize_league,
                                 normalize_status, normalize_team_name,
                                 parse_match_datetime)

logger = logging.getLogger(__name__)


def _retry(operation, retries: int = 3, base_delay: float = 1.0):
    for attempt in range(retries):
        try:
            return operation()
        except Exception as e:
            if attempt == retries - 1:
                raise
            sleep_for = base_delay * (2 ** attempt)
            logger.warning(f"ETL retryable error: {e} (attempt {attempt+1}/{retries}), sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)


def _normalize_match(raw: Dict) -> Optional[Dict]:
    """Map raw source match dict to our `Match` ORM fields."""
    try:
        league_raw = raw.get("league") or raw.get("competition") or "Unknown"
        league_id_mapped, league_display = normalize_league(league_raw)
        league_id = league_id_mapped or _league_id_from_name(league_raw)
        home = normalize_team_name(raw.get("home_team") or raw.get("homeTeam") or raw.get("home"))
        away = normalize_team_name(raw.get("away_team") or raw.get("awayTeam") or raw.get("away"))
        dt_raw = raw.get("match_date") or raw.get("utcDate") or raw.get("date")
        dt_parsed = parse_match_datetime(dt_raw)
        if not home or not away:
            return None
        home_id = _team_id_from_name(home)
        away_id = _team_id_from_name(away)
        match_id = canonical_match_id(league_id, home_id, away_id, dt_parsed)
        status = normalize_status(raw.get("status", "scheduled"))
        return {
            "id": match_id,
            "league_id": league_id,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "match_date": dt_parsed,
            "status": status,
            "venue": raw.get("venue"),
            "home_score": raw.get("home_score"),
            "away_score": raw.get("away_score"),
            "competition": league_display or league_raw,
        }
    except Exception as e:
        logger.warning(f"Failed to normalize match: {e}")
        return None


def _league_id_from_name(name: str) -> str:
    mapping = {
        "Premier League": "EPL",
        "La Liga": "LLA",
        "Bundesliga": "BL1",
        "Serie A": "SA",
        "Ligue 1": "FL1",
        "Major League Soccer": "MLS",
    }
    return mapping.get(name, name[:8].upper().replace(" ", ""))


def _team_id_from_name(name: str) -> str:
    # Keep simple: use first 3 consonants uppercase as ID fallback
    cleaned = ''.join(ch for ch in name.upper() if ch.isalpha())
    # Prefer common TLAs if present in name
    candidates = [
        ("ARS", "ARSENAL"), ("CHE", "CHELSEA"), ("MCI", "MANCHESTERCITY"), ("MUN", "MANCHESTERUNITED"),
        ("LIV", "LIVERPOOL"), ("BAR", "BARCELONA"), ("RMA", "REALMADRID"), ("ATM", "ATLETICOMADRID"),
        ("FCB", "BAYERN"), ("BVB", "DORTMUND"), ("RBL", "LEIPZIG"), ("B04", "LEVERKUSEN"),
        ("JUV", "JUVENTUS"), ("MIL", " AC MILAN"), ("INT", "INTER"), ("NAP", "NAPOLI"),
        ("PSG", "PARIS"), ("OL", "LYON"), ("OM", "MARSEILLE"),
    ]
    for tla, token in candidates:
        if token.replace(" ", "") in cleaned:
            return tla
    return (cleaned[:3] or name[:3]).upper()


def _upsert_league_and_teams(session: Session, league_id: str, league_name: str, home_id: str, home_name: str, away_id: str, away_name: str):
    league = session.query(League).get(league_id)
    if not league:
        session.add(League(id=league_id, name=league_name or league_id, country="", tier=1))
    # Teams
    home = session.query(Team).get(home_id)
    if not home:
        session.add(Team(id=home_id, name=home_name, short_name=home_name, tla=home_id, league_id=league_id))
    away = session.query(Team).get(away_id)
    if not away:
        session.add(Team(id=away_id, name=away_name, short_name=away_name, tla=away_id, league_id=league_id))


def _upsert_match(session: Session, data: Dict):
    existing = session.query(Match).get(data["id"])  # type: ignore[name-defined]
    if existing:
        for k, v in data.items():
            setattr(existing, k, v)
    else:
        session.add(Match(**data))  # type: ignore[name-defined]


def _iter_source_matches(days_back: int, days_ahead: int) -> Iterable[Dict]:
    # Try enhanced aggregator
    try:
        from utils.enhanced_data_aggregator import (get_current_fixtures,
                                                    get_todays_matches)
        for m in _retry(lambda: get_todays_matches()) or []:
            yield m
        for f in _retry(lambda: get_current_fixtures(days_ahead=days_ahead)) or []:
            yield f
    except Exception as e:
        logger.warning(f"Enhanced aggregator unavailable: {e}")

    # Try real data integrator
    try:
        from real_data_integrator import get_real_fixtures, get_real_matches
        for m in _retry(lambda: get_real_matches(days_ahead=days_back)) or []:
            yield m
        for f in _retry(lambda: get_real_fixtures(days_ahead=days_ahead)) or []:
            yield f
    except Exception as e:
        logger.warning(f"Real data integrator unavailable: {e}")


def ingest_from_sources(db_uri: Optional[str] = None, days_back: int = 2, days_ahead: int = 7) -> int:
    """Ingest matches from available sources into DB; returns rows upserted."""
    db = DatabaseManager(db_uri)
    db.create_tables()
    upserts = 0
    with db.session_scope() as session:
        seen_ids: set = set()
        for raw in _iter_source_matches(days_back, days_ahead):
            norm = _normalize_match(raw)
            if not norm:
                continue
            # dedupe
            if norm["id"] in seen_ids:
                continue
            seen_ids.add(norm["id"])

            # ensure league/teams
            _upsert_league_and_teams(
                session,
                norm["league_id"],
                norm.get("competition") or norm.get("league_id"),
                norm["home_team_id"],
                raw.get("home_team") or raw.get("homeTeam") or norm["home_team_id"],
                norm["away_team_id"],
                raw.get("away_team") or raw.get("awayTeam") or norm["away_team_id"],
            )

            _upsert_match(session, norm)
            upserts += 1

    logger.info(f"ETL ingest complete: upserted {upserts} matches")
    return upserts


