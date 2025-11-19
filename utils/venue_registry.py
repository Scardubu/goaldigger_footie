#!/usr/bin/env python3
"""Lightweight registry for football venue metadata.

Provides coordinates, capacity, and other contextual attributes for
well-known stadiums so enrichment pipelines can avoid hard-coded
fallbacks and seasonal defaults.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VenueRecord:
    """Structured venue metadata."""

    name: str
    city: str
    country: str
    capacity: Optional[int] = None
    surface: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_m: Optional[float] = None
    timezone: Optional[str] = None
    home_teams: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the record into a dictionary without empty fields."""
        payload: Dict[str, Any] = {
            "name": self.name,
            "city": self.city,
            "country": self.country,
            "capacity": self.capacity,
            "surface": self.surface,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_m": self.altitude_m,
            "timezone": self.timezone,
            "home_teams": list(self.home_teams) if self.home_teams else None,
            "aliases": list(self.aliases) if self.aliases else None,
        }
        return {key: value for key, value in payload.items() if value is not None}


def _normalize(value: str) -> str:
    """Normalize a value for case-insensitive lookup."""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _build_record(entry: Dict[str, Any]) -> VenueRecord:
    """Create a VenueRecord from a raw entry."""
    return VenueRecord(
        name=entry["name"],
        city=entry["city"],
        country=entry["country"],
        capacity=entry.get("capacity"),
        surface=entry.get("surface"),
        latitude=entry.get("latitude"),
        longitude=entry.get("longitude"),
        altitude_m=entry.get("altitude_m"),
        timezone=entry.get("timezone"),
        home_teams=tuple(entry.get("home_teams", ()) or ()),
        aliases=tuple(entry.get("aliases", ()) or ()),
    )


DEFAULT_VENUES: tuple[VenueRecord, ...] = tuple(
    _build_record(entry)
    for entry in (
        {
            "name": "Emirates Stadium",
            "city": "London",
            "country": "England",
            "capacity": 60260,
            "surface": "Grass",
            "latitude": 51.5549,
            "longitude": -0.1084,
            "altitude_m": 40,
            "timezone": "Europe/London",
            "home_teams": ("Arsenal", "Arsenal FC"),
            "aliases": ("Ashburton Grove",),
        },
        {
            "name": "Old Trafford",
            "city": "Manchester",
            "country": "England",
            "capacity": 74140,
            "surface": "Grass",
            "latitude": 53.4631,
            "longitude": -2.2914,
            "altitude_m": 33,
            "timezone": "Europe/London",
            "home_teams": ("Manchester United", "Manchester United FC", "Man United"),
        },
        {
            "name": "Anfield",
            "city": "Liverpool",
            "country": "England",
            "capacity": 54074,
            "surface": "Grass",
            "latitude": 53.4308,
            "longitude": -2.9608,
            "altitude_m": 25,
            "timezone": "Europe/London",
            "home_teams": ("Liverpool", "Liverpool FC"),
        },
        {
            "name": "Stamford Bridge",
            "city": "London",
            "country": "England",
            "capacity": 40834,
            "surface": "Grass",
            "latitude": 51.4816,
            "longitude": -0.1910,
            "altitude_m": 8,
            "timezone": "Europe/London",
            "home_teams": ("Chelsea", "Chelsea FC"),
        },
        {
            "name": "Etihad Stadium",
            "city": "Manchester",
            "country": "England",
            "capacity": 53400,
            "surface": "Grass",
            "latitude": 53.4831,
            "longitude": -2.2004,
            "altitude_m": 55,
            "timezone": "Europe/London",
            "home_teams": ("Manchester City", "Manchester City FC", "Man City"),
            "aliases": ("City of Manchester Stadium",),
        },
        {
            "name": "Tottenham Hotspur Stadium",
            "city": "London",
            "country": "England",
            "capacity": 62850,
            "surface": "Grass",
            "latitude": 51.6042,
            "longitude": -0.0664,
            "altitude_m": 35,
            "timezone": "Europe/London",
            "home_teams": ("Tottenham Hotspur", "Tottenham", "Spurs"),
            "aliases": ("New White Hart Lane",),
        },
        {
            "name": "Camp Nou",
            "city": "Barcelona",
            "country": "Spain",
            "capacity": 99354,
            "surface": "Grass",
            "latitude": 41.3809,
            "longitude": 2.1228,
            "altitude_m": 43,
            "timezone": "Europe/Madrid",
            "home_teams": ("FC Barcelona", "Barcelona", "Barca"),
            "aliases": ("Spotify Camp Nou",),
        },
        {
            "name": "Santiago Bernabeu",
            "city": "Madrid",
            "country": "Spain",
            "capacity": 81044,
            "surface": "Grass",
            "latitude": 40.4530,
            "longitude": -3.6883,
            "altitude_m": 655,
            "timezone": "Europe/Madrid",
            "home_teams": ("Real Madrid", "Real Madrid CF"),
        },
        {
            "name": "San Siro",
            "city": "Milan",
            "country": "Italy",
            "capacity": 75653,
            "surface": "Grass",
            "latitude": 45.4781,
            "longitude": 9.1240,
            "altitude_m": 80,
            "timezone": "Europe/Rome",
            "home_teams": ("Inter Milan", "AC Milan", "Internazionale"),
            "aliases": ("Giuseppe Meazza",),
        },
        {
            "name": "Allianz Arena",
            "city": "Munich",
            "country": "Germany",
            "capacity": 75024,
            "surface": "Grass",
            "latitude": 48.2188,
            "longitude": 11.6242,
            "altitude_m": 520,
            "timezone": "Europe/Berlin",
            "home_teams": ("Bayern Munich", "FC Bayern"),
        },
        {
            "name": "Signal Iduna Park",
            "city": "Dortmund",
            "country": "Germany",
            "capacity": 81365,
            "surface": "Grass",
            "latitude": 51.4926,
            "longitude": 7.4516,
            "altitude_m": 145,
            "timezone": "Europe/Berlin",
            "home_teams": ("Borussia Dortmund", "BVB"),
            "aliases": ("Westfalenstadion",),
        },
        {
            "name": "Parc des Princes",
            "city": "Paris",
            "country": "France",
            "capacity": 47929,
            "surface": "Grass",
            "latitude": 48.8414,
            "longitude": 2.2530,
            "altitude_m": 36,
            "timezone": "Europe/Paris",
            "home_teams": ("Paris Saint-Germain", "PSG"),
        },
    )
)


class VenueRegistry:
    """In-memory registry for venue metadata."""

    def __init__(self, venues: Optional[Iterable[VenueRecord]] = None):
        self._venues: tuple[VenueRecord, ...] = tuple(venues) if venues else DEFAULT_VENUES
        self._name_index: Dict[str, VenueRecord] = {}
        self._team_index: Dict[str, VenueRecord] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        for record in self._venues:
            for key in {record.name, *record.aliases}:
                normalized = _normalize(key)
                if not normalized:
                    continue
                self._name_index[normalized] = record
            for team in record.home_teams:
                normalized_team = _normalize(team)
                if not normalized_team:
                    continue
                # Prefer first entry; do not overwrite if already mapped.
                self._team_index.setdefault(normalized_team, record)

    def resolve(self, venue_name: Optional[str], home_team: Optional[str] = None) -> Optional[VenueRecord]:
        """Resolve a venue by name or associated team."""
        if venue_name:
            record = self._lookup_by_name(venue_name)
            if record:
                return record
        if home_team:
            record = self._lookup_by_team(home_team)
            if record:
                return record
        if venue_name:
            record = self._fuzzy_name_match(venue_name)
            if record:
                return record
        if home_team:
            record = self._fuzzy_team_match(home_team)
            if record:
                return record
        logger.debug("Venue registry could not resolve venue '%s' (home_team=%s)", venue_name, home_team)
        return None

    def _lookup_by_name(self, venue_name: str) -> Optional[VenueRecord]:
        normalized = _normalize(venue_name)
        return self._name_index.get(normalized)

    def _lookup_by_team(self, team_name: str) -> Optional[VenueRecord]:
        normalized = _normalize(team_name)
        return self._team_index.get(normalized)

    def _fuzzy_name_match(self, venue_name: str) -> Optional[VenueRecord]:
        normalized = _normalize(venue_name)
        for key, record in self._name_index.items():
            if normalized in key or key in normalized:
                return record
        return None

    def _fuzzy_team_match(self, team_name: str) -> Optional[VenueRecord]:
        normalized = _normalize(team_name)
        for key, record in self._team_index.items():
            if normalized in key or key in normalized:
                return record
        return None

    @lru_cache(maxsize=256)
    def get(self, venue_name: Optional[str], home_team: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return a serialized venue record if available."""
        record = self.resolve(venue_name, home_team)
        if not record:
            return None
        return record.to_dict()

    def all(self) -> tuple[VenueRecord, ...]:
        """Return all known venue records."""
        return self._venues
