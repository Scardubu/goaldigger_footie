#!/usr/bin/env python3
"""Centralized registry for standardized football team assets.

Provides a single source of truth for official team names, league metadata,
flag assets, and alias resolution across the GoalDiggers platform.

The registry currently covers every club in the top 7 European leagues:
- Premier League
- La Liga
- Bundesliga
- Serie A
- Ligue 1
- Eredivisie
- Primeira Liga
"""
from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

FLAG_CDN_BASE = "https://flagcdn.com"
DEFAULT_FLAG_SIZE = 48


@dataclass(frozen=True)
class TeamAssetRecord:
    """Data structure describing a team's standardized metadata."""

    team_id: str
    official_name: str
    display_name: str
    short_name: str
    tla: str
    league_code: str
    league_name: str
    country: str
    country_code: str
    aliases: list[str]
    primary_color: str | None = None

    def flag_png(self, size: int = DEFAULT_FLAG_SIZE) -> str:
        """Return the CDN URL for the team's country flag (PNG)."""
        size = max(16, min(size, 512))
        return f"{FLAG_CDN_BASE}/w{size}/{self.country_code.lower()}.png"

    def flag_svg(self) -> str:
        """Return the CDN URL for the team's country flag (SVG)."""
        return f"{FLAG_CDN_BASE}/{self.country_code.lower()}.svg"

    @property
    def flag_emoji(self) -> str:
        """Return the Unicode emoji for the team's country code."""
        return country_code_to_emoji(self.country_code)


# --------------------------------------------------------------------------------------
# Raw data definitions
# --------------------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "team_assets"
DATA_FILE = DATA_DIR / "top7_team_assets.json"

# The JSON file is generated manually to provide an explicit, editable source.
# It is stored in data/team_assets/top7_team_assets.json and loaded lazily.


def _ensure_data_file_exists() -> None:
    """Ensure the team assets data file is present, creating it if necessary."""
    if DATA_FILE.exists():
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # The payload was curated from current league rosters (2024-25 season baseline).
    payload: dict[str, dict[str, object]] = {
        "meta": {
            "version": "1.0.0",
            "source": "GoalDiggers Team Assets Registry",
            "description": (
                "Standardized metadata for all clubs in the top 7 European leagues."
            ),
        },
        "leagues": {
            "PL": {"name": "Premier League", "country": "England", "country_code": "GB"},
            "PD": {"name": "La Liga", "country": "Spain", "country_code": "ES"},
            "BL1": {"name": "Bundesliga", "country": "Germany", "country_code": "DE"},
            "SA": {"name": "Serie A", "country": "Italy", "country_code": "IT"},
            "FL1": {"name": "Ligue 1", "country": "France", "country_code": "FR"},
            "DED": {"name": "Eredivisie", "country": "Netherlands", "country_code": "NL"},
            "PPL": {"name": "Primeira Liga", "country": "Portugal", "country_code": "PT"},
        },
        "teams": [],
    }

    def _team_entry(
        team_id: str,
        official: str,
        display: str,
        short: str,
        tla: str,
        league_code: str,
        country: str,
        country_code: str,
        aliases: Iterable[str],
        primary_color: str | None = None,
    ) -> dict[str, object]:
        alias_set: set[str] = {official, display, short, tla}
        alias_set.update(aliases)
        alias_set = {a for a in alias_set if a}
        return {
            "team_id": team_id,
            "official_name": official,
            "display_name": display,
            "short_name": short or display,
            "tla": tla,
            "league_code": league_code,
            "country": country,
            "country_code": country_code,
            "aliases": sorted(alias_set),
            "primary_color": primary_color,
        }

    def _slug(name: str) -> str:
        return slugify(name)

    # Premier League (England)
    premier_aliases = {
        "Arsenal FC": ["Arsenal"],
        "Aston Villa FC": ["Aston Villa"],
        "Brentford FC": ["Brentford"],
        "Brighton & Hove Albion FC": ["Brighton", "Brighton Hove Albion"],
        "Burnley FC": ["Burnley"],
        "Chelsea FC": ["Chelsea"],
        "Crystal Palace FC": ["Crystal Palace"],
        "Everton FC": ["Everton"],
        "Fulham FC": ["Fulham"],
        "Liverpool FC": ["Liverpool"],
        "Luton Town FC": ["Luton", "Luton Town"],
        "Manchester City FC": ["Manchester City", "Man City"],
        "Manchester United FC": ["Manchester United", "Man Utd", "Man United"],
        "Newcastle United FC": ["Newcastle", "Newcastle United"],
        "Nottingham Forest FC": ["Nottingham Forest", "Nottm Forest"],
        "Sheffield United FC": ["Sheffield United", "Sheff Utd"],
        "Tottenham Hotspur FC": ["Tottenham", "Spurs"],
        "West Ham United FC": ["West Ham", "West Ham United"],
        "Wolverhampton Wanderers FC": ["Wolverhampton", "Wolves"],
        "AFC Bournemouth": ["Bournemouth"],
    }
    premier_primary_colors = {
        "Arsenal FC": "#EF0107",
        "Aston Villa FC": "#95BFE5",
        "Brentford FC": "#D50032",
        "Brighton & Hove Albion FC": "#0057B8",
        "Burnley FC": "#6C1D45",
        "Chelsea FC": "#034694",
        "Crystal Palace FC": "#1B458F",
        "Everton FC": "#003399",
        "Fulham FC": "#000000",
        "Liverpool FC": "#C8102E",
        "Luton Town FC": "#F36C21",
        "Manchester City FC": "#6CABDD",
        "Manchester United FC": "#DA020E",
        "Newcastle United FC": "#241F20",
        "Nottingham Forest FC": "#DD0000",
        "Sheffield United FC": "#EE2737",
        "Tottenham Hotspur FC": "#132257",
        "West Ham United FC": "#7A263A",
        "Wolverhampton Wanderers FC": "#FDB913",
        "AFC Bournemouth": "#DA291C",
    }
    for official, aliases in premier_aliases.items():
        display = aliases[0]
        tla = {
            "Arsenal FC": "ARS",
            "Aston Villa FC": "AVL",
            "Brentford FC": "BRE",
            "Brighton & Hove Albion FC": "BHA",
            "Burnley FC": "BUR",
            "Chelsea FC": "CHE",
            "Crystal Palace FC": "CRY",
            "Everton FC": "EVE",
            "Fulham FC": "FUL",
            "Liverpool FC": "LIV",
            "Luton Town FC": "LUT",
            "Manchester City FC": "MCI",
            "Manchester United FC": "MUN",
            "Newcastle United FC": "NEW",
            "Nottingham Forest FC": "NFO",
            "Sheffield United FC": "SHU",
            "Tottenham Hotspur FC": "TOT",
            "West Ham United FC": "WHU",
            "Wolverhampton Wanderers FC": "WOL",
            "AFC Bournemouth": "BOU",
        }[official]
        team_id = _slug(official)
        payload["teams"].append(
            _team_entry(
                team_id,
                official,
                display,
                display,
                tla,
                "PL",
                "England",
                "GB",
                aliases + [display.replace(" FC", ""), official.replace(" FC", "")],
                premier_primary_colors.get(official),
            )
        )

    # La Liga (Spain)
    la_liga_entries = [
        ("Real Madrid CF", ["Real Madrid", "Real"], "RMA"),
        ("FC Barcelona", ["Barcelona", "Barca"], "BAR"),
        ("Club Atlético de Madrid", ["Atlético Madrid", "Atletico Madrid", "Atleti"], "ATM"),
        ("Sevilla FC", ["Sevilla"], "SEV"),
        ("Villarreal CF", ["Villarreal"], "VIL"),
        ("Real Sociedad de Fútbol", ["Real Sociedad"], "RSO"),
        ("Real Betis Balompié", ["Real Betis", "Betis"], "BET"),
        ("Valencia CF", ["Valencia"], "VAL"),
        ("Athletic Club", ["Athletic Bilbao", "Athletic"], "ATH"),
        ("Getafe CF", ["Getafe"], "GET"),
        ("Rayo Vallecano de Madrid", ["Rayo Vallecano", "Rayo"], "RAY"),
        ("RC Celta de Vigo", ["Celta Vigo", "Celta"], "CEL"),
        ("CA Osasuna", ["Osasuna"], "OSA"),
        ("Deportivo Alavés", ["Alaves"], "ALA"),
        ("RCD Mallorca", ["Mallorca"], "MAL"),
        ("Girona FC", ["Girona"], "GIR"),
        ("Cádiz CF", ["Cadiz"], "CAD"),
        ("Granada CF", ["Granada"], "GRA"),
        ("UD Almería", ["Almeria"], "ALM"),
        ("UD Las Palmas", ["Las Palmas"], "LPA"),
    ]
    for official, aliases, tla in la_liga_entries:
        display = aliases[0]
        payload["teams"].append(
            _team_entry(
                _slug(official),
                official,
                display,
                display,
                tla,
                "PD",
                "Spain",
                "ES",
                aliases,
            )
        )

    # Bundesliga (Germany)
    bundesliga_entries = [
        ("FC Bayern München", ["Bayern Munich", "FC Bayern"], "BAY"),
        ("Borussia Dortmund", ["Borussia Dortmund", "Dortmund", "BVB"], "BVB"),
        ("RB Leipzig", ["RB Leipzig", "Leipzig"], "RBL"),
        ("Bayer 04 Leverkusen", ["Bayer Leverkusen", "Leverkusen"], "B04"),
        ("Borussia Mönchengladbach", ["Borussia Monchengladbach", "Mönchengladbach", "Gladbach"], "BMG"),
        ("VfL Wolfsburg", ["Wolfsburg"], "WOB"),
        ("Eintracht Frankfurt", ["Eintracht Frankfurt", "Frankfurt"], "SGE"),
        ("TSG 1899 Hoffenheim", ["Hoffenheim"], "HOF"),
        ("1. FC Union Berlin", ["Union Berlin", "Union"], "FCU"),
        ("SC Freiburg", ["Freiburg"], "SCF"),
        ("VfB Stuttgart", ["Stuttgart"], "VFB"),
        ("1. FSV Mainz 05", ["Mainz 05", "Mainz"], "M05"),
        ("1. FC Köln", ["1. FC Koln", "Koln", "Cologne"], "KOE"),
        ("FC Augsburg", ["Augsburg"], "AUG"),
        ("Werder Bremen", ["Werder Bremen", "Bremen"], "BRE"),
        ("VfL Bochum 1848", ["VfL Bochum", "Bochum"], "BOC"),
        ("SV Darmstadt 98", ["Darmstadt"], "DAR"),
        ("1. FC Heidenheim 1846", ["Heidenheim"], "HEI"),
    ]
    for official, aliases, tla in bundesliga_entries:
        display = aliases[0]
        payload["teams"].append(
            _team_entry(
                _slug(official),
                official,
                display,
                display,
                tla,
                "BL1",
                "Germany",
                "DE",
                aliases,
            )
        )

    # Serie A (Italy)
    serie_a_entries = [
        ("FC Internazionale Milano", ["Inter Milan", "Inter"], "INT"),
        ("AC Milan", ["AC Milan", "Milan"], "MIL"),
        ("Juventus FC", ["Juventus", "Juve"], "JUV"),
        ("SSC Napoli", ["Napoli"], "NAP"),
        ("AS Roma", ["Roma"], "ROM"),
        ("SS Lazio", ["Lazio"], "LAZ"),
        ("Atalanta BC", ["Atalanta"], "ATA"),
        ("ACF Fiorentina", ["Fiorentina"], "FIO"),
        ("Bologna FC 1909", ["Bologna"], "BOL"),
        ("Torino FC", ["Torino"], "TOR"),
        ("Udinese Calcio", ["Udinese"], "UDI"),
        ("US Sassuolo Calcio", ["Sassuolo"], "SAS"),
        ("Hellas Verona FC", ["Hellas Verona", "Verona"], "VER"),
        ("Genoa CFC", ["Genoa"], "GEN"),
        ("Cagliari Calcio", ["Cagliari"], "CAG"),
        ("US Lecce", ["Lecce"], "LEC"),
        ("AC Monza", ["Monza"], "MON"),
        ("Empoli FC", ["Empoli"], "EMP"),
        ("Parma Calcio 1913", ["Parma"], "PAR"),
        ("Venezia FC", ["Venezia"], "VEN"),
    ]
    for official, aliases, tla in serie_a_entries:
        display = aliases[0]
        payload["teams"].append(
            _team_entry(
                _slug(official),
                official,
                display,
                display,
                tla,
                "SA",
                "Italy",
                "IT",
                aliases,
            )
        )

    # Ligue 1 (France)
    ligue1_entries = [
        ("Paris Saint-Germain FC", ["Paris Saint-Germain", "PSG"], "PSG", None, "FR"),
        ("AS Monaco FC", ["AS Monaco", "Monaco"], "MON", None, "MC"),
        ("Olympique de Marseille", ["Marseille", "OM"], "MAR", None, "FR"),
        ("Olympique Lyonnais", ["Lyon", "OL"], "LYO", None, "FR"),
        ("LOSC Lille", ["Lille"], "LIL", None, "FR"),
        ("OGC Nice", ["Nice"], "NIC", None, "FR"),
        ("Stade Rennais FC", ["Rennes"], "REN", None, "FR"),
        ("Stade de Reims", ["Reims"], "REI", None, "FR"),
        ("RC Lens", ["Lens"], "LEN", None, "FR"),
        ("Toulouse FC", ["Toulouse"], "TOU", None, "FR"),
        ("FC Nantes", ["Nantes"], "NAN", None, "FR"),
        ("Montpellier HSC", ["Montpellier"], "MONP", "MON", "FR"),
        ("FC Lorient", ["Lorient"], "LOR", None, "FR"),
        ("Stade Brestois 29", ["Brest"], "BRE", None, "FR"),
        ("Racing Club de Strasbourg Alsace", ["Strasbourg"], "STR", None, "FR"),
        ("Clermont Foot 63", ["Clermont"], "CLE", None, "FR"),
        ("Le Havre AC", ["Le Havre"], "HAV", None, "FR"),
        ("FC Metz", ["Metz"], "MET", None, "FR"),
    ]
    for official, aliases, tla, short_override, country_code in ligue1_entries:
        display = aliases[0]
        payload["teams"].append(
            _team_entry(
                _slug(official),
                official,
                display,
                short_override or display,
                tla,
                "FL1",
                "France" if country_code == "FR" else "Monaco",
                country_code,
                aliases,
            )
        )

    # Eredivisie (Netherlands)
    eredivisie_entries = [
        ("AFC Ajax", ["Ajax"], "AJA"),
        ("PSV", ["PSV", "PSV Eindhoven"], "PSV"),
        ("Feyenoord", ["Feyenoord"], "FEY"),
        ("AZ Alkmaar", ["AZ Alkmaar", "AZ"], "AZA"),
        ("FC Twente", ["FC Twente", "Twente"], "TWE"),
        ("SC Heerenveen", ["Heerenveen"], "HEE"),
        ("FC Utrecht", ["Utrecht"], "UTR"),
        ("Vitesse", ["Vitesse"], "VIT"),
        ("NEC Nijmegen", ["NEC Nijmegen", "NEC"], "NEC"),
        ("Sparta Rotterdam", ["Sparta Rotterdam", "Sparta"], "SPA"),
        ("Fortuna Sittard", ["Fortuna Sittard", "Fortuna"], "FOR"),
        ("Go Ahead Eagles", ["Go Ahead Eagles", "Go Ahead"], "GAE"),
        ("PEC Zwolle", ["PEC Zwolle", "Zwolle"], "PEC"),
        ("RKC Waalwijk", ["RKC Waalwijk", "RKC"], "RKC"),
        ("Heracles Almelo", ["Heracles", "Heracles Almelo"], "HER"),
        ("Excelsior", ["Excelsior"], "EXC"),
        ("FC Volendam", ["Volendam"], "VOL"),
        ("Almere City FC", ["Almere City"], "ALM"),
    ]
    for official, aliases, tla in eredivisie_entries:
        display = aliases[0]
        payload["teams"].append(
            _team_entry(
                _slug(official),
                official,
                display,
                display,
                tla,
                "DED",
                "Netherlands",
                "NL",
                aliases,
            )
        )

    # Primeira Liga (Portugal)
    primeira_entries = [
        ("SL Benfica", ["Benfica"], "BEN"),
        ("FC Porto", ["Porto", "FC Porto"], "POR"),
        ("Sporting CP", ["Sporting", "Sporting CP"], "SPO"),
        ("SC Braga", ["Braga"], "BRA"),
        ("Vitória SC", ["Vitoria SC", "Vitória Guimarães", "Guimaraes"], "VSC"),
        ("FC Famalicão", ["Famalicao"], "FAM"),
        ("Rio Ave FC", ["Rio Ave"], "RIO"),
        ("Gil Vicente FC", ["Gil Vicente"], "GIL"),
        ("Moreirense FC", ["Moreirense"], "MOR"),
        ("Boavista FC", ["Boavista"], "BOA"),
        ("CS Marítimo", ["Maritimo"], "MAR"),
        ("Portimonense SC", ["Portimonense"], "PORI"),
        ("GD Chaves", ["Chaves"], "CHA"),
        ("SC Farense", ["Farense"], "FAR"),
        ("Estoril Praia", ["Estoril"], "EST"),
        ("Casa Pia AC", ["Casa Pia"], "CPA"),
        ("Arouca", ["Arouca"], "ARO"),
        ("Vizela", ["Vizela"], "VIZ"),
    ]
    for official, aliases, tla in primeira_entries:
        display = aliases[0]
        payload["teams"].append(
            _team_entry(
                _slug(official),
                official,
                display,
                display,
                tla,
                "PPL",
                "Portugal",
                "PT",
                aliases,
            )
        )

    DATA_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@lru_cache(maxsize=1)
def _load_raw_data() -> dict[str, object]:
    _ensure_data_file_exists()
    with DATA_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def slugify(value: str) -> str:
    """Create a normalized slug for identifiers."""
    normalized = unicodedata.normalize("NFKD", value)
    cleaned = "".join(ch for ch in normalized if ch.isalnum())
    return cleaned.lower()


def country_code_to_emoji(country_code: str) -> str:
    """Convert a two-letter ISO country code to a flag emoji."""
    cc = country_code.upper()
    if len(cc) != 2:
        return "⚽"
    base = ord("🇦") - ord("A")
    return chr(base + ord(cc[0])) + chr(base + ord(cc[1]))


class TeamAssetsRegistry:
    """Registry that resolves standardized team metadata."""

    def __init__(self):
        raw = _load_raw_data()
        self._teams: dict[str, TeamAssetRecord] = {}
        self._aliases: dict[str, str] = {}
        self._leagues: dict[str, dict[str, str]] = raw.get("leagues", {})

        for item in raw.get("teams", []):
            record = TeamAssetRecord(
                team_id=item["team_id"],
                official_name=item["official_name"],
                display_name=item["display_name"],
                short_name=item.get("short_name") or item["display_name"],
                tla=item.get("tla", ""),
                league_code=item["league_code"],
                league_name=self._leagues[item["league_code"]]["name"],
                country=item["country"],
                country_code=item["country_code"],
                aliases=item.get("aliases", []),
                primary_color=item.get("primary_color"),
            )
            self._teams[record.team_id] = record
            for alias in record.aliases:
                key = slugify(alias)
                self._aliases[key] = record.team_id

    def available_leagues(self) -> list[str]:
        return [meta["name"] for meta in self._leagues.values()]

    def get_league_info(self, league_code: str) -> dict[str, str] | None:
        return self._leagues.get(league_code)

    def get_team_by_id(self, team_id: str) -> TeamAssetRecord | None:
        return self._teams.get(team_id)

    def get_team(self, *, team_id: str | None = None, team_name: str | None = None) -> TeamAssetRecord | None:
        if team_id:
            return self.get_team_by_id(team_id)
        if not team_name:
            return None
        key = slugify(team_name)
        record_id = self._aliases.get(key)
        if record_id:
            return self._teams[record_id]
        return None

    def list_teams_for_league(self, league_code: str) -> list[TeamAssetRecord]:
        return [r for r in self._teams.values() if r.league_code == league_code]

    def all_team_records(self) -> Iterable[TeamAssetRecord]:
        """Return an iterable over all team asset records."""
        return self._teams.values()

    def standardize_team_name(self, name: str) -> str | None:
        record = self.get_team(team_name=name)
        return record.display_name if record else None

    def resolve_flag_png(self, team_name: str, size: int = DEFAULT_FLAG_SIZE) -> str | None:
        record = self.get_team(team_name=team_name)
        return record.flag_png(size) if record else None


_registry: TeamAssetsRegistry | None = None


def get_team_assets_registry() -> TeamAssetsRegistry:
    global _registry
    if _registry is None:
        _registry = TeamAssetsRegistry()
    return _registry


__all__ = [
    "TeamAssetRecord",
    "TeamAssetsRegistry",
    "get_team_assets_registry",
    "slugify",
    "country_code_to_emoji",
]
