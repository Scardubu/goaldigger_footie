"""
Enhanced team data management system with proper names, flags, and country information.

Provides comprehensive team metadata including flags, logos, country codes,
and standardized naming for all major football leagues.
"""
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from utils.team_assets_registry import (
    TeamAssetRecord,
    country_code_to_emoji,
    get_team_assets_registry,
    slugify,
)

if TYPE_CHECKING:  # pragma: no cover - used only for typing
    from utils.enhanced_team_data_manager import EnhancedTeamDataManager, TeamMetadata
else:  # Fallback aliases keep runtime lightweight if module is unavailable
    EnhancedTeamDataManager = Any  # type: ignore[misc,assignment]
    TeamMetadata = Any  # type: ignore[misc,assignment]

try:
    from utils.enhanced_team_data_manager import get_enhanced_team_data_manager
except Exception:  # pragma: no cover - optional enhancement layer
    get_enhanced_team_data_manager = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class TeamDataEnhancer:
    def enhance_team_name(
        self,
        team_name: str,
        league_hint: str | None = None,
        country_hint: str | None = None,
    ) -> str:
        """Return enhanced display name for a team with contextual flag/league info.

        Args:
            team_name: Raw team name to enhance.
            league_hint: Optional league code or human-readable league name supplied by caller.
            country_hint: Optional country hint to prefer when determining flags.
        """

        data = self.get_team_data(team_name)
        display = data.get("display_name", team_name)
        flag = data.get("flag") or data.get("country_flag") or "⚽"

        if country_hint and data.get("country"):
            if data["country"].lower() != country_hint.lower():
                display = f"{display} · {country_hint}"
        elif country_hint and not data.get("country"):
            display = f"{display} · {country_hint}"

        if league_hint:
            league_aliases = {
                data.get("league_code", "").lower(),
                data.get("league", "").lower(),
            }
            if league_hint.lower() not in league_aliases:
                display = f"{display} · {league_hint}"

        return f"{flag} {display}"
    """Enhanced team data management with proper metadata."""
    
    def __init__(self):
        self.assets_registry = get_team_assets_registry()
        self._metadata_by_team_id: dict[str, dict[str, Any]] = {}
        self._alias_index: dict[str, str] = {}
        self._enhanced_manager: EnhancedTeamDataManager | None = None
        if get_enhanced_team_data_manager:
            try:
                self._enhanced_manager = get_enhanced_team_data_manager()
            except Exception:
                logger.debug("EnhancedTeamDataManager unavailable; proceeding with registry-only metadata")
        self.team_database = self._initialize_team_database()
        
    def _initialize_team_database(self) -> dict[str, dict[str, Any]]:
        """Load the canonical team database from the centralized registry."""
        database: dict[str, dict[str, Any]] = {}
        self._metadata_by_team_id.clear()
        self._alias_index.clear()

        for record in self.assets_registry.all_team_records():
            alias_bundle = self._all_aliases(record)
            metadata = self._record_to_metadata(record, alias_bundle)
            self._metadata_by_team_id[record.team_id] = metadata
            database[record.display_name] = metadata
            for alias in alias_bundle:
                normalized = self._normalize_alias(alias)
                if not normalized:
                    continue
                if normalized not in self._alias_index:
                    self._alias_index[normalized] = record.display_name

        return database
    
    def _find_known_team_data(self, team_name: str) -> dict[str, str] | None:
        """Attempt to resolve a team to a known database entry without fallback."""
        if not team_name:
            return None

        if team_name in self.team_database:
            return self.team_database[team_name].copy()

        normalized = self._normalize_alias(team_name)
        canonical_key = self._alias_index.get(normalized)
        if canonical_key and canonical_key in self.team_database:
            return self.team_database[canonical_key].copy()

        record = self.assets_registry.get_team(team_name=team_name)
        if record:
            metadata = self._metadata_by_team_id.get(record.team_id)
            if metadata:
                return metadata.copy()

        return None

    def get_team_data(self, team_name: str) -> dict[str, str]:
        """Get enhanced team data for a given team name."""
        match = self._find_known_team_data(team_name)
        if match:
            return match

        # Create intelligent default based on team name patterns
        return self._create_default_team_data(team_name)
    
    def _create_default_team_data(self, team_name: str) -> dict[str, str]:
        """Create intelligent default team data based on name patterns."""
        name_lower = team_name.lower()

        # Determine league and country based on team name patterns
        if any(pattern in name_lower for pattern in ['city', 'united', 'arsenal', 'chelsea', 'liverpool', 'tottenham']):
            country = "England"
            country_code = "GB"
            country_flag = "🏴"
            league = "Premier League"
            league_code = "PL"
        elif any(pattern in name_lower for pattern in ['madrid', 'barcelona', 'sevilla', 'valencia', 'atletico']):
            country = "Spain"
            country_code = "ES"
            country_flag = "🇪🇸"
            league = "La Liga"
            league_code = "PD"
        elif any(pattern in name_lower for pattern in ['bayern', 'dortmund', 'leipzig', 'leverkusen']):
            country = "Germany"
            country_code = "DE"
            country_flag = "🇩🇪"
            league = "Bundesliga"
            league_code = "BL1"
        elif any(pattern in name_lower for pattern in ['juventus', 'milan', 'roma', 'napoli', 'inter']):
            country = "Italy"
            country_code = "IT"
            country_flag = "🇮🇹"
            league = "Serie A"
            league_code = "SA"
        elif any(pattern in name_lower for pattern in ['psg', 'marseille', 'lyon', 'monaco']):
            country = "France"
            country_code = "FR"
            country_flag = "🇫🇷"
            league = "Ligue 1"
            league_code = "FL1"
        elif any(pattern in name_lower for pattern in ['benfica', 'porto', 'sporting', 'braga']):
            country = "Portugal"
            country_code = "PT"
            country_flag = "🇵🇹"
            league = "Primeira Liga"
            league_code = "PPL"
        else:
            country = "Unknown"
            country_code = "XX"
            country_flag = "🏳️"
            league = "Unknown League"
            league_code = "XX"
        
        default_flag = country_code_to_emoji(country_code) if country_code != "XX" else country_flag

        return {
            "full_name": team_name,
            "display_name": team_name,
            "short_name": team_name[:3].upper(),
            "tla": team_name[:3].upper(),
            "flag": default_flag or "⚽",
            "country": country,
            "country_code": country_code,
            "country_flag": country_flag,
            "color": "#667eea",
            "league": league,
            "league_code": league_code,
            "venue": f"{team_name} Stadium",
            "capacity": 30000
        }
    
    def get_all_teams_by_league(self, league_code: str) -> list[dict[str, str]]:
        """Get all teams for a specific league."""
        league_code = (league_code or "").upper()
        records = self.assets_registry.list_teams_for_league(league_code)
        results: list[dict[str, str]] = []
        for record in records:
            metadata = self._metadata_by_team_id.get(record.team_id)
            if metadata:
                results.append(metadata.copy())
        if results:
            return sorted(results, key=lambda item: item.get("display_name", ""))
        # fall back to previously initialized database iteration for unknown leagues
        legacy_matches = [
            data.copy()
            for data in self.team_database.values()
            if data.get("league_code") == league_code
        ]
        return sorted(legacy_matches, key=lambda item: item.get("display_name", ""))

    # --- Newly added method for tests/UI that expect richer enhancement ---
    def get_enhanced_team_data(self, team_name: str, league: str = None) -> dict[str, str]:
        """Return enriched team dictionary with additional convenience fields.

        Args:
            team_name: Raw or partially formatted team name.
            league: Optional human league name (e.g., 'Premier League'). If provided
                and differs from detected league, a note is preserved.

        Returns:
            Dict containing original metadata plus:
              - display_with_flag
              - league_mismatch (bool)
              - safe_color (hex fallback)
        """
        data = self.get_team_data(team_name).copy()
        data.setdefault('display_name', team_name)
        data.setdefault('flag', '⚽')
        # Standardize the display_name for consistency
        from utils.team_name_standardizer import standardize_team_name
        data['display_name'] = standardize_team_name(data['display_name'])
        data['display_with_flag'] = f"{data['flag']} {data['display_name']}"
        if league:
            # Compare league names case-insensitively
            detected = data.get('league', '')
            mismatch = detected and league and detected.lower() != league.lower()
            data['league_mismatch'] = bool(mismatch)
            if mismatch:
                data['provided_league'] = league
        else:
            data['league_mismatch'] = False
        # Provide deterministic safe color for UI if missing
        data['safe_color'] = data.get('color', '#667eea') or '#667eea'
        return data

    def get_team_icon(self, team_name: str) -> str:
        """Return the preferred emoji/icon for a given team."""
        data = self.get_team_data(team_name)
        return data.get('flag') or data.get('country_flag') or '⚽'

    def get_team_color(self, team_name: str, fallback: str = '#667eea') -> str:
        """Return the primary brand color associated with a team."""
        data = self.get_team_data(team_name)
        color = data.get('color') or fallback
        normalized = self._normalize_hex_color(color)
        return normalized or fallback

    def get_country_flag(self, team_or_country: str) -> str:
        """Return the country flag associated with a team or country name."""
        data = self._find_known_team_data(team_or_country)
        if data:
            return data.get('country_flag') or data.get('flag') or '⚽'

        normalized = team_or_country.strip().lower()
        for entry in self.team_database.values():
            if entry.get('country', '').lower() == normalized:
                return entry.get('country_flag') or entry.get('flag') or '⚽'

        country_overrides = {
            'england': '🏴',
            'united kingdom': '🏴',
            'spain': '🇪🇸',
            'germany': '🇩🇪',
            'italy': '🇮🇹',
            'france': '🇫🇷',
            'portugal': '🇵🇹',
            'brazil': '🇧🇷',
            'argentina': '🇦🇷',
            'netherlands': '🇳🇱',
            'belgium': '🇧🇪',
        }
        return country_overrides.get(normalized, '⚽')

    def get_team_enhancement(self, team_name: str, league: str | None = None) -> dict[str, str]:
        """Return a consolidated enhancement payload used throughout the UI."""
        data = self.get_enhanced_team_data(team_name, league)
        primary = self.get_team_color(team_name)
        secondary = self._derive_secondary_color(primary)

        enhancement = {
            'full_name': data.get('full_name', team_name),
            'display_name': data.get('display_name', team_name),
            'short_name': data.get('short_name', team_name[:3].upper()),
            'country': data.get('country', 'Unknown'),
            'country_flag': data.get('country_flag') or self.get_country_flag(data.get('country', '')),
            'league': data.get('league'),
            'league_code': data.get('league_code'),
            'primary_color': primary,
            'secondary_color': secondary,
            'accent_color': self._derive_accent_color(primary),
            'flag': data.get('flag', '⚽'),
            'icon': self.get_team_icon(team_name),
            'display_with_flag': data.get('display_with_flag'),
            'league_mismatch': data.get('league_mismatch', False),
            'safe_color': data.get('safe_color'),
        }

        enhancement['glass_gradient'] = (
            f"linear-gradient(135deg, {primary}CC 0%, {secondary}99 100%)"
        )

        return enhancement

    def enhance_team_data(self, team_name: str, league: str | None = None) -> dict[str, str]:
        """Backward compatible alias for cached utility modules."""
        return self.get_team_enhancement(team_name, league)

    def create_match_title(
        self,
        home_team: str,
        away_team: str,
        league: str | None = None,
        match_date: datetime | None = None,
    ) -> str:
        """Craft a premium match title combining teams, flags, and league context."""
        home = self.get_enhanced_team_data(home_team, league)
        away = self.get_enhanced_team_data(away_team, league)

        if home_team == away_team:
            return f"{home.get('flag', '⚽')} {home.get('display_name', home_team)} Friendly"

        inferred_league = league or self._resolve_league_label(home, away)
        date_fragment = ''
        if match_date:
            date_fragment = f" • {match_date.strftime('%d %b %Y')}"

        return (
            f"{home.get('flag', '⚽')} {home.get('display_name', home_team)} vs "
            f"{away.get('display_name', away_team)} {away.get('flag', '⚽')}"
            f" • {inferred_league}{date_fragment}"
        )

    def _normalize_hex_color(self, color: str | None) -> str | None:
        if not color:
            return None
        color = color.strip().lstrip('#')
        if len(color) == 3:
            color = ''.join(ch * 2 for ch in color)
        if len(color) != 6:
            return None
        try:
            int(color, 16)
        except ValueError:
            return None
        return f"#{color.upper()}"

    def _derive_secondary_color(self, primary: str) -> str:
        normalized = self._normalize_hex_color(primary) or '#667EEA'
        r, g, b = self._hex_to_rgb(normalized)
        blend = lambda channel: min(255, int(channel + (255 - channel) * 0.35))
        return self._rgb_to_hex((blend(r), blend(g), blend(b)))

    def _derive_accent_color(self, primary: str) -> str:
        normalized = self._normalize_hex_color(primary) or '#667EEA'
        r, g, b = self._hex_to_rgb(normalized)
        blend = lambda channel: max(0, int(channel * 0.65))
        return self._rgb_to_hex((blend(r), blend(g), blend(b)))

    def _hex_to_rgb(self, color: str) -> tuple[int, int, int]:
        color = self._normalize_hex_color(color) or '#667EEA'
        color = color.lstrip('#')
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    def _rgb_to_hex(self, rgb: tuple[int, int, int]) -> str:
        return '#{0:02X}{1:02X}{2:02X}'.format(*rgb)

    def _resolve_league_label(self, home: dict[str, str], away: dict[str, str]) -> str:
        home_league = home.get('league') or home.get('league_code') or 'League'
        away_league = away.get('league') or away.get('league_code') or 'League'
        if home_league.lower() == away_league.lower():
            return home_league
        return f"Inter-League • {home_league} x {away_league}"
    
    def get_league_info(self, league_code: str) -> dict[str, str]:
        """Get league information from the registry with sensible fallbacks."""
        if not league_code:
            return {"name": "Unknown League", "country": "Unknown", "flag": "🏳️"}

        normalized = league_code.upper()
        info = self.assets_registry.get_league_info(normalized)
        if info:
            flag = country_code_to_emoji(info.get("country_code", ""))
            return {
                "code": normalized,
                "name": info.get("name", normalized),
                "country": info.get("country", "Unknown"),
                "country_code": info.get("country_code"),
                "flag": flag if flag and flag != "⚽" else "🏳️",
                "country_flag": flag if flag and flag != "⚽" else "🏳️",
            }

        if self._enhanced_manager:
            mappings = getattr(self._enhanced_manager, "league_mappings", {}) or {}
            mapped_name = mappings.get(normalized)
            if mapped_name:
                return {
                    "code": normalized,
                    "name": mapped_name,
                    "country": "Unknown",
                    "flag": "🏳️",
                    "country_flag": "🏳️",
                }

        return {
            "code": normalized,
            "name": normalized,
            "country": "Unknown",
            "flag": "🏳️",
            "country_flag": "🏳️",
        }

    def _normalize_alias(self, alias: str | None) -> str:
        if not alias:
            return ""
        return slugify(alias)

    def _all_aliases(self, record: TeamAssetRecord) -> list[str]:
        alias_set = set(record.aliases)
        alias_set.update(
            filter(
                None,
                [
                    record.display_name,
                    record.official_name,
                    record.short_name,
                    record.tla,
                    record.team_id,
                    record.display_name.replace(" FC", ""),
                ],
            )
        )
        if record.tla:
            alias_set.add(record.tla.upper())
        return sorted({alias.strip() for alias in alias_set if alias and alias.strip()})

    def _lookup_enhanced_metadata(self, aliases: list[str]) -> TeamMetadata | None:
        if not self._enhanced_manager:
            return None

        for alias in aliases:
            try:
                match = self._enhanced_manager.resolve_team(alias)
            except Exception:
                match = None
            if match:
                return match
        return None

    def _apply_enhanced_metadata(
        self, base: dict[str, Any], match: TeamMetadata | None
    ) -> None:
        if not match:
            return

        if match.stadium:
            base["venue"] = match.stadium
        if match.capacity:
            base["capacity"] = match.capacity
        if match.colors:
            primary = match.colors.get("primary")
            secondary = match.colors.get("secondary")
            if primary:
                base["color"] = self._normalize_hex_color(primary) or base["color"]
            if secondary:
                base["secondary_color"] = self._normalize_hex_color(secondary) or base[
                    "secondary_color"
                ]
        if hasattr(match, "aliases") and match.aliases:
            base_aliases = set(base.get("aliases", []))
            base_aliases.update(match.aliases)
            base["aliases"] = sorted({alias for alias in base_aliases if alias})
        if getattr(match, "country", None):
            base["country"] = match.country or base["country"]
        if getattr(match, "league", None):
            base["league"] = match.league or base["league"]
        if getattr(match, "short_name", None):
            base["short_name"] = match.short_name or base["short_name"]
        # Do not override display_name from enhanced metadata
        # if getattr(match, "display_name", None):
        #     base["display_name"] = match.display_name or base["display_name"]

    def _record_to_metadata(self, record: TeamAssetRecord, aliases: list[str]) -> dict[str, Any]:
        league_info = self.assets_registry.get_league_info(record.league_code) or {}
        league_name = league_info.get("name", record.league_name)
        league_country = league_info.get("country", record.country)
        league_country_code = league_info.get("country_code", record.country_code)
        primary_color = self._normalize_hex_color(record.primary_color) or "#667EEA"

        metadata: dict[str, Any] = {
            "team_id": record.team_id,
            "full_name": record.official_name,
            "display_name": record.display_name,
            "canonical_name": record.display_name,
            "short_name": record.short_name or record.display_name,
            "tla": (record.tla or record.short_name or record.display_name[:3]).upper(),
            "flag": record.flag_emoji,
            "country": record.country,
            "country_code": record.country_code,
            "country_flag": record.flag_emoji,
            "flag_png": record.flag_png(),
            "flag_svg": record.flag_svg(),
            "color": primary_color,
            "secondary_color": self._derive_secondary_color(primary_color),
            "accent_color": self._derive_accent_color(primary_color),
            "league": league_name,
            "league_code": record.league_code,
            "league_country": league_country,
            "league_country_code": league_country_code,
            "league_flag": country_code_to_emoji(league_country_code),
            "aliases": aliases,
            "source": "team_assets_registry",
            "venue": None,
            "capacity": None,
            "badge_url": None,
            "website": None,
        }
        enhanced = self._lookup_enhanced_metadata(aliases)
        self._apply_enhanced_metadata(metadata, enhanced)
        if enhanced:
            badge = getattr(enhanced, "badge_url", None)
            if badge:
                metadata["badge_url"] = badge
            website = getattr(enhanced, "website", None)
            if website:
                metadata["website"] = website
            founded = getattr(enhanced, "founded", None)
            if founded:
                metadata["founded"] = founded
            current_form = getattr(enhanced, "current_form", None)
            if current_form:
                metadata["current_form"] = current_form

        primary_effective = metadata.get("color", primary_color)
        metadata["secondary_color"] = metadata.get("secondary_color") or self._derive_secondary_color(
            primary_effective
        )
        metadata["accent_color"] = self._derive_accent_color(primary_effective)
        metadata.setdefault("capacity", 0)
        metadata.setdefault("venue", None)
        # Standardize the display_name
        from utils.team_name_standardizer import standardize_team_name
        metadata["display_name"] = standardize_team_name(metadata["display_name"])
        metadata["display_with_flag"] = f"{metadata['flag']} {metadata['display_name']}"
        return metadata


# Global instance
team_enhancer = TeamDataEnhancer()


def get_enhanced_team_data(team_name: str) -> dict[str, str]:
    """Convenience function to get enhanced team data."""
    # Backwards compatibility: original helper returned basic data; now expose richer structure.
    return team_enhancer.get_enhanced_team_data(team_name)


def get_teams_by_league(league_code: str) -> list[dict[str, str]]:
    """Convenience function to get teams by league."""
    return team_enhancer.get_all_teams_by_league(league_code)


def get_league_info(league_code: str) -> dict[str, str]:
    """Convenience function to get league information."""
    return team_enhancer.get_league_info(league_code)