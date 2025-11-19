"""Reusable team display components leveraging the centralized team assets registry."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st

from utils.team_data_enhancer import TeamDataEnhancer, team_enhancer


_TEAM_DISPLAY_STYLE_KEY = "_team_display_styles_injected"


@dataclass(slots=True)
class TeamChipOptions:
    """Display configuration for a team chip."""

    show_country: bool = True
    show_league: bool = True
    emphasize_badge: bool = False
    flag_variant: str = "emoji"  # emoji|png
    subtitle: Optional[str] = None


class TeamDisplayComponent:
    """Centralized helpers for rendering team visuals in Streamlit UIs."""

    CHIP_STYLES = """
    <style>
    .gd-team-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        background: rgba(20, 20, 20, 0.35);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.15);
        backdrop-filter: blur(12px);
        color: #f8fafc;
        font-size: 0.95rem;
        line-height: 1.2;
        min-width: 0;
        max-width: 100%;
    }
    .gd-team-chip__flag {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2rem;
        height: 2rem;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.08);
        font-size: 1.35rem;
        flex-shrink: 0;
    }
    .gd-team-chip__flag img {
        width: 100%;
        height: 100%;
        border-radius: 50%;
        object-fit: cover;
        border: 1px solid rgba(255, 255, 255, 0.22);
        background: rgba(255, 255, 255, 0.92);
    }
    .gd-team-chip__meta {
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
        min-width: 0;
    }
    .gd-team-chip__name {
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .gd-team-chip__subtitle {
        font-size: 0.78rem;
        opacity: 0.8;
        color: rgba(226, 232, 240, 0.9);
        display: flex;
        gap: 0.35rem;
        flex-wrap: wrap;
        align-items: center;
    }
    .gd-team-chip__subtitle span {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
    }
    .gd-matchup-banner {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.2rem;
        align-items: center;
        width: 100%;
    }
    .gd-matchup-banner__center {
        text-align: center;
        font-weight: 600;
        color: rgba(226, 232, 240, 0.85);
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .gd-matchup-banner__meta {
        font-size: 0.75rem;
        opacity: 0.75;
        margin-top: 0.25rem;
    }
    </style>
    """

    def __init__(self, enhancer: TeamDataEnhancer | None = None):
        self.enhancer = enhancer or team_enhancer

    def _ensure_styles(self) -> None:
        try:
            if not st.session_state.get(_TEAM_DISPLAY_STYLE_KEY):
                st.markdown(self.CHIP_STYLES, unsafe_allow_html=True)
                st.session_state[_TEAM_DISPLAY_STYLE_KEY] = True
        except RuntimeError:
            # Streamlit session state is unavailable (e.g., during CLI execution).
            pass

    def _compose_subtitle(self, metadata: Dict[str, Any], options: TeamChipOptions) -> str:
        if options.subtitle:
            return options.subtitle

        subtitle_bits: list[str] = []
        if options.show_country and metadata.get("country"):
            subtitle_bits.append(
                f"{metadata.get('country_flag', '')} {metadata.get('country', '')}".strip()
            )
        if options.show_league and metadata.get("league"):
            subtitle_bits.append(metadata.get("league", ""))
        return " • ".join(bit for bit in subtitle_bits if bit)

    def _flag_markup(self, metadata: Dict[str, Any], options: TeamChipOptions) -> str:
        if options.flag_variant == "png" and metadata.get("flag_png"):
            return (
                "<span class=\"gd-team-chip__flag\">"
                f"<img src='{metadata['flag_png']}' alt='{metadata.get('country','')}' />"
                "</span>"
            )
        return f"<span class=\"gd-team-chip__flag\">{metadata.get('flag', '⚽')}</span>"

    def _build_chip_html(self, metadata: Dict[str, Any], options: TeamChipOptions) -> str:
        safe_color = metadata.get("safe_color") or metadata.get("color") or "#667EEA"
        subtitle = self._compose_subtitle(metadata, options)
        badge_html = ""
        if options.emphasize_badge:
            badge_url = metadata.get("badge_url")
            if badge_url:
                badge_html = (
                    "<div class='gd-team-chip__subtitle'>"
                    f"<span><img src='{badge_url}' alt='{metadata.get('display_name','')}' style='width:20px;height:20px;border-radius:4px;object-fit:contain;border:1px solid rgba(255,255,255,0.25);' /></span>"
                    "</div>"
                )

        flag_html = self._flag_markup(metadata, options)

        subtitle_html = (
            f"<div class='gd-team-chip__subtitle'>{subtitle}</div>" if subtitle else ""
        )
        return (
            "<div class='gd-team-chip' style="
            f"border-color: {safe_color}33; box-shadow: 0 8px 32px {safe_color}1A;"
            "">"
            f"{flag_html}"
            "<div class='gd-team-chip__meta'>"
            f"<div class='gd-team-chip__name'>{metadata.get('display_name', 'Unknown Team')}</div>"
            f"{subtitle_html}{badge_html}"
            "</div></div>"
        )

    def get_team_metadata(self, team_name: str, league: str | None = None) -> Dict[str, Any]:
    base = self.enhancer.get_team_data(team_name)
    enhanced = self.enhancer.get_team_enhancement(team_name, league)
    metadata: Dict[str, Any] = {**base, **enhanced}
    metadata.setdefault('flag', metadata.get('country_flag') or '⚽')
    metadata.setdefault('country_flag', metadata.get('flag'))
    metadata.setdefault('safe_color', metadata.get('color', '#667EEA') or '#667EEA')
    metadata.setdefault('flag_png', base.get('flag_png'))
    metadata.setdefault('flag_svg', base.get('flag_svg'))
    return metadata

    def render_team_chip(
        self,
        team_name: str,
        *,
        league: str | None = None,
        options: TeamChipOptions | None = None,
    ) -> Dict[str, Any]:
        """Render a stylized team chip and return the underlying metadata."""
        options = options or TeamChipOptions()
        metadata = self.get_team_metadata(team_name, league)
        self._ensure_styles()
        st.markdown(self._build_chip_html(metadata, options), unsafe_allow_html=True)
        return metadata

    def render_matchup_banner(
        self,
        home_team: str,
        away_team: str,
        *,
        league: str | None = None,
        kickoff: datetime | str | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Render a matchup banner and return resolved metadata for both teams."""
        self._ensure_styles()
        home_meta = self.get_team_metadata(home_team, league)
        away_meta = self.get_team_metadata(away_team, league)

        kickoff_text = ""
        if isinstance(kickoff, datetime):
            kickoff_text = kickoff.strftime("%d %b %Y • %H:%M")
        elif isinstance(kickoff, str):
            kickoff_text = kickoff

        home_html = self._build_chip_html(home_meta, TeamChipOptions())
        away_html = self._build_chip_html(away_meta, TeamChipOptions())
        center_html = "<div class='gd-matchup-banner__center'>VS</div>"
        if league or kickoff_text:
            center_meta = " • ".join(filter(None, [league, kickoff_text]))
            if center_meta:
                center_html += f"<div class='gd-matchup-banner__meta'>{center_meta}</div>"

        st.markdown(
            "<div class='gd-matchup-banner'>"
            f"{home_html}{center_html}{away_html}"
            "</div>",
            unsafe_allow_html=True,
        )

        return {"home": home_meta, "away": away_meta}


_display_component: TeamDisplayComponent | None = None


def get_team_display_component(enhancer: TeamDataEnhancer | None = None) -> TeamDisplayComponent:
    """Return a shared TeamDisplayComponent instance."""
    global _display_component
    if _display_component is None:
        _display_component = TeamDisplayComponent(enhancer=enhancer or team_enhancer)
    return _display_component

``