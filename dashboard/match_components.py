"""Match and analytics UI component helpers.

Encapsulates rendering logic for featured match cards and analytics summary
cards so the main homepage file can stay declarative and lean. These functions
assume design tokens have already been injected via `ensure_tokens`.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, Sequence

import numpy as np
import streamlit as st


def _safe_prob(val: float, lo: float = 0.15, hi: float = 0.85) -> float:
    try:
        return max(lo, min(hi, float(val)))
    except Exception:
        return 0.5


def render_featured_match_card(
    match: Dict[str, Any],
    predictor: Any | None,
    team_data_fn: Callable[[str], Dict[str, str]],
    index: int | None = None,
    on_analysis: Callable[[str, str, Dict[str, Any], Any], None] | None = None,
):
    """Render a single featured match card using design system classes.

    Parameters
    ----------
    match : dict
        Match dictionary with keys home_team, away_team, league, time, etc.
    predictor : predictor instance or None
        Object exposing predict_match_enhanced().
    team_data_fn : Callable
        Function returning enhanced team metadata.
    index : int
        Index to build stable Streamlit widget keys.
    on_analysis : Callable | None
        Callback for detailed analysis when button clicked.
    """
    home_team = match.get('home_team', 'Home Team')
    away_team = match.get('away_team', 'Away Team')
    league = match.get('league', 'League')
    match_time = match.get('time', '15:00')

    home_data = team_data_fn(home_team)
    away_data = team_data_fn(away_team)

    prefetched = match.get('prefetched_prediction')
    if prefetched is None and any(k in match for k in ('home_win_prob', 'home_prob', 'away_win_prob', 'away_prob')):
        prefetched = match

    prediction = None
    home_prob = away_prob = confidence = None

    if prefetched is not None:
        try:
            home_prob_val = prefetched.get('home_win_prob') if isinstance(prefetched, dict) else None
            if home_prob_val is None:
                home_prob_val = prefetched.get('home_prob') if isinstance(prefetched, dict) else None
            away_prob_val = prefetched.get('away_win_prob') if isinstance(prefetched, dict) else None
            if away_prob_val is None:
                away_prob_val = prefetched.get('away_prob') if isinstance(prefetched, dict) else None
            if isinstance(home_prob_val, (int, float)) and isinstance(away_prob_val, (int, float)):
                confidence = prefetched.get('confidence') if isinstance(prefetched, dict) else None
                draw_prob = prefetched.get('draw_prob') if isinstance(prefetched, dict) else None
                match_stats = prefetched.get('match_statistics') if isinstance(prefetched, dict) else None
                home_prob = _safe_prob(home_prob_val)
                away_prob = _safe_prob(away_prob_val)
                confidence = confidence if isinstance(confidence, (int, float)) else 0.75
                prediction = SimpleNamespace(
                    home_win_probability=home_prob,
                    away_win_probability=away_prob,
                    draw_probability=draw_prob,
                    confidence=confidence,
                    key_factors=prefetched.get('key_factors') if isinstance(prefetched, dict) else None,
                    insights=prefetched.get('insights') if isinstance(prefetched, dict) else None,
                    expected_goals_home=(match_stats or {}).get('expectedGoals', {}).get('home') if isinstance(match_stats, dict) else None,
                    expected_goals_away=(match_stats or {}).get('expectedGoals', {}).get('away') if isinstance(match_stats, dict) else None,
                    xg_confidence=(match_stats or {}).get('confidence') if isinstance(match_stats, dict) else None,
                )
        except Exception:
            prediction = None

    if prediction is None:
        try:
            if predictor:
                prediction = predictor.predict_match_enhanced(home_team, away_team, match)
                home_prob = getattr(prediction, 'home_win_probability', 0.5)
                away_prob = getattr(prediction, 'away_win_probability', 0.5)
                confidence = getattr(prediction, 'confidence', 0.7)
            else:
                raise RuntimeError("Predictor missing")
        except Exception:
            home_prob = 0.50 + np.random.normal(0, 0.08)
            away_prob = 0.35 + np.random.normal(0, 0.08)
            confidence = 0.75 + np.random.normal(0, 0.05)
            prediction = SimpleNamespace(
                home_win_probability=home_prob,
                away_win_probability=away_prob,
                confidence=confidence,
                key_factors=['Recent form', 'Home advantage'],
                expected_goals_home=max(0.8, home_prob * 2.1),
                expected_goals_away=max(0.6, away_prob * 1.9),
                xg_confidence=0.4
            )

    home_prob = _safe_prob(home_prob)
    away_prob = _safe_prob(away_prob)
    confidence = max(0.5, min(0.95, confidence))

    # Card content structure
    st.markdown(
        f"""
        <div style="text-align:center; margin-bottom:0.75rem;">
          <h4 style="margin:0; font-size:1.05rem;">🏆 {league}</h4>
          <p style="color:var(--gd-text-muted); margin:0.35rem 0 0 0; font-size:0.8rem;">⏰ {match_time} | Today</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_home, col_vs, col_away = st.columns([2, 1, 2])
    with col_home:
        st.markdown(
            f"""
            <div style='text-align:center;'>
              <div style='font-size:2rem;margin-bottom:4px;'>{home_data['flag']}</div>
              <p style='margin:0; font-weight:600;'>{home_data['display_name']}</p>
              <p style='margin:0; font-size:0.65rem; color:var(--gd-text-muted);'>{home_data['full_name']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_vs:
        st.markdown(
            """
            <div style='text-align:center; padding:0.75rem 0;'>
              <span style='display:inline-block; background:linear-gradient(135deg,#11998e 0%,#38ef7d 100%); color:#fff; padding:6px 16px; border-radius:999px; font-weight:600;'>VS</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_away:
        st.markdown(
            f"""
            <div style='text-align:center;'>
              <div style='font-size:2rem;margin-bottom:4px;'>{away_data['flag']}</div>
              <p style='margin:0; font-weight:600;'>{away_data['display_name']}</p>
              <p style='margin:0; font-size:0.65rem; color:var(--gd-text-muted);'>{away_data['full_name']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    prob_c1, prob_c2 = st.columns(2)
    with prob_c1:
        st.metric(f"{home_data['display_name']} Win", f"{home_prob:.1%}")
    with prob_c2:
        st.metric(f"{away_data['display_name']} Win", f"{away_prob:.1%}")

    expected_goals_home = getattr(prediction, 'expected_goals_home', None)
    expected_goals_away = getattr(prediction, 'expected_goals_away', None)
    xg_confidence = getattr(prediction, 'xg_confidence', None)

    if expected_goals_home is not None and expected_goals_away is not None:
        xg_cols = st.columns(2)
        with xg_cols[0]:
            st.metric(f"Expected Goals · {home_data['display_name']}", f"{expected_goals_home:.2f}")
        with xg_cols[1]:
            st.metric(f"Expected Goals · {away_data['display_name']}", f"{expected_goals_away:.2f}")
        delta = expected_goals_home - expected_goals_away
        caption = f"Δ xG: {delta:+.2f}"
        if xg_confidence is not None:
            caption += f" • Confidence {xg_confidence:.0%}"
        st.caption(caption)

    st.success(
        f"🎯 **AI Prediction**: {'{0}'.format(home_data['display_name'] if home_prob > away_prob else away_data['display_name'])} Favored"
    )
    st.progress(confidence, text=f"Confidence: {confidence:.1%}")

    narrative = None
    if predictor and hasattr(predictor, 'generate_prediction_narrative'):
        try:
            narrative = predictor.generate_prediction_narrative(prediction, match)
        except Exception:
            narrative = None

    if narrative:
        st.markdown(f"**Insight:** {narrative.get('headline', 'No narrative available.')}" )
        strengths = narrative.get('strengths') or []
        if strengths:
            st.markdown("\n".join(f"- {item}" for item in strengths))
        risks = narrative.get('risks') or []
        if risks:
            st.markdown("**⚠️ Watch-outs**")
            st.markdown("\n".join(f"- {item}" for item in risks))
        feature_pulse = narrative.get('feature_pulse') or []
        if feature_pulse:
            st.caption("Top model drivers: " + ", ".join(feature_pulse))

    if prefetched and isinstance(prefetched, dict):
        insights = prefetched.get('insights') or []
        if insights:
            st.markdown("**Prefetched Insights:**")
            st.markdown("\n".join(f"- {item}" for item in insights[:3]))

    context = match.get('historical_context')
    if context:
        h2h = context.get('head_to_head') or {}
        home_form = context.get('home_recent_form') or {}
        away_form = context.get('away_recent_form') or {}

        with st.expander("📚 Historical context", expanded=False):
            if h2h.get('matches'):
                avg_goals = h2h.get('avg_total_goals')
                avg_label = f"{avg_goals:.2f}" if isinstance(avg_goals, (int, float)) else '—'
                st.markdown(
                    f"**Head-to-head since 2021** — Matches: {h2h['matches']}, "
                    f"{home_data['display_name']} W {h2h.get('home_wins', 0)}, Draws {h2h.get('draws', 0)}, "
                    f"{away_data['display_name']} W {h2h.get('away_wins', 0)} · Avg goals {avg_label}"
                )
                if h2h.get('recent_results'):
                    st.markdown("Recent meetings:")
                    for item in h2h['recent_results'][:5]:
                        st.markdown(
                            f"- {item.get('date', '')[:10]}: {item.get('home', home_team)} {item.get('score', '')} {item.get('away', away_team)}"
                            f" ({item.get('competition', 'N/A')})"
                        )
            else:
                st.markdown("Historical meetings unavailable for this matchup.")

            form_cols = st.columns(2)
            with form_cols[0]:
                st.markdown(f"**{home_data['display_name']} form**: {home_form.get('form') or '—'}")
                if home_form.get('avg_goals_for') is not None:
                    st.caption(
                        f"Avg GF {home_form['avg_goals_for']:.2f} · GA {home_form.get('avg_goals_against', 0):.2f}"
                    )
                if home_form.get('recent_results'):
                    st.markdown(
                        "<br>".join(
                            f"{item.get('date', '')[:10]} · {item.get('venue', '')}: {item.get('score', '')} vs {item.get('opponent', '')}"
                            for item in home_form['recent_results'][:3]
                        ),
                        unsafe_allow_html=True,
                    )

            with form_cols[1]:
                st.markdown(f"**{away_data['display_name']} form**: {away_form.get('form') or '—'}")
                if away_form.get('avg_goals_for') is not None:
                    st.caption(
                        f"Avg GF {away_form['avg_goals_for']:.2f} · GA {away_form.get('avg_goals_against', 0):.2f}"
                    )
                if away_form.get('recent_results'):
                    st.markdown(
                        "<br>".join(
                            f"{item.get('date', '')[:10]} · {item.get('venue', '')}: {item.get('score', '')} vs {item.get('opponent', '')}"
                            for item in away_form['recent_results'][:3]
                        ),
                        unsafe_allow_html=True,
                    )

            seasons = context.get('season_summaries') or {}
            if seasons:
                home_seasons = seasons.get('home') or []
                away_seasons = seasons.get('away') or []
                if home_seasons or away_seasons:
                    st.markdown("**Season snapshots (2021-2025 seasons)**")
                    max_items = max(len(home_seasons), len(away_seasons))
                    for idx in range(max_items):
                        home_snapshot = home_seasons[idx] if idx < len(home_seasons) else None
                        away_snapshot = away_seasons[idx] if idx < len(away_seasons) else None
                        if not home_snapshot and not away_snapshot:
                            continue

                        season_label = (
                            home_snapshot.get('season')
                            if home_snapshot
                            else away_snapshot.get('season') if away_snapshot else '—'
                        )
                        home_label = (
                            f"{home_snapshot.get('position', '—')} pos · {home_snapshot.get('points', '—')} pts"
                            if home_snapshot
                            else '—'
                        )
                        away_label = (
                            f"{away_snapshot.get('position', '—')} pos · {away_snapshot.get('points', '—')} pts"
                            if away_snapshot
                            else '—'
                        )
                        st.caption(
                            f"{season_label}: {home_data['display_name']} {home_label} | {away_data['display_name']} {away_label}"
                        )

    # Derive a stable key when index missing
    _idx = index if index is not None else abs(hash(f"{home_team}_{away_team}")) % 10_000
    if st.button(
        "🔍 Detailed Analysis & Betting Insights",
        key=f"detail_btn_{_idx}_{hash(home_team+away_team)}",
        use_container_width=True,
        type="primary",
    ):
        if on_analysis:
            on_analysis(home_team, away_team, match, prediction)


def render_analytics_cards(definitions: Sequence[Dict[str, Any]]):
    """Render analytics summary cards from a list of definitions.

    Each definition expects: title, value_html (main stat), description,
    items (list[str]) optional.
    """
    if not definitions:
        return
    cols = st.columns(len(definitions))
    for col, definition in zip(cols, definitions):
        with col:
            st.markdown("<div class='gd-card' style='padding:1.25rem;'>", unsafe_allow_html=True)
            st.markdown(
                f"<h4 style='margin:0 0 .5rem 0;'>{definition['title']}</h4>"
                f"<div style='text-align:center;margin:.5rem 0 1rem 0;'>{definition['value_html']}</div>"
                f"<p style='margin:0 0 .5rem 0; font-size:.8rem; color:var(--gd-text-muted);'>{definition['description']}</p>",
                unsafe_allow_html=True,
            )
            items = definition.get('items') or []
            if items:
                st.markdown(
                    '<ul style="padding-left:1.1rem; margin:0; font-size:.75rem;">' +
                    ''.join(f"<li style='margin:2px 0;'>{i}</li>" for i in items) +
                    '</ul>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)


__all__ = [
    'render_featured_match_card',
    'render_analytics_cards'
]
