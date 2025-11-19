"""
Matches list component for the dashboard.
Handles displaying and filtering upcoming matches.
"""
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

from dashboard.components.ui_elements import create_themed_card
from dashboard.components.unified_design_system import \
    get_unified_design_system
from dashboard.error_log import log_error

logger = logging.getLogger(__name__)


def render_matches_list(
    matches_df: pd.DataFrame,
    on_match_select_fn: Callable[[str], None],
    selected_match_id: Optional[str] = None,
) -> None:
    """
    Render a list of upcoming matches with filtering and selection capabilities.

    Args:
        matches_df: DataFrame containing match information.
        on_match_select_fn: Callback function when a match is selected.
        selected_match_id: Currently selected match ID.
    """
    try:
        # Attempt to acquire the unified design system; fall back if unavailable
        uds = None
        try:
            uds = get_unified_design_system()
            uds.inject_unified_css(dashboard_type="premium")
        except Exception:
            uds = None

        if matches_df.empty:
            st.info("No matches found for the selected criteria.")
            return

        required_columns = ["id", "home_team", "away_team", "match_date", "competition"]
        missing_columns = [col for col in required_columns if col not in matches_df.columns]
        if missing_columns:
            st.error(f"Matches data is missing required columns: {', '.join(missing_columns)}")
            return

        if not pd.api.types.is_datetime64_any_dtype(matches_df["match_date"]):
            matches_df["match_date"] = pd.to_datetime(matches_df["match_date"], errors="coerce")

        matches_df = matches_df.sort_values(by="match_date").dropna(subset=["match_date"])

        search_term = st.text_input("Search by team or competition", "")
        
        filtered_df = matches_df
        if search_term:
            filtered_df = matches_df[
                matches_df["home_team"].str.contains(search_term, case=False, na=False) |
                matches_df["away_team"].str.contains(search_term, case=False, na=False) |
                matches_df["competition"].str.contains(search_term, case=False, na=False)
            ]

        if filtered_df.empty:
            st.info("No matches found for your search.")
            return

        st.caption(f"Displaying {len(filtered_df)} matches.")

        competitions = filtered_df["competition"].unique()
        for competition in competitions:
            comp_matches = filtered_df[filtered_df["competition"] == competition]
            with st.expander(f"{competition} ({len(comp_matches)} matches)", expanded=True):
                dates = sorted(comp_matches["match_date"].dt.date.unique())
                for date in dates:
                    st.markdown(f"##### {date.strftime('%A, %d %B %Y')}")
                    date_matches = comp_matches[comp_matches["match_date"].dt.date == date]

                    for _, match in date_matches.iterrows():
                        match_id = str(match.get("id", ""))
                        home_team = match.get("home_team", "N/A")
                        away_team = match.get("away_team", "N/A")
                        match_time = match["match_date"].strftime("%H:%M")

                        is_selected = match_id == selected_match_id
                        
                        card_html = f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
                            <div>
                                <span style="font-weight: 600;">{home_team}</span> vs <span style="font-weight: 600;">{away_team}</span>
                            </div>
                            <div style="font-size: 0.9em; color: #666;">
                                {match_time}
                            </div>
                        </div>
                        """

                        # Render via UnifiedDesignSystem when available, otherwise fallback to themed card
                        if uds:
                            try:
                                def _match_card():
                                    st.markdown(card_html, unsafe_allow_html=True)

                                uds.create_unified_card(_match_card)
                            except Exception:
                                bg_color = "#e3f2fd" if is_selected else "#f8f9fa"
                                create_themed_card(
                                    title="",
                                    content=card_html,
                                    bg_color=bg_color,
                                    border_color="#1e88e5" if is_selected else "#dee2e6",
                                )
                        else:
                            bg_color = "#e3f2fd" if is_selected else "#f8f9fa"
                            create_themed_card(
                                title="",
                                content=card_html,
                                bg_color=bg_color,
                                border_color="#1e88e5" if is_selected else "#dee2e6",
                            )
                        
                        if st.button("View Details", key=f"btn_view_{match_id}", use_container_width=True):
                            on_match_select_fn(match_id)
                        st.markdown("---")
    except Exception as e:
        log_error(f"Error in render_matches_list: {e}", e)
        st.error("An unexpected error occurred while rendering the matches list.")


def render_batch_predictions(
    matches_df: pd.DataFrame,
    get_batch_predictions_fn: Callable[[List[str], Dict[str, bool]], Dict[str, Any]],
    context_toggles: Dict[str, bool],
) -> None:
    """
    Render batch predictions for multiple matches with an enhanced UI.

    Args:
        matches_df: DataFrame containing match information.
        get_batch_predictions_fn: Function to get predictions for multiple matches.
        context_toggles: Dictionary of enabled/disabled context features.
    """
    try:
        if matches_df.empty:
            st.info("No matches available for batch prediction.")
            return

        select_tab, results_tab = st.tabs(["📋 Select Matches", "📊 Prediction Results"])

        with select_tab:
            st.header("Select Matches for Batch Analysis")
            st.markdown("Choose matches to analyze. Use the quick selectors or pick them individually from the list below.")

            if "selected_match_ids_batch" not in st.session_state:
                st.session_state.selected_match_ids_batch = []

            # Quick selection buttons
            col1, col2, col3 = st.columns(3)
            top_leagues = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
            
            with col1:
                if st.button("🔥 Select Top 5 Matches", help="Select the first 5 matches from top leagues"):
                    top_matches = matches_df[matches_df["competition"].isin(top_leagues)].head(5)
                    st.session_state.selected_match_ids_batch = top_matches["id"].astype(str).tolist()
            
            with col2:
                if st.button("⚽ All Premier League", help="Select all available Premier League matches"):
                    pl_matches = matches_df[matches_df["competition"] == "Premier League"]
                    st.session_state.selected_match_ids_batch = pl_matches["id"].astype(str).tolist()

            with col3:
                if st.button("🔄 Clear Selection"):
                    st.session_state.selected_match_ids_batch = []
            
            st.markdown("---")

            # Match selection list
            all_leagues = matches_df["competition"].unique().tolist()
            prioritized_leagues = [lg for lg in top_leagues if lg in all_leagues] + [lg for lg in all_leagues if lg not in top_leagues]

            for league in prioritized_leagues:
                league_matches = matches_df[matches_df["competition"] == league]
                with st.expander(f"{league} ({len(league_matches)} matches)", expanded=league in top_leagues):
                    for _, match in league_matches.iterrows():
                        match_id = str(match["id"])
                        home_team = match["home_team"]
                        away_team = match["away_team"]
                        
                        is_selected = st.checkbox(
                            f"{home_team} vs {away_team}", 
                            key=f"batch_check_{match_id}", 
                            value=match_id in st.session_state.selected_match_ids_batch
                        )
                        if is_selected and match_id not in st.session_state.selected_match_ids_batch:
                            st.session_state.selected_match_ids_batch.append(match_id)
                        elif not is_selected and match_id in st.session_state.selected_match_ids_batch:
                            st.session_state.selected_match_ids_batch.remove(match_id)

            # Batch prediction execution
            selected_count = len(st.session_state.selected_match_ids_batch)
            if selected_count > 0:
                st.success(f"{selected_count} matches selected for analysis.")
                if st.button("🔮 Generate Batch Predictions", type="primary", use_container_width=True):
                    with st.spinner("Analyzing matches and generating predictions..."):
                        batch_results = get_batch_predictions_fn(st.session_state.selected_match_ids_batch, context_toggles)
                    
                    if not batch_results or "predictions" not in batch_results:
                        st.error("Failed to generate batch predictions. The model may be offline or an error occurred.")
                        return
                    
                    st.session_state.batch_results = batch_results
                    st.session_state.batch_matches_df = matches_df
                    st.success("✅ Batch predictions complete! View them in the 'Prediction Results' tab.")
            else:
                st.info("Select one or more matches to generate batch predictions.")

        with results_tab:
            if "batch_results" not in st.session_state:
                st.info("Generate predictions in the 'Select Matches' tab to see the results here.")
                return

            batch_results = st.session_state.batch_results
            matches_df = st.session_state.batch_matches_df
            
            st.header("Batch Prediction Results")
            st.markdown(f"Analyzed **{len(batch_results.get('predictions', {}))}** matches in **{batch_results.get('processing_time', 0):.2f}** seconds.")

            results_data = []
            for match_id, pred in batch_results.get("predictions", {}).items():
                match_info = matches_df[matches_df["id"].astype(str) == match_id].iloc[0]
                results_data.append({
                    "Competition": match_info["competition"],
                    "Date": pd.to_datetime(match_info["match_date"]).strftime('%d %b, %H:%M'),
                    "Match": f"{match_info['home_team']} vs {match_info['away_team']}",
                    "Home Win %": pred.get("home_win", 0) * 100,
                    "Draw %": pred.get("draw", 0) * 100,
                    "Away Win %": pred.get("away_win", 0) * 100,
                })

            if not results_data:
                st.warning("No prediction data to display.")
                return

            results_df = pd.DataFrame(results_data)
            
            # Download button
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download as CSV",
                csv,
                f"goaldiggers_batch_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )

            # Display results in themed cards
            for _, row in results_df.iterrows():
                card_content = f"""
                <div style="display: flex; justify-content: space-around; text-align: center;">
                    <div>
                        <div style="font-size: 1.1em; font-weight: 600; color: #1e88e5;">{row['Home Win %']:.1f}%</div>
                        <div style="font-size: 0.9em;">Home Win</div>
                    </div>
                    <div>
                        <div style="font-size: 1.1em; font-weight: 600; color: #f57c00;">{row['Draw %']:.1f}%</div>
                        <div style="font-size: 0.9em;">Draw</div>
                    </div>
                    <div>
                        <div style="font-size: 1.1em; font-weight: 600; color: #43a047;">{row['Away Win %']:.1f}%</div>
                        <div style="font-size: 0.9em;">Away Win</div>
                    </div>
                </div>
                """
                # Use unified cards for batch results where possible
                try:
                    if uds:
                        def _result_card():
                            st.markdown(card_content, unsafe_allow_html=True)
                        uds.create_unified_card(_result_card)
                    else:
                        create_themed_card(
                            title=f"**{row['Match']}**",
                            content=card_content,
                            sub_header=f"{row['Competition']} - {row['Date']}"
                        )
                except Exception:
                    create_themed_card(
                        title=f"**{row['Match']}**",
                        content=card_content,
                        sub_header=f"{row['Competition']} - {row['Date']}"
                    )

            # Display aggregate feature importance if available
            if "aggregate_importance" in batch_results:
                st.subheader("Aggregate Feature Importance")
                st.markdown("Key factors influencing predictions across the selected matches.")
                
                importance_df = pd.DataFrame(
                    batch_results["aggregate_importance"].items(),
                    columns=["Feature", "Importance"]
                ).sort_values("Importance", ascending=False).head(10)

                st.bar_chart(importance_df.set_index("Feature"))

    except Exception as e:
        log_error(f"Error in render_batch_predictions: {e}", e)
        st.error("An unexpected error occurred while rendering batch predictions.")
