def render_advanced_metrics_section(match_data: Dict[str, Any]) -> None:
    """Render advanced metrics (xG, possession, form, key player stats) for the selected match."""
    st.markdown("#### 🧠 Advanced Match Metrics")
    features = match_data.get('features', {})
    stats = match_data.get('stats', {})
    prediction = match_data.get('prediction', {})
    # xG
    xg_home = prediction.get('expected_goals_home') or features.get('home_match_xg') or 'N/A'
    xg_away = prediction.get('expected_goals_away') or features.get('away_match_xg') or 'N/A'
    # Possession
    possession_home = stats.get('possession', {}).get('home', 'N/A')
    possession_away = stats.get('possession', {}).get('away', 'N/A')
    # Form
    form_home = stats.get('home_form', None)
    form_away = stats.get('away_form', None)
    # Key player (placeholder: extend with real data if available)
    key_player_home = features.get('key_player_home', 'Top Home Player')
    key_player_away = features.get('key_player_away', 'Top Away Player')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("xG (Exp. Goals)", f"{xg_home} - {xg_away}", help="Expected goals for home and away teams")
    col2.metric("Possession", f"{possession_home}% - {possession_away}%", help="Ball possession %")
    if form_home and form_away:
        col3.metric("Form (Last 5)", f"{' '.join(form_home)} / {' '.join(form_away)}", help="Recent form: W=Win, D=Draw, L=Loss")
    else:
        col3.metric("Form (Last 5)", "N/A", help="Recent form: W=Win, D=Draw, L=Loss")
    col4.metric("Key Players", f"{key_player_home} / {key_player_away}", help="Most impactful player for each team")

    # Visual form trend
    if form_home and form_away:
        st.markdown("##### 📈 Form Trend")
        form_map = {'W': '🟩', 'D': '🟨', 'L': '🟥'}
        st.write(f"**Home:** {' '.join([form_map.get(x, x) for x in form_home])}")
        st.write(f"**Away:** {' '.join([form_map.get(x, x) for x in form_away])}")

    st.caption("Advanced metrics are powered by the GoalDiggers data pipeline. Extend with player heatmaps, passing networks, and more for future releases.")
#!/usr/bin/env python3
"""
Enhanced Betting Insights Component
Focused on delivering clear, actionable betting recommendations with visual appeal.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class EnhancedBettingInsights:
    """
    Enhanced betting insights component with focus on clarity and actionability.
    """
    
    def __init__(self):
        """Initialize the enhanced betting insights component."""
        self.insight_categories = {
            'value_bets': 'Value Betting Opportunities',
            'form_analysis': 'Team Form Analysis',
            'head_to_head': 'Head-to-Head Statistics',
            'market_analysis': 'Market Intelligence',
            'risk_assessment': 'Risk Assessment'
        }
        
        self.risk_colors = {
            'Low': '#10b981',      # Green
            'Medium': '#f59e0b',   # Amber
            'High': '#ef4444'      # Red
        }
    
    def render_comprehensive_insights(self, match_data: Dict[str, Any]) -> None:
        """Render comprehensive betting insights for a match."""
        try:
            self._render_insights_header(match_data)
            # Add advanced metrics as a visually distinct section
            with st.expander("Show Advanced Metrics", expanded=True):
                render_advanced_metrics_section(match_data)
            # Create tabs for different insight categories
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "💰 Value Bets",
                "📊 Form Analysis", 
                "🔄 Head-to-Head",
                "📈 Market Intel",
                "⚠️ Risk Assessment"
            ])
            with tab1:
                self._render_value_betting_insights(match_data)
            with tab2:
                self._render_form_analysis(match_data)
            with tab3:
                self._render_head_to_head_analysis(match_data)
            with tab4:
                self._render_market_intelligence(match_data)
            with tab5:
                self._render_risk_assessment(match_data)
        except Exception as e:
            logger.error(f"Error rendering comprehensive insights: {e}")
            st.error("Error loading betting insights. Please try again.")
    
    def _render_insights_header(self, match_data: Dict[str, Any]) -> None:
        """Render the insights header with match information."""
        home_team = match_data.get('home_team', 'Unknown')
        away_team = match_data.get('away_team', 'Unknown')
        league = match_data.get('league', 'Unknown League')
        match_time = match_data.get('match_time', 'TBD')
        
        # Overall confidence score
        confidence = match_data.get('confidence', 0.7) * 100
        
        header_html = f"""
        <div style="
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(30, 60, 114, 0.3);
        ">
            <div style="text-align: center;">
                <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">
                    {home_team} vs {away_team}
                </h1>
                <p style="margin: 0.5rem 0; font-size: 1.1rem; opacity: 0.9;">
                    {league} • {match_time}
                </p>
                <div style="
                    background: rgba(255, 255, 255, 0.2);
                    padding: 1rem;
                    border-radius: 12px;
                    margin-top: 1rem;
                    display: inline-block;
                ">
                    <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {confidence:.0f}%
                    </div>
                    <div style="font-size: 0.9rem; opacity: 0.8;">
                        AI Confidence Score
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
    
    def _render_value_betting_insights(self, match_data: Dict[str, Any]) -> None:
        """Render value betting opportunities and recommendations."""
        st.subheader("💰 Value Betting Analysis")
        
        # Expected value calculation
        expected_value = match_data.get('expected_value', 0.05) * 100
        is_value_bet = match_data.get('is_value_bet', False)
        
        if is_value_bet:
            st.success(f"🎉 **VALUE BET DETECTED!** Expected value: +{expected_value:.1f}%")
            
            # Betting recommendations
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Recommended Bet",
                    match_data.get('recommended_bet', 'Home Win'),
                    help="AI-recommended betting option"
                )
            
            with col2:
                stake_amount = match_data.get('recommended_stake', 50)
                st.metric(
                    "Suggested Stake",
                    f"${stake_amount}",
                    help="Recommended stake amount based on Kelly Criterion"
                )
            
            with col3:
                potential_return = stake_amount * (1 + expected_value/100)
                st.metric(
                    "Potential Return",
                    f"${potential_return:.2f}",
                    delta=f"+${potential_return - stake_amount:.2f}",
                    help="Expected return on investment"
                )
            
            # Value bet explanation
            st.info("""
            **Why this is a value bet:**
            - Our AI model predicts higher probability than bookmaker odds suggest
            - Expected value calculation shows positive long-term profitability
            - Risk-adjusted returns meet our value betting criteria
            """)
            
        else:
            st.warning("⚠️ No significant value betting opportunities detected for this match.")
            st.info("Consider waiting for better odds or exploring other matches with higher expected value.")
        
        # Odds comparison (mock data)
        self._render_odds_comparison(match_data)
    
    def _render_odds_comparison(self, match_data: Dict[str, Any]) -> None:
        """Render odds comparison across different bookmakers."""
        st.subheader("📊 Odds Comparison")
        
        # Mock odds data
        bookmakers = ['Bet365', 'Ladbrokes', 'Paddy Power', 'William Hill', 'Betfair']
        home_odds = [2.10, 2.15, 2.08, 2.12, 2.20]
        draw_odds = [3.40, 3.35, 3.45, 3.38, 3.50]
        away_odds = [3.80, 3.75, 3.85, 3.78, 3.70]
        
        odds_df = pd.DataFrame({
            'Bookmaker': bookmakers,
            'Home Win': home_odds,
            'Draw': draw_odds,
            'Away Win': away_odds
        })
        
        # Highlight best odds
        st.dataframe(
            odds_df.style.highlight_max(axis=0, subset=['Home Win', 'Draw', 'Away Win'], color='lightgreen'),
            use_container_width=True
        )
        
        # Best odds summary
        best_home = max(home_odds)
        best_draw = max(draw_odds)
        best_away = max(away_odds)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Home Odds", f"{best_home:.2f}")
        with col2:
            st.metric("Best Draw Odds", f"{best_draw:.2f}")
        with col3:
            st.metric("Best Away Odds", f"{best_away:.2f}")
    
    def _render_form_analysis(self, match_data: Dict[str, Any]) -> None:
        """Render team form analysis with visual indicators."""
        st.subheader("📊 Team Form Analysis")
        
        home_team = match_data.get('home_team', 'Home Team')
        away_team = match_data.get('away_team', 'Away Team')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {home_team}")
            home_form = match_data.get('home_form', ['W', 'W', 'D', 'L', 'W'])
            self._render_form_indicator(home_form, "home")
            
            # Form statistics
            wins = home_form.count('W')
            draws = home_form.count('D')
            losses = home_form.count('L')
            
            st.metric("Recent Form", f"{wins}W-{draws}D-{losses}L")
            self._render_form_chart(home_form, home_team)
        
        with col2:
            st.markdown(f"### {away_team}")
            away_form = match_data.get('away_form', ['L', 'W', 'D', 'W', 'L'])
            self._render_form_indicator(away_form, "away")
            
            # Form statistics
            wins = away_form.count('W')
            draws = away_form.count('D')
            losses = away_form.count('L')
            
            st.metric("Recent Form", f"{wins}W-{draws}D-{losses}L")
            self._render_form_chart(away_form, away_team)
        
        # Form comparison
        self._render_form_comparison(match_data)
    
    def _render_form_indicator(self, form_list: List[str], team_type: str) -> None:
        """Render visual form indicator."""
        form_html = "<div style='display: flex; gap: 4px; margin: 1rem 0;'>"
        
        for result in form_list[-5:]:  # Last 5 matches
            if result == 'W':
                color = '#10b981'
                symbol = 'W'
            elif result == 'D':
                color = '#f59e0b'
                symbol = 'D'
            else:  # L
                color = '#ef4444'
                symbol = 'L'
            
            form_html += f"""
            <div style="
                background: {color};
                color: white;
                width: 32px;
                height: 32px;
                border-radius: 6px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 14px;
            ">{symbol}</div>
            """
        
        form_html += "</div>"
        st.markdown(form_html, unsafe_allow_html=True)
    
    def _render_form_chart(self, form_list: List[str], team_name: str) -> None:
        """Render form trend chart."""
        # Convert form to points (W=3, D=1, L=0)
        points = []
        for result in form_list[-10:]:  # Last 10 matches
            if result == 'W':
                points.append(3)
            elif result == 'D':
                points.append(1)
            else:
                points.append(0)
        
        # Create trend chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=points,
            mode='lines+markers',
            name=team_name,
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8, color='#3b82f6')
        ))
        
        fig.update_layout(
            height=200,
            showlegend=False,
            margin=dict(l=0, r=0, t=20, b=20),
            xaxis=dict(title="Recent Matches", showgrid=False),
            yaxis=dict(title="Points", range=[0, 3], tickvals=[0, 1, 3], ticktext=['L', 'D', 'W']),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_form_comparison(self, match_data: Dict[str, Any]) -> None:
        """Render form comparison between teams."""
        st.subheader("⚖️ Form Comparison")
        
        home_form = match_data.get('home_form', ['W', 'W', 'D', 'L', 'W'])
        away_form = match_data.get('away_form', ['L', 'W', 'D', 'W', 'L'])
        
        # Calculate form points
        def calculate_form_points(form_list):
            return sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in form_list[-5:])
        
        home_points = calculate_form_points(home_form)
        away_points = calculate_form_points(away_form)
        max_points = 15  # 5 matches * 3 points
        
        col1, col2 = st.columns(2)
        
        with col1:
            home_percentage = (home_points / max_points) * 100
            st.metric(
                match_data.get('home_team', 'Home Team'),
                f"{home_points}/{max_points} pts",
                delta=f"{home_percentage:.0f}%"
            )
        
        with col2:
            away_percentage = (away_points / max_points) * 100
            st.metric(
                match_data.get('away_team', 'Away Team'),
                f"{away_points}/{max_points} pts",
                delta=f"{away_percentage:.0f}%"
            )
        
        # Form advantage indicator
        if home_points > away_points:
            advantage_team = match_data.get('home_team', 'Home Team')
            advantage_points = home_points - away_points
        elif away_points > home_points:
            advantage_team = match_data.get('away_team', 'Away Team')
            advantage_points = away_points - home_points
        else:
            advantage_team = None
            advantage_points = 0
        
        if advantage_team:
            st.info(f"📈 **Form Advantage**: {advantage_team} (+{advantage_points} pts)")
        else:
            st.info("📊 **Even Form**: Both teams have similar recent form")
    
    def _render_head_to_head_analysis(self, match_data: Dict[str, Any]) -> None:
        """Render head-to-head historical analysis."""
        st.subheader("🔄 Head-to-Head Statistics")
        
        # Mock H2H data
        h2h_data = {
            'total_matches': 20,
            'home_wins': 8,
            'draws': 5,
            'away_wins': 7,
            'avg_goals_home': 1.8,
            'avg_goals_away': 1.5,
            'last_5_results': ['Home Win', 'Draw', 'Away Win', 'Home Win', 'Home Win']
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total H2H Matches", h2h_data['total_matches'])
            
        with col2:
            home_win_rate = (h2h_data['home_wins'] / h2h_data['total_matches']) * 100
            st.metric("Home Win Rate", f"{home_win_rate:.0f}%")
            
        with col3:
            away_win_rate = (h2h_data['away_wins'] / h2h_data['total_matches']) * 100
            st.metric("Away Win Rate", f"{away_win_rate:.0f}%")
        
        # H2H Results visualization
        results_data = {
            'Result': ['Home Wins', 'Draws', 'Away Wins'],
            'Count': [h2h_data['home_wins'], h2h_data['draws'], h2h_data['away_wins']]
        }
        
        fig = px.pie(
            results_data,
            values='Count',
            names='Result',
            title="Head-to-Head Results Distribution",
            color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444']
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent encounters
        st.subheader("📋 Recent Encounters")
        for i, result in enumerate(h2h_data['last_5_results'][:5], 1):
            if result == 'Home Win':
                st.success(f"Match {i}: {match_data.get('home_team', 'Home')} Won")
            elif result == 'Away Win':
                st.error(f"Match {i}: {match_data.get('away_team', 'Away')} Won")
            else:
                st.warning(f"Match {i}: Draw")
    
    def _render_market_intelligence(self, match_data: Dict[str, Any]) -> None:
        """Render market intelligence and betting trends."""
        st.subheader("📈 Market Intelligence")
        
        # Betting trends
        st.markdown("### 💰 Public Betting Trends")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Home Win Bets",
                "45%",
                delta="+5%",
                help="Percentage of public bets on home team"
            )
        
        with col2:
            st.metric(
                "Draw Bets", 
                "20%",
                delta="-2%",
                help="Percentage of public bets on draw"
            )
        
        with col3:
            st.metric(
                "Away Win Bets",
                "35%",
                delta="-3%",
                help="Percentage of public bets on away team"
            )
        
        # Market movement
        st.markdown("### 📊 Odds Movement")
        
        # Mock odds movement data
        time_points = ['Opening', '24h ago', '12h ago', '6h ago', 'Current']
        home_odds_movement = [2.20, 2.15, 2.10, 2.08, 2.05]
        away_odds_movement = [3.50, 3.60, 3.75, 3.80, 3.85]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=home_odds_movement,
            mode='lines+markers',
            name=match_data.get('home_team', 'Home Team'),
            line=dict(color='#10b981', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=away_odds_movement,
            mode='lines+markers',
            name=match_data.get('away_team', 'Away Team'),
            line=dict(color='#ef4444', width=3)
        ))
        
        fig.update_layout(
            title="Odds Movement Trend",
            xaxis_title="Time",
            yaxis_title="Decimal Odds",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Market insights
        st.info("""
        **Market Insights:**
        - Home team odds have shortened, indicating increased backing
        - Away team odds have drifted, suggesting less confidence
        - Sharp money appears to be on the home team
        """)
    
    def _render_risk_assessment(self, match_data: Dict[str, Any]) -> None:
        """Render comprehensive risk assessment."""
        st.subheader("⚠️ Risk Assessment")
        
        risk_level = match_data.get('risk_level', 'Medium')
        risk_color = self.risk_colors.get(risk_level, '#f59e0b')
        
        # Overall risk indicator
        risk_html = f"""
        <div style="
            background: {risk_color};
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        ">
            <h3 style="margin: 0; font-size: 1.5rem;">Risk Level: {risk_level}</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                Based on volatility, form, and market analysis
            </p>
        </div>
        """
        st.markdown(risk_html, unsafe_allow_html=True)
        
        # Risk factors breakdown
        st.markdown("### 🔍 Risk Factors Analysis")
        
        risk_factors = {
            'Team Form Stability': 85,
            'H2H Predictability': 70,
            'Market Consensus': 90,
            'Injury Impact': 60,
            'Motivation Level': 80
        }
        
        for factor, score in risk_factors.items():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Color code based on score
                if score >= 80:
                    color = 'success'
                elif score >= 60:
                    color = 'warning'
                else:
                    color = 'danger'
                
                st.progress(score / 100)
                
            with col2:
                st.metric(factor, f"{score}%")
        
        # Risk mitigation recommendations
        st.markdown("### 🛡️ Risk Mitigation")
        
        if risk_level == 'High':
            st.warning("""
            **High Risk Recommendations:**
            - Consider smaller stake sizes
            - Look for value in alternative markets
            - Monitor for late team news
            - Consider avoiding this match
            """)
        elif risk_level == 'Medium':
            st.info("""
            **Medium Risk Recommendations:**
            - Use standard stake sizing
            - Consider hedging strategies
            - Monitor market movements
            - Stick to main markets
            """)
        else:
            st.success("""
            **Low Risk Recommendations:**
            - Can consider larger stakes
            - Explore multiple markets
            - High confidence betting
            - Good for accumulators
            """)

def render_betting_insights(match_data: Dict[str, Any]) -> None:
    """Main function to render betting insights for a match."""
    insights_component = EnhancedBettingInsights()
    insights_component.render_comprehensive_insights(match_data)
