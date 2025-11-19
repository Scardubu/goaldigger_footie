import streamlit as st


def render_predictions_page():
    st.title("Match Predictions")
    st.write("Select a league and match to see AI-powered predictions and betting insights.")

    # Data Source Status Widget
    st.sidebar.subheader("Data Source Status")
    import os

    import pandas as pd
    status = []
    # Check for key data files
    for fname, label in [
        ("understat_matches_epl.csv", "Understat Matches (EPL)"),
        ("wikipedia_fixtures_epl.csv", "Wikipedia Fixtures (EPL)"),
        ("fbref_matches_epl.csv", "FBref Matches (EPL)"),
        ("transfermarkt_teams_epl.csv", "Transfermarkt Teams (EPL)")
    ]:
        if os.path.exists(fname):
            try:
                df = pd.read_csv(fname, nrows=1)
                if not df.empty:
                    status.append(f"✅ {label}: Data available")
                else:
                    status.append(f"⚠️ {label}: File empty")
            except Exception:
                status.append(f"⚠️ {label}: Error reading file")
        else:
            status.append(f"❌ {label}: Not found")
    st.sidebar.markdown("\n".join(status))

    # Fallback for missing predictions data
    db_path = os.path.join('data', 'football.db')
        if os.path.exists(db_path):
        import sqlite3
        conn = sqlite3.connect(db_path)
        try:
            # Get available leagues by joining to the leagues table (normalized schema)
            try:
                leagues_df = pd.read_sql(
                    'SELECT DISTINCT l.name as league_name FROM leagues l JOIN matches m ON m.league_id = l.id', conn
                )
                leagues = leagues_df['league_name'].dropna().unique().tolist()
            except Exception:
                # Fallback to older schema field if present
                leagues_df = pd.read_sql('SELECT DISTINCT league FROM matches', conn)
                leagues = leagues_df['league'].dropna().unique().tolist()

            if leagues:
                selected_league = st.selectbox("Select a League", leagues)

                # Use JOINs to retrieve team names using normalized league mapping
                query = (
                    "SELECT m.id as match_id, ht.name as home_team, at.name as away_team "
                    "FROM matches m "
                    "LEFT JOIN teams ht ON m.home_team_id = ht.id "
                    "LEFT JOIN teams at ON m.away_team_id = at.id "
                    "LEFT JOIN leagues l ON m.league_id = l.id "
                    "WHERE l.name = ? "
                    "ORDER BY m.match_date"
                )
                matches_df = pd.read_sql(query, conn, params=(selected_league,))

                if not matches_df.empty:
                    matches = [f"{row['home_team']} vs {row['away_team']}" for index, row in matches_df.iterrows()]
                    selected_match = st.selectbox("Select a Match", matches)

                    if selected_match:
                        home_team, away_team = selected_match.split(' vs ')
                        # Placeholder for prediction display
                        st.subheader(f"Prediction for {selected_match}")
                        st.info("Betting insights and predictions will be shown here.")
                else:
                    st.info("No matches found for the selected league.")
            else:
                st.info("No leagues found in the database. Please run the data ingestion pipeline.")

        except Exception as e:
            st.warning(f"Could not load data from database: {e}")
        finally:
            conn.close()
    else:
        st.warning("Database not found. Please run the ingestion pipeline.")
