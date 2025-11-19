import logging
from datetime import datetime

import pandas as pd
import yaml

# Configure logging when module is run directly
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
logger = logging.getLogger(__name__)


def harmonize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names across different data sources.
    
    Args:
        df: DataFrame to harmonize
        
    Returns:
        DataFrame with standardized column names
    """
    logger.info(f"Harmonizing column names for DataFrame with {df.shape[0]} rows")
    
    # Standard column mapping
    column_mapping = {
        # Team names
        'home_team_name': 'home_team',
        'away_team_name': 'away_team',
        'team_name': 'team',
        
        # Scores
        'home_goals': 'home_score',
        'away_goals': 'away_score',
        'goals_home': 'home_score',
        'goals_away': 'away_score',
        
        # Date/time
        'date': 'match_date',
        'datetime': 'match_date',
        'kick_off': 'match_date',
        
        # IDs
        'fixture_id': 'match_id',
        'game_id': 'match_id',
        'id': 'match_id',
        
        # League/competition
        'competition': 'league',
        'league_name': 'league',
        'division': 'league',
        
        # Performance metrics
        'xg_home': 'home_xg',
        'xg_away': 'away_xg',
        'expected_goals_home': 'home_xg',
        'expected_goals_away': 'away_xg',
    }
    
    # Apply column mapping
    df = df.rename(columns=column_mapping)
    
    # Ensure critical columns exist
    required_columns = ['match_id', 'home_team', 'away_team', 'match_date']
    for col in required_columns:
        if col not in df.columns:
            if col == 'match_id' and 'id' not in df.columns:
                # Generate match IDs if missing
                df['match_id'] = df.index.astype(str)
                logger.info("Generated match_id column from index")
            elif col == 'match_date' and 'date' not in df.columns:
                # Use current date if no date column exists
                df['match_date'] = datetime.now().strftime('%Y-%m-%d')
                logger.warning(f"No date column found, using current date for {col}")
    
    logger.info(f"Column harmonization completed. Final shape: {df.shape}")
    return df


def preprocess(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # basic clean
    df = df.dropna(subset=["team_perf", "opp_perf"])
    # domain transforms moved into Dataset
    return df


def harmonize_valid_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize columns from valid_matches.csv to match pipeline expectations.
    - feature1 -> team_perf
    - feature2 -> opp_perf
    - result -> target (int: 0=home win, 1=draw, 2=away win)
    - Adds manager_change_date if missing (blank)
    - Adds team_id/away_team_id if missing (using sequential IDs)
    """
    logger.info(f"Harmonizing valid_matches.csv data with {df.shape[0]} rows")

    # First, rename the basic columns needed by the model
    col_map = {
        'feature1': 'team_perf',
        'feature2': 'opp_perf',
        'result': 'target',
    }
    df = df.rename(columns=col_map)

    # Ensure required columns exist
    for col in ['team_perf', 'opp_perf', 'target']:
        if col not in df.columns:
            raise ValueError(f"Missing required column after harmonization: {col}")

    # Handle team IDs if needed
    if 'home_team_id' not in df.columns:
        # Create unique team ids from team names
        all_teams = set(df['home_team'].tolist() + df['away_team'].tolist())
        team_id_map = {team: f"team_{i+100}" for i, team in enumerate(all_teams)}
        df['home_team_id'] = df['home_team'].map(team_id_map)
        df['away_team_id'] = df['away_team'].map(team_id_map)
        logger.info(f"Created {len(team_id_map)} team IDs from team names")

    # Add manager_change_date if missing
    if 'manager_change_date' not in df.columns:
        # Use a date 6 months before match_date as a placeholder
        try:
            df['match_date'] = pd.to_datetime(df['match_date'])
            df['manager_change_date'] = (df['match_date'] - pd.Timedelta(days=180)).dt.strftime('%Y-%m-%d')
        except Exception as e:
            logger.warning(f"Could not parse match_date: {e}")
            df['manager_change_date'] = ''

    # Calculate team column if needed for the FootballDataset class
    if 'team' not in df.columns:
        df['team'] = df['home_team']  # Use home_team as the team column

    # Ensure numeric columns are correctly typed
    numeric_columns = ['team_perf', 'opp_perf', 'target', 'feature3', 'feature4', 'feature5', 
                       'home_score', 'away_score']
    
    for col in numeric_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
                logger.info(f"Converted column {col} to numeric type")
            except Exception as e:
                logger.warning(f"Could not convert column {col} to numeric: {e}")

    # Only keep relevant columns for training
    keep_cols = ['match_id', 'league', 'home_team', 'away_team', 'team',
                 'home_team_id', 'away_team_id', 'match_date',
                 'home_score', 'away_score', 'team_perf', 'opp_perf', 'target', 'manager_change_date']

    # Keep extra features if present
    for extra in ['feature3', 'feature4', 'feature5']:
        if extra in df.columns:
            keep_cols.append(extra)

    # Keep only the columns we need
    result_df = df[keep_cols]
    logger.info(f"Harmonized data shape: {result_df.shape}")
    return result_df
