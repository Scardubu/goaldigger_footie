"""
Orchestrator script to generate actionable betting insights for user-selected teams/matches.
Usage:
    python scripts/generate_betting_insights.py --teams "TeamA,TeamB" --date_from 2025-04-25 --date_to 2025-04-27
"""
import argparse
import logging
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage

import joblib  # Reintroduced for test monkeypatching of joblib.load

# import joblib # No longer loading static model directly
import numpy as np
import pandas as pd
import tenacity  # Import tenacity explicitly for error handling

from dashboard.error_log import log_error  # Import log_error
from data.api_clients.football_data_api import FootballDataAPI

# Import config and storage
from data.storage.database import DBManager  # Use DBManager

# from data.api_clients.openweather_api import OpenWeatherAPI # Not used directly here
from models.feature_eng.feature_generator import FeatureGenerator
from models.predictive.ensemble_model import (
    EnsemblePredictor,  # Import EnsemblePredictor
)
from scripts.core.ai_validator import AIDataValidator
from utils.config import Config, ConfigError  # Import centralized config
from utils.notifications import send_slack_message, send_telegram_message

# from models.predictive.model import load_model, predict  # Uncomment and adjust as needed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Generate actionable betting insights.")
    parser.add_argument('--teams', type=str, help='Comma-separated list of team names (e.g., "Chelsea,Arsenal")')
    parser.add_argument('--date_from', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--date_to', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--leagues', type=str, default=None, help='Comma-separated list of league codes (optional)')
    parser.add_argument('--model-version', type=str, default='default', help='Identifier for the model version used (optional)') # Add model version arg
    return parser.parse_args()

def send_email_report(report_path: str, recipient: str): # Add type hints
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")
    if not EMAIL_USER or not EMAIL_PASS:
        logger.warning("Email credentials not set in environment. Skipping email report.")
        return

    logger.info(f"Attempting to send email report to {recipient}...")
    msg = EmailMessage()
    msg["Subject"] = "Daily Betting Insights Report" # Slightly more specific subject
    msg["From"] = EMAIL_USER
    msg["To"] = recipient
    msg.set_content("Find attached your actionable betting insights.")
    try:
        with open(report_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="application",
                subtype="octet-stream", # Generic attachment type
                filename=os.path.basename(report_path)
            )
    except FileNotFoundError:
        logger.error(f"Email report attachment file not found: {report_path}. Sending email without attachment.")
    except Exception as attach_e:
         log_error("High-level operation failed", attach_e)
         logger.error(f"Error attaching report file {report_path} to email: {attach_e}. Sending email without attachment.") # Keep existing logic

    try:
        # Consider making SMTP server/port configurable if needed
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp: # Added timeout
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
            logger.info(f"Email report sent successfully to {recipient}.")
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP Authentication Error. Check EMAIL_USER and EMAIL_PASS.")
    except smtplib.SMTPException as smtp_e:
        logger.error(f"SMTP Error sending email report: {smtp_e}")
    except OSError as e: # Catch potential network/socket errors
        logger.error(f"Network error sending email report: {e}")
    except Exception as e:
        log_error("High-level operation failed", e)
        logger.error(f"Unexpected error sending email report: {e}") # Keep existing logic


def main():
    args = parse_args()
    team_list = [t.strip() for t in args.teams.split(",")] if args.teams else []
    date_from = args.date_from
    date_to = args.date_to
    league_list = [l.strip() for l in args.leagues.split(",")] if args.leagues else None
    model_version_arg = getattr(args, 'model_version', 'default') # Get model version from args

    # --- Configuration Loading ---
    try:
        # Load configuration using the centralized utility
        Config.load()

        # Get paths using Config.get()
        project_root = Config.get('paths.project_root', os.getenv('PROJECT_ROOT', '.')) # Fallback to env var if needed
        model_base_path_rel = Config.get('paths.models.base')
        predictor_filename = Config.get('paths.models.predictor_filename')
        reports_base_path_rel = Config.get('paths.reports')

        if not reports_base_path_rel:
            # Provide safe fallback for tests / minimal environments
            reports_base_path_rel = "reports"
            logger.warning("paths.reports missing in config; defaulting to ./reports")
        missing_required = [k for k,v in {
            "paths.project_root": project_root,
            "paths.models.base": model_base_path_rel,
            "paths.models.predictor_filename": predictor_filename,
        }.items() if not v]
        if missing_required:
            raise KeyError(f"Missing essential path(s) in config: {', '.join(missing_required)}")

        # Construct absolute paths relative to project root
        model_path = os.path.normpath(os.path.join(project_root, model_base_path_rel, predictor_filename))
        reports_dir = os.path.normpath(os.path.join(project_root, reports_base_path_rel))

        logger.info(f"Using Project Root: {project_root}")
        logger.info(f"Using Model Path: {model_path}")
        logger.info(f"Using Reports Directory: {reports_dir}")

    except KeyError as e:
        logger.critical(f"CRITICAL: Configuration Error - {e}. Please check config/config.yaml. Exiting.")
        return 1 # Indicate failure due to config error
    except Exception as e:
        log_error("High-level operation failed", e)
        logger.critical(f"CRITICAL: Unexpected error processing configuration paths: {e}. Exiting.") # Keep existing logic
        return 1 # Indicate failure due to other config processing error


    try:
        # 1. Fetch fixtures for selected teams and date range
        api_client = FootballDataAPI()
        
        # Critical section where API errors can occur
        try:
            # This is where tenacity is being used, so we need to handle the RetryError
            fixtures = api_client.get_matches_for_competitions(
                competition_codes=league_list,
                date_from=date_from,
                date_to=date_to,
                status=None
            )
        except Exception as e:
            log_error("High-level operation failed", e)
            # Make sure we catch ANY exception, including RetryError
            error_msg = f"[ALERT] Betting insights pipeline failed: {e}" # Keep existing logic
            logger.error(f"Error generating insights: {e}") # Keep existing logic
            # Send alerts for API failures # Keep existing logic
            send_slack_message(error_msg)
            send_telegram_message(error_msg)
            return 1  # Indicate failure
            
        fixtures_df = pd.DataFrame(fixtures)
        if team_list:
            fixtures_df = fixtures_df[
                fixtures_df['homeTeam'].apply(lambda x: x.get('name') in team_list) |
                fixtures_df['awayTeam'].apply(lambda x: x.get('name') in team_list)
            ]
        if fixtures_df.empty:
            logger.info("No fixtures found for the selected teams and date range.")
            return 0 # Indicate success (no error, just no data)
        logger.info(f"Found {len(fixtures_df)} fixtures for selected teams.")

        # 2. Feature engineering and validation
        db = DBManager() # Instantiate DBManager
        feature_gen = FeatureGenerator(db) # Pass DBManager instance if needed by FeatureGenerator (check its definition later if necessary)
        validator = AIDataValidator()
        features = feature_gen.generate_features_for_dataset(fixtures_df)
        # --- Robust feature alignment (enforce canonical feature list) ---
        feature_list = Config.get("models.normalization.feature_list", [])
        if feature_list:
            missing = [feat for feat in feature_list if feat not in features.columns]
            extra = [feat for feat in features.columns if feat not in feature_list and feat != 'match_id']
            if missing:
                logger.warning(f"Missing features for betting insights: {missing}. Imputing with 0.0.")
                for feat in missing:
                    features[feat] = 0.0
            if extra:
                logger.warning(f"Extra features for betting insights: {extra}. Dropping them.")
                features = features.drop(columns=extra)
            # Ensure correct order
            features = features[[col for col in ['match_id'] + feature_list if col in features.columns]]
        else:
            logger.warning("No canonical feature_list found in config. Skipping strict alignment.")

        _, validation_report = validator.validate_dataset(features)
        if features.empty:
             logger.error("Feature generation resulted in an empty DataFrame. Cannot generate insights.")
             return 1 # Indicate failure

        _, validation_report = validator.validate_dataset(features)
        logger.info(f"Validation report: {validation_report}")
        # Optionally, stop if validation fails critical checks
        # if not validation_report.get('validation_passed', True):
        #     logger.error("Data validation failed. Aborting insight generation.")
        #     return 1

        # 3. Model prediction using EnsemblePredictor
        logger.info("Initializing EnsemblePredictor...")
        try:
            ensemble_predictor = EnsemblePredictor()
            # Check if models loaded correctly within the ensemble predictor
            if ensemble_predictor.static_model is None or ensemble_predictor.dynamic_model is None:
                 raise ValueError("Ensemble predictor failed to load one or more internal models.")
            logger.info("EnsemblePredictor initialized successfully.")
        except Exception as ensemble_init_e:
            log_error("High-level operation failed", ensemble_init_e)
            logger.critical(f"CRITICAL: Failed to initialize EnsemblePredictor: {ensemble_init_e}. Cannot generate predictions.")
            return 1 # Indicate critical failure

        # Prepare lists to store prediction results
        home_probs, draw_probs, away_probs = [], [], []
        prediction_successful = True # Flag to track if any prediction fails

        # Iterate through features DataFrame row by row for prediction
        # Ensure 'match_id' is available if needed for news fetching later
        if 'match_id' not in features.columns:
             # Try to get it from index if it was set there
             if features.index.name == 'match_id':
                  features = features.reset_index()
             else:
                  # If still not found, generate a placeholder or log error
                  logger.warning("match_id column not found in features DataFrame. News fetching might be affected.")
                  # features['match_id'] = [f"unknown_{i}" for i in range(len(features))] # Placeholder if needed

        logger.info(f"Generating predictions for {len(features)} matches using EnsemblePredictor...")
        for index, row in features.iterrows():
            match_id = row.get('match_id', f"index_{index}") # Get match_id if available
            # Prepare features for the single row (ensure it's a DataFrame)
            single_features_df = pd.DataFrame([row])

            # --- Placeholder for News Text ---
            # In a real scenario, fetch relevant news/text for the match_id here
            # Example: news_text = fetch_news_for_match(match_id)
            news_text = f"Placeholder news summary for match {match_id}. Teams are preparing."
            # ---------------------------------

            try:
                # Predict using the ensemble model
                pred_probs = ensemble_predictor.predict(single_features_df, news_text)

                if pred_probs is not None and len(pred_probs) == 3:
                    home_probs.append(pred_probs[0])
                    draw_probs.append(pred_probs[1])
                    away_probs.append(pred_probs[2])
                    logger.debug(f"Prediction successful for match {match_id}")
                else:
                    logger.warning(f"Ensemble prediction failed or returned unexpected format for match {match_id}. Using stub values.")
                    home_probs.append(1/3) # Fallback stub
                    draw_probs.append(1/3)
                    away_probs.append(1/3)
                    prediction_successful = False # Mark that at least one prediction failed

            except Exception as predict_e:
                log_error("High-level operation failed", predict_e)
                logger.error(f"Error predicting match {match_id} with ensemble: {predict_e}. Using stub values.")
                home_probs.append(1/3) # Fallback stub
                draw_probs.append(1/3)
                away_probs.append(1/3)
                prediction_successful = False # Mark failure

        # Add predictions to the features DataFrame
        features['home_win_prob'] = home_probs
        features['draw_prob'] = draw_probs
        features['away_win_prob'] = away_probs

        if not prediction_successful:
             logger.warning("One or more predictions failed or used stub values.")
        else:
             logger.info("Predictions generated successfully using EnsemblePredictor.")


        # 4. Value bet calculation (fetch real odds if available)
        odds = []
        for idx, match in fixtures_df.iterrows():
            match_id = match.get('id') or match.get('match_id') # Ensure match_id is correctly extracted
            if match_id is None:
                logger.warning(f"Could not determine match_id for row {idx}. Skipping odds fetch.")
                odds.append(2.1) # Default odds if no ID
                continue

            try:
                # Use DBManager's fetchone method - it already returns a single row or None
                result = db.fetchone(
                    """
                    SELECT home_win_odds FROM odds WHERE match_id = ? ORDER BY fetched_at DESC LIMIT 1
                    """,
                    (match_id,)
                )
                # Check if result is not None and the first element is not None
                odds.append(result[0] if result and result[0] is not None else 2.1) # Use a default odd if not found or None
            except Exception as e:
                # DBManager already logs the specific error, log context here
                logger.warning(f"Error fetching odds for match {match_id} using DBManager: {e}. Using default odds.")
                odds.append(2.1) # Use default on other errors # Keep existing logic
        features['bookmaker_home_odds'] = odds
        features['value_home'] = features['home_win_prob'] * features['bookmaker_home_odds']
        # Add model version used for this prediction run
        features['model_version_used'] = model_version_arg

        # 5. Save report
        try:
            os.makedirs(reports_dir, exist_ok=True) # Use configured reports_dir
            report_filename = f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv" # Added seconds for uniqueness
            report_path = os.path.join(reports_dir, report_filename)
            features.to_csv(report_path, index=False)
            logger.info(f"Saved insights report to {report_path}")
        except OSError as report_err:
             logger.error(f"Error creating report directory or saving report {report_path}: {report_err}")
             # Continue without report if saving fails, but log error
             report_path = None # Ensure report_path is None if saving failed

        # 6. Email report (optional, set EMAIL_USER and EMAIL_PASS in .env)
        recipient = os.getenv("INSIGHTS_EMAIL_RECIPIENT")
        if recipient:
            send_email_report(report_path, recipient)

        # 7. Slack and Telegram notifications
        summary_text = f"Betting Insights generated for {len(features)} matches on {datetime.now().strftime('%Y-%m-%d')}."
        send_slack_message(summary_text, file_path=report_path)
        send_telegram_message(summary_text, file_path=report_path)

        # 8. Output actionable insights
        for idx, row in features.iterrows():
            match = fixtures_df.iloc[idx]
            home = match['homeTeam']['name']
            away = match['awayTeam']['name']
            print(f"\nMatch: {home} vs {away}")
            print(f"Recommended Bet: Home Win" if row['value_home'] > 1 else "No value bet found")
            print(f"Model Probability: {row['home_win_prob']:.2f}")
            print(f"Bookmaker Odds: {row['bookmaker_home_odds']}")
            print(f"Value: {row['value_home']:.2f}")
            print(f"Confidence: {int(row['home_win_prob']*100)}%")
            print(f"Rationale: (stub) Model favors home team based on features.")
        
        return 0 # Indicate success
    except Exception as e:
        log_error("High-level operation failed", e)
        logger.exception(f"Error in betting insights pipeline: {e}") # Keep existing logic
        # Error alerting # Keep existing logic
        send_slack_message(f"[ALERT] Betting insights pipeline failed: {e}") # Keep existing logic
        send_telegram_message(f"[ALERT] Betting insights pipeline failed: {e}")
        return 1 # Indicate failure

if __name__ == "__main__":
    main()
