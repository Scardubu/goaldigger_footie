# utils/ai_insights.py

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# --- AI Provider Imports ---
# Gemini
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# DeepSeek (via OpenRouter)
try:
    from deepseek import DeepSeekClient
    from deepseek import Message as DeepSeekMessage
except ImportError:
    logging.warning("Could not import DeepSeekClient. Ensure deepseek.py is accessible.")
    DeepSeekClient = None
    DeepSeekMessage = None

# OpenAI
openai_available = True
try:
    # Prefer the modern SDK if available
    from openai import OpenAI  # type: ignore
    _openai_client_type = "v1"
except Exception:
    try:
        # Fall back to legacy SDK
        import openai  # type: ignore
        OpenAI = None  # type: ignore
        _openai_client_type = "legacy"
    except Exception:
        openai_available = False
        OpenAI = None  # type: ignore
        openai = None  # type: ignore

# Local imports
from dashboard.error_log import log_error
from utils.config import Config
from utils.prediction_handler import PredictionHandler

logger = logging.getLogger(__name__)

class MatchAnalyzer:
    def __init__(self, gemini_api_key: Optional[str] = None, openrouter_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        """
        Initializes the MatchAnalyzer using available AI providers (Gemini, DeepSeek, OpenAI).
        Args:
            gemini_api_key (Optional[str]): Gemini API key.
            openrouter_api_key (Optional[str]): OpenRouter API key for DeepSeek.
            openai_api_key (Optional[str]): OpenAI API key.
        """
        # Initialize prediction handler for XGBoost integration
        self.prediction_handler = PredictionHandler()
        # --- Gemini Setup ---
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_model = None
        self.gemini_model_name = "N/A"
        if not self.gemini_api_key:
            logger.warning(
                "Gemini API key not provided or found in GEMINI_API_KEY environment variable. Gemini analysis disabled."
            )
        else:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
                self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
                logger.info(f"Using Gemini model: {self.gemini_model_name}")
            except Exception as e:
                log_error("Failed to configure Gemini client or model", e) # Use log_error
                self.gemini_model = None

        # --- DeepSeek (OpenRouter) Setup ---
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.deepseek_client = None
        # Use a potentially correct OpenRouter ID format as default
        self.deepseek_model_name = os.getenv(
            "DEEPSEEK_MODEL_NAME", "deepseek/deepseek-coder-33b-instruct"
        )
        if not DeepSeekClient:
             logger.warning("DeepSeekClient class not available. DeepSeek analysis disabled.")
        elif not self.openrouter_api_key:
            logger.warning(
                "OpenRouter API key not provided or found in OPENROUTER_API_KEY environment variable. DeepSeek analysis disabled."
            )
        else:
            try:
                self.deepseek_client = DeepSeekClient(api_key=self.openrouter_api_key)
                # Ensure the default model ID is consistent here too
                self.deepseek_model_name = os.getenv(
                    "DEEPSEEK_MODEL_NAME", "deepseek/deepseek-coder-33b-instruct" # Corrected default model ID
                )
                logger.info(f"Using DeepSeek model via OpenRouter: {self.deepseek_model_name}")
            except Exception as e:
                log_error("Failed to configure DeepSeek client", e) # Use log_error
                self.deepseek_client = None

        # --- OpenAI Setup ---
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        self.openai_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        if self.openai_api_key and openai_available:
            try:
                if _openai_client_type == "v1" and OpenAI is not None:
                    # New SDK style
                    self.openai_client = OpenAI(api_key=self.openai_api_key)
                elif _openai_client_type == "legacy":
                    # Legacy SDK style
                    openai.api_key = self.openai_api_key  # type: ignore
                    self.openai_client = "legacy"
                logger.info(f"OpenAI client configured (mode={_openai_client_type}).")
            except Exception as e:
                log_error("Failed to configure OpenAI client", e)
                self.openai_client = None
        elif not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not set; OpenAI analysis disabled.")
        else:
            logger.warning("OpenAI SDK not available; install 'openai' to enable.")

        # Load config for provider/model/aspects
        self.config = Config.get("dashboard.ai_analysis", {})
        self.provider = self.config.get("provider", "gemini")
        self.model = self.config.get("model", "gpt-4")
        self.analysis_aspects = self.config.get("analysis_aspects", [
            {"name": "Match Overview", "key": "overview"},
            {"name": "Team Form", "key": "form"},
            {"name": "Key Factors", "key": "factors"},
            {"name": "Prediction Confidence", "key": "confidence"}
        ])

    def _generate_common_prompt(self, home_team, away_team, stats, prediction):
        """Generates the common part of the prompt for analysis."""
        prompt_parts = [
            f"Provide expert betting analysis for {home_team} vs {away_team} football match.\n",
            "Use the following data to inform your analysis:\n"
        ]

        # Add stats if provided
        if stats:
            prompt_parts.append("Statistics:\n")
            for key, value in stats.items():
                if key == 'odds' and isinstance(value, dict):
                    prompt_parts.append(f"- Betting Odds: Home Win {value.get('home_win', '?')}, Draw {value.get('draw', '?')}, Away Win {value.get('away_win', '?')}\n")
                else:
                    prompt_parts.append(f"- {key.replace('_', ' ').title()}: {value}\n")

        # Add model prediction if provided
        if prediction:
            prompt_parts.append("\nModel Prediction Probabilities:\n")
            prompt_parts.append(f"- Home Win: {prediction.get('home_win', 0)*100:.1f}%\n")
            prompt_parts.append(f"- Draw: {prediction.get('draw', 0)*100:.1f}%\n")
            prompt_parts.append(f"- Away Win: {prediction.get('away_win', 0)*100:.1f}%\n")
            
            # Add top features if available
            if 'explanations' in prediction and 'top_features' in prediction['explanations']:
                prompt_parts.append("\nKey Influential Factors (XGBoost Model):\n")
                for feature, importance in list(prediction['explanations']['top_features'].items())[:5]:
                    feature_name = feature.replace('_', ' ').title()
                    prompt_parts.append(f"- {feature_name}: Impact score {importance:.3f}\n")

        return "".join(prompt_parts)

    def _generate_aspect_prompt(self, aspect, home_team, away_team, stats, prediction):
        """
        Generate a prompt for a specific analysis aspect.
        """
        aspect_name = aspect.get("name", "Analysis")
        aspect_key = aspect.get("key", "aspect")
        # You can customize the prompt per aspect here
        base_context = self._generate_common_prompt(home_team, away_team, stats, prediction)
        return f"""
        FOCUS: {aspect_name}
        {base_context}
        Please provide a concise analysis focusing on: {aspect_name}.
        """

    def analyze_match_aspects(self, home_team, away_team, stats, prediction, provider=None, model=None, aspects=None):
        """
        Generate aspect-based AI analysis for a match using the configured provider/model.
        Returns a dict of {aspect_key: analysis_text}.
        """
        provider = provider or self.provider
        model = model or self.model
        aspects = aspects or self.analysis_aspects
        results = {}
        for aspect in aspects:
            prompt = self._generate_aspect_prompt(aspect, home_team, away_team, stats, prediction)
            # Use the correct provider
            if provider == "gemini":
                if not self.gemini_model:
                    results[aspect["key"]] = "Gemini not initialized."
                    continue
                try:
                    response = self.gemini_model.generate_content(prompt)
                    if response.parts:
                        analysis = response.text
                    elif response.candidates and response.candidates[0].content.parts:
                        analysis = response.candidates[0].content.parts[0].text
                    else:
                        analysis = "Gemini response blocked or empty."
                    results[aspect["key"]] = analysis.strip()
                except Exception as e:
                    log_error(f"Gemini error for aspect {aspect['key']}", e)
                    results[aspect["key"]] = f"Error: {e}"
            elif provider == "deepseek":
                if not self.deepseek_client or not DeepSeekMessage:
                    results[aspect["key"]] = "DeepSeek not initialized."
                    continue
                try:
                    messages = [DeepSeekMessage(role="user", content=prompt)]
                    response = self.deepseek_client.chat_completion(messages, model=self.deepseek_model_name)
                    if response and "choices" in response and response["choices"]:
                        analysis = response["choices"][0]["message"]["content"]
                        results[aspect["key"]] = analysis.strip()
                    else:
                        results[aspect["key"]] = "DeepSeek response invalid or empty."
                except Exception as e:
                    log_error(f"DeepSeek error for aspect {aspect['key']}", e)
                    results[aspect["key"]] = f"Error: {e}"
            elif provider == "openai":
                if not self.openai_client:
                    results[aspect["key"]] = "OpenAI not initialized."
                    continue
                try:
                    model_name = self.model or self.openai_model_name
                    # Allow the user to set a preview model like 'gpt-5-preview'
                    if _openai_client_type == "v1" and OpenAI is not None:
                        resp = self.openai_client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                        )
                        if resp and resp.choices:
                            analysis = resp.choices[0].message.content or ""
                            results[aspect["key"]] = analysis.strip()
                        else:
                            results[aspect["key"]] = "OpenAI response empty."
                    else:
                        # Legacy SDK
                        resp = openai.ChatCompletion.create(  # type: ignore
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                        )
                        text = resp["choices"][0]["message"]["content"] if resp and resp.get("choices") else ""
                        results[aspect["key"]] = (text or "").strip() or "OpenAI response empty."
                except Exception as e:
                    log_error(f"OpenAI error for aspect {aspect['key']}", e)
                    results[aspect["key"]] = f"Error: {e}"
            else:
                results[aspect["key"]] = f"Unsupported provider: {provider}"
        return results

    def generate_match_analysis(
        self,
        home_team: str,
        away_team: str,
        stats: Dict[str, Any],
        prediction: Optional[Dict[str, float]] = None,
        provider: str = "gemini",
        use_xgboost: bool = True,
        match_features: Optional[Dict[str, Any]] = None
    ):
        """
        Generates a betting analysis using the specified AI provider with XGBoost prediction integration.

        Args:
            home_team (str): Name of the home team.
            away_team (str): Name of the away team.
            stats (Dict[str, Any]): Dictionary of match statistics and form data.
            prediction (Optional[Dict[str, float]]): Dictionary with prediction probabilities.
            provider (str): AI provider to use ('gemini' or 'deepseek').
            use_xgboost (bool): Whether to use XGBoost for predictions.
            match_features (Optional[Dict[str, Any]]): Features for XGBoost prediction.

        Returns:
            str: Generated analysis text.
        """
        start_time = time.time()
        
        # If use_xgboost is True and we have features but no prediction, get XGBoost prediction
        if use_xgboost and match_features and not prediction:
            try:
                import pandas as pd

                # Convert features dict to DataFrame
                if isinstance(match_features, dict):
                    features_df = pd.DataFrame([match_features])
                else:
                    features_df = pd.DataFrame(match_features)
                
                # Get prediction from handler
                match_id = f"{home_team}_{away_team}".replace(" ", "_").lower()
                xgb_prediction = self.prediction_handler.get_match_prediction(features_df, match_id)
                
                # Use XGBoost prediction if available
                if xgb_prediction and xgb_prediction['status'] == 'success':
                    prediction = xgb_prediction
                    logger.info(f"Using XGBoost prediction for {home_team} vs {away_team}")
            except Exception as e:
                log_error("Failed to get XGBoost prediction", e)
                # Continue with provided prediction or None
        
        prompt = self._generate_common_prompt(home_team, away_team, stats, prediction)
        provider = provider.lower() # Normalize provider name
        
        if provider == "gemini":
            if not self.gemini_model:
                return "Gemini client/model not initialized. Cannot generate analysis."
            logger.debug(f"Generating analysis via Gemini ({self.gemini_model_name}) for {home_team} vs {away_team}")
            try:
                response = self.gemini_model.generate_content(prompt)
                logger.info(f"Gemini raw response object: type={type(response)}, value={str(response)[:200]}")
                
                # Process response object
                _potential_analysis = None

                if response.parts:
                    logger.info(f"Accessing response.text. Type of response.text: {type(response.text)}")
                    _potential_analysis = response.text
                    logger.info(f"Value from response.text (first 100): {str(_potential_analysis)[:100]}")
                elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    _part_text_source = response.candidates[0].content.parts[0].text
                    logger.info(f"Accessing candidate part text. Type of part text: {type(_part_text_source)}")
                    _potential_analysis = _part_text_source
                    logger.info(f"Value from candidate part text (first 100): {str(_potential_analysis)[:100]}")
                else:
                    safety_ratings = response.prompt_feedback.safety_ratings if response.prompt_feedback else "N/A"
                    logger.warning(f"Gemini response blocked or empty. Safety Ratings: {safety_ratings}. Full response: {str(response)[:500]}")
                    _potential_analysis = "Error: Gemini analysis generation failed (blocked or empty response)."
                
                analysis = _potential_analysis

                logger.info(f"Final type of 'analysis' before .strip(): {type(analysis)}")
                logger.info(f"Final value of 'analysis' (first 100) before .strip(): {str(analysis)[:100]}")
                
                if asyncio.iscoroutine(analysis):
                    logger.error("CRITICAL: 'analysis' is a coroutine right before .strip() in a sync function. This should not happen. Value: %s", analysis)
                    return "Error: Internal processing error (async type mismatch)."

                return analysis.strip()
            except google_exceptions.ResourceExhausted as e:
                 logger.error(f"Gemini API quota error: {e}")
                 return f"Error: Gemini API quota issue ({e})."
            except google_exceptions.GoogleAPIError as e:
                logger.error(f"Gemini API error: {e}")
                return f"Error: Gemini API issue ({e})."
            except Exception as e:
                log_error("Unexpected Gemini error", e) # Use log_error
                return f"Error: Unexpected Gemini issue ({e})."

        elif provider == "deepseek":
            if not self.deepseek_client or not DeepSeekMessage:
                return "DeepSeek client not initialized or class not available. Cannot generate analysis."
            logger.debug(f"Generating analysis via DeepSeek ({self.deepseek_model_name}) for {home_team} vs {away_team}")
            try:
                messages = [DeepSeekMessage(role="user", content=prompt)]
                response = self.deepseek_client.chat_completion(
                    messages,
                    model=self.deepseek_model_name # Use configured model
                )
                if response and "choices" in response and response["choices"]:
                    analysis = response["choices"][0]["message"]["content"]
                    logger.info(f"Successfully generated analysis via DeepSeek for {home_team} vs {away_team}")
                    return analysis.strip()
                else:
                    logger.warning(f"DeepSeek response invalid or empty: {response}")
                    return "Error: DeepSeek analysis generation failed (invalid or empty response)."
            except Exception as e:
                log_error("Unexpected DeepSeek error", e) # Use log_error
                return f"Error: Unexpected DeepSeek issue ({e})."

        elif provider == "openai":
            if not self.openai_client:
                return "OpenAI client not initialized. Cannot generate analysis."
            try:
                model_name = self.model or self.openai_model_name
                # Support preview model names such as 'gpt-5-preview'
                if _openai_client_type == "v1" and OpenAI is not None:
                    resp = self.openai_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                    )
                    if resp and resp.choices:
                        analysis = resp.choices[0].message.content or ""
                        return analysis.strip() or "OpenAI returned empty content."
                    return "OpenAI response empty."
                else:
                    resp = openai.ChatCompletion.create(  # type: ignore
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                    )
                    text = resp["choices"][0]["message"]["content"] if resp and resp.get("choices") else ""
                    return (text or "").strip() or "OpenAI returned empty content."
            except Exception as e:
                log_error("Unexpected OpenAI error", e)
                return f"Error: Unexpected OpenAI issue ({e})."
        else:
            logger.error(f"Unsupported AI provider specified: {provider}")
            return f"Error: Unsupported AI provider '{provider}'. Choose 'gemini', 'deepseek', or 'openai'."
        
        # Log performance
        analysis_time = time.time() - start_time
        logger.debug(f"Analysis generated in {analysis_time:.2f}s for {home_team} vs {away_team}")


def create_enhanced_analysis(home_team: str, away_team: str, match_data: Dict[str, Any], 
                            match_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create enhanced match analysis with XGBoost predictions and AI insights.
    
    Args:
        home_team: Name of the home team
        away_team: Name of the away team
        match_data: Match statistics and data
        match_features: Features for XGBoost prediction
        
    Returns:
        Dict with analysis components
    """
    start_time = time.time()
    
    try:
        # Initialize components
        analyzer = MatchAnalyzer()
        
        # Extract stats from match data
        stats = {
            'home_form': match_data.get('home_form', 'Unknown'),
            'away_form': match_data.get('away_form', 'Unknown'),
            'head_to_head': match_data.get('h2h', 'No data'),
            'odds': match_data.get('odds', {})
        }
        
        # Add additional stats if available
        for key in ['recent_performance', 'weather', 'injuries', 'league_position']:
            if key in match_data:
                stats[key] = match_data[key]
        
        # Generate analysis with XGBoost integration
        analysis = analyzer.generate_match_analysis(
            home_team=home_team,
            away_team=away_team,
            stats=stats,
            use_xgboost=True,
            match_features=match_features
        )
        
        # Format prediction for display
        prediction_display = None
        if hasattr(analyzer, 'prediction_handler'):
            # Get the match ID
            match_id = f"{home_team}_{away_team}".replace(" ", "_").lower()
            
            # Check if we have a prediction in the cache
            if match_id in analyzer.prediction_handler.cache:
                prediction = analyzer.prediction_handler.cache[match_id]
                prediction_display = analyzer.prediction_handler.format_prediction_for_display(prediction)
        
        # Measure performance
        processing_time = time.time() - start_time
        
        return {
            'analysis_text': analysis,
            'prediction': prediction_display,
            'processing_time': processing_time,
            'success': True
        }
        
    except Exception as e:
        log_error(f"Error creating enhanced analysis for {home_team} vs {away_team}", e)
        return {
            'analysis_text': f"Unable to generate analysis due to an error: {str(e)}",
            'prediction': None,
            'processing_time': time.time() - start_time,
            'success': False,
            'error': str(e)
        }


# Example Usage (Illustrative)
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file for API keys
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            from utils.logging_config import configure_logging  # type: ignore
            configure_logging()
        except Exception:
            logging.basicConfig(level=logging.INFO)
    
    analyzer = MatchAnalyzer()
    
    # Test basic analysis
    mock_stats = {
        'home_form': 'WWDLW', 
        'away_form': 'DWLWD', 
        'head_to_head': 'H:2, D:1, A:2',
        'odds': {'home_win': 2.0, 'draw': 3.5, 'away_win': 4.0}
    }
    
    # Mock XGBoost features
    mock_features = {
        'home_goals_avg': 1.8,
        'away_goals_avg': 1.2,
        'home_defense_rating': 75,
        'away_defense_rating': 65,
        'home_recent_form': 0.7,
        'away_recent_form': 0.5
    }
    
    # Test enhanced analysis function
    enhanced = create_enhanced_analysis(
        home_team="Liverpool",
        away_team="Manchester City",
        match_data=mock_stats,
        match_features=mock_features
    )
    
    print("\n--- Enhanced Analysis Result ---")
    print(f"Success: {enhanced['success']}")
    print(f"Processing Time: {enhanced['processing_time']:.2f}s")
    
    if enhanced['prediction']:
        print("\n--- Prediction ---")
        print(f"Home Win: {enhanced['prediction']['home_win_pct']}%")
        print(f"Draw: {enhanced['prediction']['draw_pct']}%")
        print(f"Away Win: {enhanced['prediction']['away_win_pct']}%")
        
    print("\n--- Analysis Text ---")
    print(enhanced['analysis_text'])
