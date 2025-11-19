import logging  # Import logging
import os
import re

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__) # Initialize logger for this module

# Determine the absolute path to the project's root directory and default config file
# Assumes this script (config.py) is in <project_root>/utils/
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, "config", "config.yaml")

# Dynamically set crucial environment variables if not already present
if not os.getenv('PROJECT_ROOT'):
    os.environ['PROJECT_ROOT'] = PROJECT_ROOT_DIR
    logger.info(f"Environment variable PROJECT_ROOT not set. Dynamically set to: {PROJECT_ROOT_DIR}")

if not os.getenv('DATA_DIR'):
    DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data')
    os.environ['DATA_DIR'] = DEFAULT_DATA_DIR
    logger.info(f"Environment variable DATA_DIR not set. Dynamically set to: {DEFAULT_DATA_DIR}")

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

# Regex to find ${VAR_NAME} style placeholders
ENV_VAR_PATTERN = re.compile(r'\$\{(?P<var_name>[^}]+)\}')
# List of known environment variables that represent paths
PATH_ENV_VARS = {"PROJECT_ROOT", "DATA_DIR"} # Add more if needed

def _replace_env_var(match):
    """Helper function to replace environment variable matches."""
    try:
        var_name = match.group('var_name')
        env_val = os.getenv(var_name)

        if env_val is None:
            # Keep the original placeholder if env var is not set.
            # Validation of required vars happens later in Config.load().
            print(f"Warning: Environment variable '{var_name}' not set. Keeping placeholder.")
            return match.group(0)

        # Normalize paths to forward slashes for consistency
        if var_name in PATH_ENV_VARS:
             # Ensure forward slashes, especially important on Windows
             return env_val.replace('\\', '/')
        return env_val
    except Exception as e:
        print(f"Error processing environment variable replacement: {e}")
        return match.group(0)


def _interpolate_value(value):
    """
    Recursively interpolates environment variables in strings, lists, or dicts.
    Normalizes known path variables to use forward slashes.
    """
    try:
        if isinstance(value, str):
            # Repeatedly substitute until no more placeholders are found
            # This handles nested variables like "${DATA_DIR}/raw/"
            interpolated_value = value
            max_iterations = 10  # Prevent infinite loops
            iteration = 0

            while iteration < max_iterations:
                # Use re.sub which handles multiple occurrences in one pass
                new_value = ENV_VAR_PATTERN.sub(_replace_env_var, interpolated_value)
                # Check if any substitution actually happened
                if new_value == interpolated_value:
                    break
                interpolated_value = new_value
                iteration += 1

            if iteration >= max_iterations:
                print(f"Warning: Maximum interpolation iterations reached for value: {value}")

            return interpolated_value
        elif isinstance(value, dict):
            # Recursively interpolate dictionary values
            return {k: _interpolate_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            # Recursively interpolate list items
            return [_interpolate_value(item) for item in value]
        else:
            # Return non-string/list/dict types as is (e.g., numbers, booleans)
            return value
    except Exception as e:
        print(f"Error in _interpolate_value: {e}")
        return value

# Define the path for model_params.yaml relative to PROJECT_ROOT_DIR
DEFAULT_MODEL_PARAMS_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, "config", "model_params.yaml")

from pydantic import ConfigDict


# TODO: Migrate to full Pydantic V2 usage. For now, use ConfigDict for compatibility.
class Config(ConfigDict):
    _config_data = None
    _config_path = DEFAULT_CONFIG_FILE_PATH # Use absolute path
    _model_params_path = DEFAULT_MODEL_PARAMS_FILE_PATH # Add this

    @staticmethod
    def set_config_path(path):
        """Allows setting a different config path, e.g., for tests."""
        Config._config_path = path
        Config._config_data = None # Reset loaded data if path changes

    @staticmethod
    def load(config_path=None, model_params_path=None): # Add model_params_path argument
        """
        Loads, interpolates, and validates the configuration.
        Uses the path set by set_config_path or the default.
        If already loaded and no specific path override is given, returns existing data.
        """
        # Determine the paths to use
        current_load_path = config_path if config_path is not None else Config._config_path
        current_model_params_path = model_params_path if model_params_path is not None else Config._model_params_path

        # Check if config (including merged model_params) is already loaded and paths haven't changed
        # This condition needs to be robust if we allow dynamic path setting for model_params too.
        # For now, assume if _config_data is not None, it includes model_params if they were available at last load.
        if Config._config_data is not None and \
           (config_path is None or config_path == Config._config_path) and \
           (model_params_path is None or model_params_path == Config._model_params_path):
            logger.debug("Configuration (including model_params) already loaded. Skipping file read.")
            return Config._config_data
        if Config._config_data is not None and config_path is None and Config._config_path == "config/config.yaml":
            # Be careful if set_config_path was used to change the default, then a subsequent
            # Config.load() without args should still reload from the new default.
            # This check is a bit simplistic; a more robust way would be to store the path
            # from which _config_data was loaded and compare.
            # For now, assume if _config_data exists and default path is requested, it's fine.
            # A better check: if config_path is None (meaning use default) AND _config_data is populated from that default.
            # Let's refine: if config_path is None (use default) and _config_data is already loaded, just return.
            # If config_path is provided, it implies a specific load request, so proceed.
            if config_path is None: # Request to load default config
                 if Config._config_data is not None: # And it's already loaded
                    logger.debug("Configuration already loaded. Skipping file read.")
                    return Config._config_data

        current_load_path = config_path if config_path is not None else Config._config_path

        load_dotenv()  # Populate os.environ from .env if present

        try:
            # Ensure the path is treated as relative to the script's execution context (usually project root)
            # For robustness, one might construct an absolute path if CWD issues are suspected.
            # However, standard practice is relative to CWD.
            logger.debug(f"Attempting to load configuration from: {current_load_path} (CWD: {os.getcwd()})")

            # Check if file exists and is readable
            if not os.path.exists(current_load_path):
                raise ConfigError(f"Configuration file not found: {current_load_path}")

            if not os.access(current_load_path, os.R_OK):
                raise ConfigError(f"Configuration file not readable: {current_load_path}")

            with open(current_load_path, "r", encoding='utf-8-sig') as f:
                # Verify file content is not empty
                content = f.read()
                if not content.strip():
                    raise ConfigError(f"Configuration file is empty: {current_load_path}")

                # Reset file pointer and load YAML
                f.seek(0)
                cfg_raw = yaml.safe_load(f)

                # Verify YAML loaded successfully
                if cfg_raw is None:
                    raise ConfigError(f"Configuration file contains no valid YAML data: {current_load_path}")

        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {current_load_path}")
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML file {current_load_path}: {e}")
        except Exception as e: # Catch other potential file reading errors
             raise ConfigError(f"Error reading configuration file {current_load_path}: {e}")


        if not isinstance(cfg_raw, dict):
             raise ConfigError(f"Invalid configuration format in {config_path}. Root must be a dictionary.")

        # Interpolate environment variables recursively AFTER loading
        try:
            cfg_interpolated = _interpolate_value(cfg_raw)
            if cfg_interpolated is None:
                logger.warning("Configuration interpolation returned None, using raw configuration")
                cfg_interpolated = cfg_raw
        except Exception as e:
            logger.error(f"Error during main configuration interpolation: {e}")
            logger.warning("Using raw configuration without interpolation")
            cfg_interpolated = cfg_raw

        # Load model parameters configuration
        model_params_data = {}
        try:
            logger.debug(f"Attempting to load model parameters from: {current_model_params_path}")
            if os.path.exists(current_model_params_path):
                with open(current_model_params_path, "r", encoding='utf-8') as f:
                    model_params_raw = yaml.safe_load(f)
                if model_params_raw and isinstance(model_params_raw, dict):
                    model_params_data = _interpolate_value(model_params_raw)
                    logger.info(f"Successfully loaded and interpolated model parameters from {current_model_params_path}.")
                elif model_params_raw: # Loaded but not a dict
                    logger.warning(f"Model parameters file {current_model_params_path} did not load as a dictionary. Skipping merge.")
            else:
                logger.info(f"Model parameters file not found at {current_model_params_path}. Skipping merge.")
        except yaml.YAMLError as e:
            logger.warning(f"Error parsing YAML file {current_model_params_path}: {e}. Skipping model params merge.")
        except Exception as e:
             logger.warning(f"Error reading model parameters file {current_model_params_path}: {e}. Skipping model params merge.")

        # Merge model_params_data into cfg_interpolated
        if model_params_data:
            if 'models' not in cfg_interpolated:
                cfg_interpolated['models'] = {}
            
            if 'normalization' in model_params_data and isinstance(model_params_data['normalization'], dict):
                if 'normalization' not in cfg_interpolated['models']:
                    cfg_interpolated['models']['normalization'] = {}
                cfg_interpolated['models']['normalization'].update(model_params_data['normalization'])
                logger.info("Merged 'normalization' from model_params into config['models'].")

        # Validate required env vars (check AFTER interpolation attempt)
        # Ensure 'env' and 'required' keys exist before accessing
        env_config = cfg_interpolated.get('env', {})
        required_vars = env_config.get('required', []) if isinstance(env_config, dict) else []

        missing = []
        if isinstance(required_vars, list):
            for var in required_vars:
                # Check if the variable is actually set in the environment
                if os.getenv(var) is None:
                    missing.append(var)
                # Additionally, check if the placeholder might still exist in the config
                # This indicates the env var was missing during interpolation.
                # This check is complex, maybe rely solely on os.getenv check.

        if missing:
            raise ConfigError(f"Missing required environment variables: {missing}")

        Config._config_data = cfg_interpolated
        Config._config_path = current_load_path # Store the actual path used for main config
        Config._model_params_path = current_model_params_path # Store path used for model params
        return cfg_interpolated

    @staticmethod
    def get(key, default=None):
        """
        Get a configuration value using dot notation (e.g., 'database.host').
        Loads config if not already loaded.
        If key is None or an empty string, returns the entire configuration.
        """
        if Config._config_data is None:
            try:
                Config.load()
            except ConfigError as e:
                logger.error(f"Failed to load configuration for key '{key}': {e}")
                # Use fallback configuration for production stability
                Config._config_data = Config._get_fallback_config()
                logger.warning("Using fallback configuration due to loading failure")
        if key is None or key == "": # Handle request for the whole config
            return Config._config_data

        val = Config._config_data
        parts = key.split('.')
        for part in parts:
            if not isinstance(val, dict):
                # If we encounter a non-dict structure while traversing, key path is invalid
                return default
            val = val.get(part)
            if val is None:
                # Key part not found
                return default
        return val

    @staticmethod
    def get_path(*keys, default=None):
        """
        Get a configuration value using multiple keys (e.g., 'database', 'host').
        DEPRECATED in favor of get() with dot notation, but kept for compatibility.
        """
        # This can be simplified by joining keys with '.' and calling get()
        key_string = ".".join(keys)
        return Config.get(key_string, default)

    @staticmethod
    def _get_fallback_config():
        """Get fallback configuration for production stability."""
        return {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'goaldiggers',
                'user': 'postgres',
                'password': 'password'
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'ml': {
                'model_path': 'models/',
                'cache_size': 100,
                'timeout': 30
            },
            'data': {
                'cache_ttl': 300,
                'max_retries': 3,
                'timeout': 10
            },
            'error_handling': {
                'max_retries': 3,
                'retry_delay': 1,
                'fallback_enabled': True
            }
        }

class ConfigLoader:
    """Configuration loader with fallback capabilities"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or DEFAULT_CONFIG_FILE_PATH
        self._config = None
    
    def get_fallback_config(self):
        """Get fallback configuration for production stability."""
        return {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'goaldiggers',
                'user': 'postgres',
                'password': 'password'
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'ml': {
                'model_path': 'models/',
                'cache_size': 100,
                'timeout': 30
            },
            'data': {
                'cache_ttl': 300,
                'max_retries': 3,
                'timeout': 10
            },
            'error_handling': {
                'max_retries': 3,
                'retry_delay': 1,
                'fallback_enabled': True
            }
        }

def load_config(config_file: str = None):
    """Load configuration with fallback handling"""
    try:
        config_path = config_file or DEFAULT_CONFIG_FILE_PATH
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return config_data
        else:
            logger.warning(f"Config file {config_path} not found, using fallback")
            loader = ConfigLoader()
            return loader.get_fallback_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        loader = ConfigLoader()
        return loader.get_fallback_config()

def get_config():
    """Get the configuration dictionary, ensuring required env vars are set."""
    import os

    # Fallbacks for missing environment variables
    if 'APP_ENV' not in os.environ:
        os.environ['APP_ENV'] = 'production'
    if 'DATA_DB_PATH' not in os.environ:
        os.environ['DATA_DB_PATH'] = 'data/football.db'
    if 'LOG_LEVEL' not in os.environ:
        os.environ['LOG_LEVEL'] = 'INFO'
    try:
        return load_config()
    except Exception as e:
        logger.warning(f"Could not load config: {e}, using fallback")
        config_loader = ConfigLoader()
        return config_loader.get_fallback_config()
