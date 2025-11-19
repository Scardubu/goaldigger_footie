from utils.config import Config, ConfigError


def validate_env():
    try:
        Config.load()
    except ConfigError as e:
        print(f"ERROR: {e}")
        exit(1)