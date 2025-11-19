import os

import yaml


class AppConfig:
    _config = None
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AppConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._load_config()

    @classmethod
    def _load_config(cls):
        if cls._config is None:
            cls._config = {}
            config_paths = [
                os.path.join(os.path.dirname(__file__), 'config.yaml'),
                os.path.join(os.path.dirname(__file__), 'dashboard_config.yaml')
            ]
            for path in config_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        cls._config.update(yaml.safe_load(f))

    def get(self, key, default=None):
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default


    # General App Info
    APP_NAME = "GoalDiggers"
    APP_VERSION = "2.0.0"
    APP_ICON = "⚽"

def load_config():
    """Load and return the AppConfig instance."""
    return AppConfig()

    # Theming
    @property
    def THEME_PRIMARY_COLOR(self):
        return self.get('dashboard.theme.primary_color', '#FF4B4B')

    @property
    def THEME_SECONDARY_COLOR(self):
        return self.get('dashboard.theme.secondary_color', '#FFFFFF')

    @property
    def THEME_BACKGROUND_COLOR(self):
        return self.get('dashboard.theme.background_color', '#0E1117')

    @property
    def THEME_TEXT_COLOR(self):
        return self.get('dashboard.theme.text_color', '#FAFAFA')

    @property
    def THEME_FONT(self):
        return self.get('dashboard.theme.font_family', 'sans-serif')

    # Database
    @property
    def DATABASE_URI(self):
        return self.get('database.uri', 'sqlite:///data/football.db')
        
    def generate_custom_css(self):
        """Generate custom CSS based on theme configuration."""
        return f"""
        /* GoalDiggers Theme - Generated from AppConfig */
        :root {{
            --primary-color: {self.THEME_PRIMARY_COLOR};
            --secondary-color: {self.THEME_SECONDARY_COLOR};
            --background-color: {self.THEME_BACKGROUND_COLOR};
            --text-color: {self.THEME_TEXT_COLOR};
            --font-family: {self.THEME_FONT};
            --goaldiggers-gradient: linear-gradient(45deg, {self.THEME_PRIMARY_COLOR}, {self.THEME_SECONDARY_COLOR});
        }}
        
        .main-header {{
            font-family: var(--font-family);
            color: var(--primary-color);
            padding: 1rem 0;
            font-weight: 600;
        }}
        
        .card {{
            border-radius: 8px;
            border: 1px solid rgba(49, 51, 63, 0.2);
            padding: 1.5rem;
            margin-bottom: 1rem;
            background: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }}
        
        .card:hover {{
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 2rem;
            font-weight: 600;
            font-size: 0.875rem;
            white-space: nowrap;
        }}
        
        .badge.success {{
            background-color: rgba(25, 135, 84, 0.15);
            color: #198754;
        }}
        
        .badge.danger {{
            background-color: rgba(220, 53, 69, 0.15);
            color: #dc3545;
        }}
        
        .badge.warning {{
            background-color: rgba(255, 193, 7, 0.15);
            color: #ffc107;
        }}
        
        .badge.info {{
            background-color: rgba(13, 202, 240, 0.15);
            color: #0dcaf0;
        }}
        """
