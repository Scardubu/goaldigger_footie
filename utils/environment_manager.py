#!/usr/bin/env python3
"""
GoalDiggers Environment Manager
Comprehensive environment configuration and validation system
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class EnvironmentManager:
    """Manages environment configuration and validation for production deployment"""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize environment manager with optional custom env file"""
        self.project_root = Path(__file__).parent.parent
        self.env_file = env_file or self.project_root / ".env"
        self.config = {}
        self.validation_errors = []
        self.load_environment()
    
    def load_environment(self):
        """Load environment variables from .env file"""
        try:
            if self.env_file.exists():
                load_dotenv(self.env_file)
                logger.info(f"✅ Loaded environment from: {self.env_file}")
            else:
                logger.warning(f"⚠️ Environment file not found: {self.env_file}")
            
            self._build_config()
            
        except Exception as e:
            logger.error(f"❌ Failed to load environment: {e}")
            raise
    
    def _build_config(self):
        """Build comprehensive configuration from environment variables"""
        self.config = {
            # Environment Settings
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'debug_mode': os.getenv('DEBUG_MODE', 'true').lower() == 'true',
            'use_sample_data': os.getenv('USE_SAMPLE_DATA', 'true').lower() == 'true',
            'enable_caching': os.getenv('ENABLE_CACHING', 'true').lower() == 'true',
            
            # API Configuration
            'api_keys': {
                'football_data': os.getenv('FOOTBALL_DATA_API_KEY'),
                'api_football': os.getenv('API_FOOTBALL_KEY'),
                'weather_api': os.getenv('WEATHER_API_KEY') or os.getenv('OPENWEATHER_API_KEY'),
                'openai': os.getenv('OPENAI_API_KEY'),
                'deepseek': os.getenv('DEEPSEEK_API_KEY'),
                'huggingface': os.getenv('HUGGING_FACE_HUB_TOKEN'),
            },
            
            # Database Configuration
            'database': {
                'url': os.getenv('DATABASE_URL'),
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', '5432')),
                'name': os.getenv('DB_NAME', 'goaldiggers_prod'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'sqlite_path': os.getenv('SQLITE_DB_PATH', './data/football.db'),
            },
            
            # Redis Configuration
            'redis': {
                'url': os.getenv('REDIS_URL'),
                'host': os.getenv('REDIS_HOST', 'localhost'),
                'port': int(os.getenv('REDIS_PORT', '6379')),
                'db': int(os.getenv('REDIS_DB', '0')),
                'enabled': os.getenv('ENABLE_REDIS_CACHE', 'false').lower() == 'true',
            },
            
            # Server Configuration
            'server': {
                'port': int(os.getenv('SERVER_PORT', '8501')),
                'host': os.getenv('SERVER_HOST', '0.0.0.0'),
                'max_workers': int(os.getenv('MAX_WORKERS', '4')),
                'timeout': int(os.getenv('WORKER_TIMEOUT', '30')),
            },
            
            # Streamlit Configuration
            'streamlit': {
                'port': int(os.getenv('STREAMLIT_SERVER_PORT', '8501')),
                'enable_cors': os.getenv('STREAMLIT_SERVER_ENABLE_CORS', 'true').lower() == 'true',
                'enable_xsrf': os.getenv('STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION', 'false').lower() == 'true',
            },
            
            # Cache Configuration
            'cache': {
                'ttl': int(os.getenv('CACHE_TTL', '3600')),
                'enable_api_caching': os.getenv('ENABLE_API_CACHING', 'true').lower() == 'true',
                'enable_db_caching': os.getenv('ENABLE_DATABASE_CACHING', 'true').lower() == 'true',
            },
            
            # Security Configuration
            'security': {
                'secret_key': os.getenv('SECRET_KEY'),
                'jwt_secret': os.getenv('JWT_SECRET'),
                'allowed_origins': os.getenv('ALLOWED_ORIGINS', '').split(','),
            },
            
            # Feature Flags
            'features': {
                'real_data': os.getenv('ENABLE_REAL_DATA', 'true').lower() == 'true',
                'analytics': os.getenv('ENABLE_ANALYTICS', 'false').lower() == 'true',
                'dark_mode': os.getenv('ENABLE_DARK_MODE', 'true').lower() == 'true',
                'responsive_design': os.getenv('ENABLE_RESPONSIVE_DESIGN', 'true').lower() == 'true',
            },
            
            # Logging Configuration
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'format': os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            },
            
            # Paths
            'paths': {
                'project_root': os.getenv('PROJECT_ROOT', str(self.project_root)),
                'data_dir': os.getenv('DATA_DIR', str(self.project_root / 'data')),
                'static_url': os.getenv('STATIC_URL', '/static/'),
                'media_url': os.getenv('MEDIA_URL', '/media/'),
            }
        }
    
    def validate_production_config(self) -> Dict[str, Any]:
        """Validate configuration for production deployment"""
        validation_result = {
            'is_production_ready': True,
            'errors': [],
            'warnings': [],
            'missing_required': [],
            'recommendations': []
        }
        
        # Check environment setting
        if self.config['environment'] != 'production':
            validation_result['warnings'].append(
                f"Environment is '{self.config['environment']}', expected 'production'"
            )
        
        # Check debug mode
        if self.config['debug_mode']:
            validation_result['errors'].append("DEBUG_MODE should be false in production")
            validation_result['is_production_ready'] = False
        
        # Check sample data usage
        if self.config['use_sample_data']:
            validation_result['errors'].append("USE_SAMPLE_DATA should be false in production")
            validation_result['is_production_ready'] = False
        
        # Check required API keys
        required_api_keys = ['football_data', 'api_football']
        for key_name in required_api_keys:
            if not self.config['api_keys'].get(key_name):
                validation_result['missing_required'].append(f"{key_name.upper()}_API_KEY")
                validation_result['is_production_ready'] = False
        
        # Check optional API keys
        optional_api_keys = ['weather_api', 'openai']
        for key_name in optional_api_keys:
            if not self.config['api_keys'].get(key_name):
                validation_result['warnings'].append(f"Optional API key not configured: {key_name}")
        
        # Check database configuration
        if not self.config['database']['url'] and not self.config['database']['user']:
            validation_result['warnings'].append("Using SQLite database - consider PostgreSQL for production")
        
        # Check security configuration
        if not self.config['security']['secret_key']:
            validation_result['errors'].append("SECRET_KEY must be set for production")
            validation_result['is_production_ready'] = False
        
        # Check caching configuration
        if not self.config['enable_caching']:
            validation_result['recommendations'].append("Enable caching for better performance")
        
        return validation_result
    
    def get_api_config(self) -> Dict[str, str]:
        """Get API configuration for data integrators"""
        return {
            'football_data_key': self.config['api_keys']['football_data'],
            'api_football_key': self.config['api_keys']['api_football'],
            'weather_api_key': self.config['api_keys']['weather_api'],
            'football_data_host': os.getenv('API_FOOTBALL_HOST', 'api-football-v1.p.rapidapi.com'),
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        if self.config['database']['url']:
            return {
                'type': 'postgresql',
                'url': self.config['database']['url'],
                'host': self.config['database']['host'],
                'port': self.config['database']['port'],
                'name': self.config['database']['name'],
                'user': self.config['database']['user'],
                'password': self.config['database']['password'],
            }
        else:
            return {
                'type': 'sqlite',
                'path': self.config['database']['sqlite_path'],
            }
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get Streamlit configuration for deployment"""
        return {
            'port': self.config['streamlit']['port'],
            'host': self.config['server']['host'],
            'enable_cors': self.config['streamlit']['enable_cors'],
            'enable_xsrf_protection': self.config['streamlit']['enable_xsrf'],
            'max_upload_size': 50,  # MB
            'theme': {
                'primaryColor': '#1f77b4',
                'backgroundColor': '#ffffff',
                'secondaryBackgroundColor': '#f0f2f6',
                'textColor': '#262730',
            }
        }
    
    def is_production_ready(self) -> bool:
        """Quick check if configuration is production ready"""
        validation = self.validate_production_config()
        return validation['is_production_ready']
    
    def generate_render_config(self) -> Dict[str, Any]:
        """Generate Render.com deployment configuration"""
        return {
            'name': 'goaldiggers-platform',
            'type': 'web',
            'env': 'python',
            'buildCommand': 'pip install -r requirements.txt',
            'startCommand': f'streamlit run production_homepage.py --server.port=$PORT --server.address=0.0.0.0',
            'envVars': [
                {'key': 'ENVIRONMENT', 'value': 'production'},
                {'key': 'DEBUG_MODE', 'value': 'false'},
                {'key': 'USE_SAMPLE_DATA', 'value': 'false'},
                {'key': 'ENABLE_CACHING', 'value': 'true'},
                {'key': 'FOOTBALL_DATA_API_KEY', 'value': '[Add your key]'},
                {'key': 'API_FOOTBALL_KEY', 'value': '[Add your key]'},
                {'key': 'WEATHER_API_KEY', 'value': '[Add your key]'},
            ],
            'healthCheckPath': '/health',
            'autoDeploy': True,
        }
    
    def export_config(self, output_path: str):
        """Export current configuration to JSON file"""
        config_data = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'environment': self.config['environment'],
            'configuration': self.config,
            'validation': self.validate_production_config(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"✅ Configuration exported to: {output_path}")

# Global environment manager instance
env_manager = EnvironmentManager()

def get_config() -> Dict[str, Any]:
    """Get the current configuration"""
    return env_manager.config

def get_api_config() -> Dict[str, str]:
    """Get API configuration"""
    return env_manager.get_api_config()

def get_database_config() -> Dict[str, Any]:
    """Get database configuration"""
    return env_manager.get_database_config()

def is_production() -> bool:
    """Check if running in production mode"""
    return env_manager.config['environment'] == 'production'

def is_debug_enabled() -> bool:
    """Check if debug mode is enabled"""
    return env_manager.config['debug_mode']

def should_use_real_data() -> bool:
    """Check if should use real data (not sample data)"""
    return not env_manager.config['use_sample_data']

if __name__ == "__main__":
    # Test the environment manager
    print("🔧 GoalDiggers Environment Manager")
    print("=" * 50)
    
    validation = env_manager.validate_production_config()
    
    print(f"🌍 Environment: {env_manager.config['environment']}")
    print(f"🔍 Debug Mode: {env_manager.config['debug_mode']}")
    print(f"📊 Use Sample Data: {env_manager.config['use_sample_data']}")
    print(f"⚡ Caching Enabled: {env_manager.config['enable_caching']}")
    print(f"✅ Production Ready: {validation['is_production_ready']}")
    
    if validation['errors']:
        print("\n❌ Errors:")
        for error in validation['errors']:
            print(f"  • {error}")
    
    if validation['warnings']:
        print("\n⚠️ Warnings:")
        for warning in validation['warnings']:
            print(f"  • {warning}")
    
    if validation['missing_required']:
        print("\n🔑 Missing Required:")
        for missing in validation['missing_required']:
            print(f"  • {missing}")
    
    # Export configuration
    config_path = env_manager.project_root / "environment_config_export.json"
    env_manager.export_config(str(config_path))
    print(f"\n📝 Configuration exported to: {config_path}")
