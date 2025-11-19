#!/usr/bin/env python3
"""
Unified Configuration Manager for GoalDiggers Platform

Integrates all configuration systems with cascading lookup:
- Primary: utils/config.py (main configuration)
- Secondary: config/app_config.py (application-specific)
- Fallback: Production defaults (robust fallback)

Features:
- Standardized ${VAR_NAME} environment variable interpolation
- Singleton pattern for consistent access
- Configuration change propagation
- Robust error handling and fallback mechanisms
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)

class UnifiedConfigManager:
    """
    Unified configuration manager with cascading lookup and robust fallback.
    
    Configuration Priority:
    1. Primary Config (utils/config.py)
    2. App Config (config/app_config.py) 
    3. Environment Variables
    4. Production Fallback Defaults
    """
    
    def __init__(self):
        """Initialize unified configuration manager."""
        self.project_root = Path(__file__).parent.parent
        self.config_cache = {}
        self.last_reload = datetime.now()
        
        # Configuration sources
        self.primary_config = None
        self.app_config = None
        self.fallback_config = self._create_production_fallback()
        
        # Environment variable pattern
        self.env_var_pattern = r'\$\{([^}]+)\}'
        
        # Initialize configuration sources
        self._initialize_config_sources()
        
        logger.info("🔧 Unified Configuration Manager initialized")
    
    def _initialize_config_sources(self):
        """Initialize all configuration sources with error handling."""
        # Initialize primary config (utils/config.py)
        try:
            from utils.config import Config
            Config.load()
            self.primary_config = Config
            logger.info("✅ Primary config (utils/config.py) loaded")
        except Exception as e:
            logger.warning(f"⚠️ Primary config loading failed: {e}")
            self.primary_config = None
        
        # Initialize app config (config/app_config.py)
        try:
            from config.app_config import AppConfig
            self.app_config = AppConfig()
            logger.info("✅ App config (config/app_config.py) loaded")
        except Exception as e:
            logger.warning(f"⚠️ App config loading failed: {e}")
            self.app_config = None
    
    def _create_production_fallback(self) -> Dict[str, Any]:
        """Create robust production fallback configuration."""
        return {
            # Database Configuration
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'goaldiggers',
                'user': 'goaldiggers_user',
                'password': 'fallback_password',
                'pool_size': 5,
                'timeout': 30
            },
            
            # ML Configuration
            'ml': {
                'model_path': 'models/',
                'prediction_timeout': 5.0,
                'ensemble_weights': {
                    'xgboost': 0.4,
                    'lightgbm': 0.3,
                    'random_forest': 0.3
                },
                'confidence_threshold': 0.6,
                'max_memory_mb': 400,
                'cross_league_enabled': True,
                'cross_league_confidence_penalty': 0.1
            },

            # Multi-League Configuration
            'multi_league': {
                'supported_leagues': [
                    'Premier League', 'La Liga', 'Bundesliga',
                    'Serie A', 'Ligue 1', 'Eredivisie'
                ],
                'default_leagues': ['Premier League', 'La Liga', 'Bundesliga', 'Serie A'],
                'data_sources': {
                    'primary': ['football-data', 'api-football'],
                    'secondary': ['understat', 'fbref']
                },
                'normalization': {
                    'enabled': True,
                    'quality_weights': {
                        'Premier League': 1.0,
                        'La Liga': 0.95,
                        'Bundesliga': 0.90,
                        'Serie A': 0.85,
                        'Ligue 1': 0.80,
                        'Eredivisie': 0.75
                    }
                },
                'cache_ttl': 300,  # 5 minutes
                'max_concurrent_requests': 6
            },
            
            # Dashboard Configuration
            'dashboard': {
                'port': 8501,
                'host': 'localhost',
                'title': 'GoalDiggers - Football Betting Insights',
                'theme': 'light',
                'cache_ttl': 300,
                'max_upload_size': 50,
                'enhanced_features': {
                    'Interface': True,
                    'enhanced_team_data': True,
                    'advanced_prediction_display': True,
                    'cross_league_analysis': True,
                    'animated_visualizations': True
                },
                'performance_targets': {
                    'load_time_seconds': 1.0,
                    'memory_usage_mb': 400.0,
                    'component_init_seconds': 0.5,
                    'prediction_render_seconds': 1.0,
                    'team_resolution_seconds': 0.1
                },
                'design_system': {
                    'brand_colors': {
                        'primary': '#1f4e79',
                        'secondary': '#28a745',
                        'accent': '#fd7e14',
                        'success': '#28a745',
                        'warning': '#ffc107',
                        'danger': '#dc3545'
                    },
                    'typography': {
                        'font_family': '"Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif',
                        'font_size_base': '1rem'
                    },
                    'responsive_breakpoints': {
                        'mobile': '768px',
                        'tablet': '1024px',
                        'desktop': '1200px'
                    }
                }
            },

            # Enhanced Team Data Configuration
            'team_data': {
                'supported_leagues': [
                    'Premier League', 'La Liga', 'Bundesliga',
                    'Serie A', 'Ligue 1', 'Eredivisie'
                ],
                'metadata_enabled': True,
                'fuzzy_matching_threshold': 0.7,
                'cache_ttl': 300,
                'resolution_timeout': 0.1,
                'validation_enabled': True,
                'cross_league_validation': True
            },

            # Prediction Display Configuration
            'prediction_display': {
                'animation_duration_ms': 800,
                'confidence_thresholds': {
                    'very_high': 0.8,
                    'high': 0.6,
                    'moderate': 0.4,
                    'low': 0.0
                },
                'visualization_settings': {
                    'chart_height': 400,
                    'show_probability_distribution': True,
                    'show_cross_league_analysis': True,
                    'show_entertaining_commentary': True,
                    'enable_interactive_elements': True
                },
                'performance_targets': {
                    'render_time_seconds': 1.0,
                    'chart_load_time_seconds': 0.5
                }
            },

            # Performance Configuration
            'performance': {
                'startup_timeout': 20.0,
                'component_timeout': 5.0,
                'memory_target_mb': 400,
                'gc_threshold': 0.8,
                'enhanced_components_enabled': True,
                'fallback_mechanisms_enabled': True
            },
            
            # Logging Configuration
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'file': 'goaldiggers.log',
                'max_size_mb': 10
            }
        }
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with cascading lookup.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            # Check cache first
            if key in self.config_cache:
                return self.config_cache[key]
            
            # Cascading lookup
            value = self._cascade_config_lookup(key, default)
            
            # Interpolate environment variables
            if isinstance(value, str):
                value = self._interpolate_env_vars(value)
            
            # Cache the result
            self.config_cache[key] = value
            
            return value
            
        except Exception as e:
            logger.error(f"Config lookup failed for key '{key}': {e}")
            return default
    
    def _cascade_config_lookup(self, key: str, default: Any = None) -> Any:
        """Perform cascading configuration lookup."""
        # 1. Try primary config (utils/config.py)
        if self.primary_config:
            try:
                value = self._get_nested_value(self.primary_config.get_config(), key)
                if value is not None:
                    return value
            except Exception as e:
                logger.debug(f"Primary config lookup failed for '{key}': {e}")
        
        # 2. Try app config (config/app_config.py)
        if self.app_config:
            try:
                value = self._get_nested_value(self.app_config.get_config(), key)
                if value is not None:
                    return value
            except Exception as e:
                logger.debug(f"App config lookup failed for '{key}': {e}")
        
        # 3. Try environment variables
        try:
            env_key = key.upper().replace('.', '_')
            env_value = os.getenv(env_key)
            if env_value is not None:
                return self._parse_env_value(env_value)
        except Exception as e:
            logger.debug(f"Environment variable lookup failed for '{key}': {e}")
        
        # 4. Try fallback config
        try:
            value = self._get_nested_value(self.fallback_config, key)
            if value is not None:
                return value
        except Exception as e:
            logger.debug(f"Fallback config lookup failed for '{key}': {e}")
        
        # 5. Return default
        return default
    
    def _get_nested_value(self, config_dict: Dict[str, Any], key: str) -> Any:
        """Get nested value from configuration dictionary using dot notation."""
        if not isinstance(config_dict, dict):
            return None
        
        keys = key.split('.')
        value = config_dict
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
    
    def _interpolate_env_vars(self, value: str) -> str:
        """Interpolate environment variables in string values."""
        import re
        
        def replace_env_var(match):
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            if env_value is not None:
                return env_value
            else:
                logger.warning(f"Environment variable '{var_name}' not found, keeping placeholder")
                return match.group(0)
        
        return re.sub(self.env_var_pattern, replace_env_var, value)
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse environment variable value to appropriate type."""
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value and update cache."""
        try:
            # Update cache
            self.config_cache[key] = value
            
            # Propagate to primary config if available
            if self.primary_config and hasattr(self.primary_config, 'set_config'):
                self.primary_config.set_config(key, value)
            
            logger.debug(f"Configuration updated: {key} = {value}")
            
        except Exception as e:
            logger.error(f"Failed to set config '{key}': {e}")
    
    def reload_config(self) -> bool:
        """Reload all configuration sources."""
        try:
            logger.info("🔄 Reloading configuration sources...")
            
            # Clear cache
            self.config_cache.clear()
            
            # Reinitialize sources
            self._initialize_config_sources()
            
            self.last_reload = datetime.now()
            logger.info("✅ Configuration reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Configuration reload failed: {e}")
            return False
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as a merged dictionary."""
        try:
            merged_config = {}
            
            # Start with fallback
            merged_config.update(self.fallback_config)
            
            # Merge app config
            if self.app_config:
                try:
                    app_config_dict = self.app_config.get_config()
                    if isinstance(app_config_dict, dict):
                        merged_config.update(app_config_dict)
                except Exception as e:
                    logger.debug(f"Failed to merge app config: {e}")
            
            # Merge primary config
            if self.primary_config:
                try:
                    primary_config_dict = self.primary_config.get_config()
                    if isinstance(primary_config_dict, dict):
                        merged_config.update(primary_config_dict)
                except Exception as e:
                    logger.debug(f"Failed to merge primary config: {e}")
            
            return merged_config
            
        except Exception as e:
            logger.error(f"Failed to get all config: {e}")
            return self.fallback_config.copy()
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return validation results."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'sources': {
                'primary_config': self.primary_config is not None,
                'app_config': self.app_config is not None,
                'fallback_config': True
            }
        }
        
        # Validate required configuration keys
        required_keys = [
            'database.host',
            'ml.model_path',
            'dashboard.port',
            'performance.startup_timeout'
        ]
        
        for key in required_keys:
            try:
                value = self.get_config(key)
                if value is None:
                    validation_results['errors'].append(f"Required config key missing: {key}")
                    validation_results['valid'] = False
            except Exception as e:
                validation_results['errors'].append(f"Config validation error for {key}: {e}")
                validation_results['valid'] = False
        
        # Check for warnings
        if not self.primary_config:
            validation_results['warnings'].append("Primary config (utils/config.py) not available")
        
        if not self.app_config:
            validation_results['warnings'].append("App config (config/app_config.py) not available")
        
        return validation_results

# Global unified config manager instance
_unified_config_instance = None

def get_unified_config() -> UnifiedConfigManager:
    """Get global unified configuration manager instance."""
    global _unified_config_instance
    if _unified_config_instance is None:
        _unified_config_instance = UnifiedConfigManager()
    return _unified_config_instance

def get_config(key: str, default: Any = None) -> Any:
    """Convenient function to get configuration value."""
    config_manager = get_unified_config()
    return config_manager.get_config(key, default)

def set_config(key: str, value: Any) -> None:
    """Convenient function to set configuration value."""
    config_manager = get_unified_config()
    config_manager.set_config(key, value)

def reload_config() -> bool:
    """Convenient function to reload configuration."""
    config_manager = get_unified_config()
    return config_manager.reload_config()
