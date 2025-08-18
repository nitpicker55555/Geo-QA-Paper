"""
Configuration file for the optimized Flask application.

This module contains all configuration settings, constants, and environment-specific
configurations for the geospatial Flask application.
"""

import os
from typing import Dict, List, Set, Union


class BaseConfig:
    """Base configuration class with common settings."""
    
    # Application settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # File upload settings
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS: Set[str] = {
        'txt', 'pdf', 'png', 'jpg', 'jpeg', 'doc', 'docx', 'xlsx', 'csv', 'ttl'
    }
    
    # Cache settings
    GEOJSON_CACHE_SIZE = 128
    RESPONSE_CACHE_SIZE = 256
    IP_LOCATION_CACHE_SIZE = 1000
    
    # Session settings
    SESSION_TIMEOUT = 3600  # 1 hour
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    PERMANENT_SESSION_LIFETIME = SESSION_TIMEOUT
    
    # File paths
    ENV_PATH = 'config/.env'
    GEOJSON_DIR = 'static/geojson'
    RESPONSES_FILE = 'responses.jsonl'
    DATA_LOG_FILE = 'static/data3.txt'
    TEST_DATA_FILE = 'jsonl_files/eval_relationships_four_final_eval3.jsonl'
    STATIC_FOLDER = 'static'
    TEMPLATE_FOLDER = 'templates'
    
    # GeoJSON file mappings
    GEOJSON_FILES: Dict[str, str] = {
        '1': 'buildings_geojson.geojson',
        '2': 'land_geojson.geojson',
        '3': 'soil_maxvorstadt_geojson.geojson',
        '4': 'points_geojson.geojson',
        '5': 'lines_geojson.geojson',
    }
    
    # Default bounding box (Munich)
    DEFAULT_BOUNDING_BOX = {
        'region_name': 'Munich',
        'coordinates': [48.061625, 48.248098, 11.360777, 11.72291],
        'wkb': '01030000000100000005000000494C50C3B7B82640D9CEF753E3074840494C50C3B7B82640FC19DEACC11F484019E76F4221722740FC19DEACC11F484019E76F4221722740D9CEF753E3074840494C50C3B7B82640D9CEF753E3074840'
    }
    
    # Security settings
    DANGEROUS_CODE_PATTERNS: List[str] = [
        r'import\s+os',
        r'import\s+subprocess',
        r'from\s+os\s+import',
        r'from\s+subprocess\s+import',
        r'__import__',
        r'eval\s*\(',
        r'exec\s*\(',
    ]
    
    # Function patterns for code processing
    GEOSPATIAL_FUNCTIONS: List[str] = [
        'geo_filter(',
        'id_list_of_entity(',
        'id_list_of_entity_fast(',
        'add_or_subtract_entities(',
        'area_filter(',
        'set_bounding_box(',
        'traffic_navigation('
    ]
    
    # Response processing settings
    MAX_OUTPUT_LENGTH = 4000
    MAX_RESPONSE_LENGTH = 600
    SHORT_RESPONSE_THRESHOLD = 1000
    SHORT_LIST_LENGTH = 40
    LARGE_DATA_THRESHOLD = 10000
    MAX_DATA_DISPLAY = 20000
    
    # API settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    REQUEST_TIMEOUT = 30
    IP_API_TIMEOUT = 5
    
    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # WebSocket settings
    SOCKETIO_ASYNC_MODE = 'threading'
    SOCKETIO_MANAGE_SESSION = True
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"


class DevelopmentConfig(BaseConfig):
    """Development configuration."""
    
    DEBUG = True
    SESSION_COOKIE_SECURE = False
    LOG_LEVEL = 'DEBUG'
    
    # Development-specific overrides
    GEOJSON_CACHE_SIZE = 32  # Smaller cache for development
    RESPONSE_CACHE_SIZE = 64


class ProductionConfig(BaseConfig):
    """Production configuration."""
    
    DEBUG = False
    SESSION_COOKIE_SECURE = True
    LOG_LEVEL = 'WARNING'
    
    # Production-specific settings
    SECRET_KEY = os.getenv('SECRET_KEY')  # Must be set in production
    
    # Enhanced security
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'
    
    # Production optimizations
    GEOJSON_CACHE_SIZE = 256
    RESPONSE_CACHE_SIZE = 512


class TestingConfig(BaseConfig):
    """Testing configuration."""
    
    TESTING = True
    DEBUG = True
    SESSION_COOKIE_SECURE = False
    
    # Testing-specific settings
    UPLOAD_FOLDER = 'test_uploads'
    RESPONSES_FILE = 'test_responses.jsonl'
    DATA_LOG_FILE = 'test_data.txt'
    
    # Disable caching for testing
    GEOJSON_CACHE_SIZE = 1
    RESPONSE_CACHE_SIZE = 1


# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name: str = None) -> BaseConfig:
    """
    Get configuration class based on environment.
    
    Args:
        config_name: Configuration name (development, production, testing)
        
    Returns:
        Configuration class instance
    """
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'default')
    
    return config_map.get(config_name, config_map['default'])


# Utility functions for configuration
def validate_config(config: BaseConfig) -> bool:
    """
    Validate configuration settings.
    
    Args:
        config: Configuration instance to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_settings = [
        'SECRET_KEY',
        'UPLOAD_FOLDER',
        'GEOJSON_DIR'
    ]
    
    for setting in required_settings:
        if not hasattr(config, setting) or not getattr(config, setting):
            print(f"Missing required configuration: {setting}")
            return False
    
    # Validate file paths exist
    paths_to_check = [
        config.UPLOAD_FOLDER,
        config.STATIC_FOLDER,
        config.TEMPLATE_FOLDER
    ]
    
    for path in paths_to_check:
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
                print(f"Created directory: {path}")
            except OSError as e:
                print(f"Cannot create directory {path}: {e}")
                return False
    
    return True


def setup_logging(config: BaseConfig) -> None:
    """
    Setup application logging based on configuration.
    
    Args:
        config: Configuration instance
    """
    import logging
    
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )


# Export commonly used configuration
Config = get_config()