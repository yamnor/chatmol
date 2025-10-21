# Batch processor configuration
import json
import os
from typing import Dict, Any, Optional

class BatchConfig:
    """Configuration management for batch processing."""
    
    DEFAULT_CONFIG = {
        "api": {
            "key": "",
            "model": "gemini-2.5-flash-lite",
            "use_google_search": True,
            "timeout": 10
        },
        "execution": {
            "max_iterations": 10,
            "max_duration_minutes": 30,
            "stop_on_error": False,
            "continue_on_xyz_failure": True
        },
        "retry": {
            "max_retries": 3,
            "retry_delay_seconds": 5,
            "exponential_backoff": True
        },
        "logging": {
            "log_level": "INFO",
            "log_file": "data/output/logs/batch_execution.log",
            "detailed_log_file": "data/output/logs/batch_detailed.json",
            "console_output": True
        },
        "cache": {
            "enabled": True,
            "base_directory": "cache"
        },
        "output": {
            "results_directory": "data/output/results",
            "include_statistics": True,
            "export_format": "json"
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration."""
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # Override with environment variables
        self.load_from_env()
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                self._merge_config(file_config)
            print(f"Configuration loaded from: {config_file}")
        except Exception as e:
            print(f"Error loading config file {config_file}: {e}")
    
    def load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            'GEMINI_API_KEY': ['api', 'key'],
            'GEMINI_MODEL': ['api', 'model'],
            'BATCH_MAX_ITERATIONS': ['execution', 'max_iterations'],
            'BATCH_MAX_DURATION': ['execution', 'max_duration_minutes'],
            'LOG_LEVEL': ['logging', 'log_level']
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested_value(config_path, value)
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration into existing config."""
        for key, value in new_config.items():
            if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def _set_nested_value(self, path: list, value: Any):
        """Set nested configuration value."""
        current = self.config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def get(self, path: str, default=None):
        """Get configuration value by dot notation."""
        keys = path.split('.')
        current = self.config
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def save_to_file(self, config_file: str):
        """Save configuration to JSON file."""
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            print(f"Configuration saved to: {config_file}")
        except Exception as e:
            print(f"Error saving config file {config_file}: {e}")
    
    def validate(self) -> bool:
        """Validate configuration."""
        required_fields = [
            'api.key',
            'execution.max_iterations',
            'logging.log_level'
        ]
        
        for field in required_fields:
            if not self.get(field):
                print(f"Missing required configuration: {field}")
                return False
        
        return True
