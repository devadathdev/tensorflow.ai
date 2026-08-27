"""Configuration management for TensorVision AI."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Configuration manager using YAML files with environment variable overrides."""
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls, config_path: Optional[str] = None) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance
    
    def _load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.environ.get(
                'TENSORVISION_CONFIG',
                str(Path(__file__).parent.parent.parent / 'config.yaml')
            )
        
        self.config_path = Path(config_path)
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = {}
        
        self._apply_env_overrides()
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_mappings = {
            'TENSORVISION_EPOCHS': ('training', 'epochs', int),
            'TENSORVISION_BATCH_SIZE': ('dataset', 'batch_size', int),
            'TENSORVISION_LEARNING_RATE': ('training', 'learning_rate', float),
            'TENSORVISION_IMAGE_SIZE': ('dataset', 'image_size', lambda x: list(map(int, x.split(',')))),
            'TENSORVISION_MODEL_DIR': ('output', 'model_dir', str),
            'TENSORVISION_USE_GPU': ('hardware', 'use_gpu', lambda x: x.lower() == 'true'),
            'TENSORVISION_API_PORT': ('api', 'port', int),
        }
        
        for env_var, (section, key, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                if section not in self._config:
                    self._config[section] = {}
                self._config[section][key] = converter(value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'dataset.batch_size')."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self._config.get(section, {})
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save current configuration to YAML file."""
        save_path = Path(path) if path else self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    @property
    def dataset(self) -> Dict[str, Any]:
        return self.get_section('dataset')
    
    @property
    def augmentation(self) -> Dict[str, Any]:
        return self.get_section('augmentation')
    
    @property
    def model(self) -> Dict[str, Any]:
        return self.get_section('model')
    
    @property
    def training(self) -> Dict[str, Any]:
        return self.get_section('training')
    
    @property
    def output(self) -> Dict[str, Any]:
        return self.get_section('output')
    
    @property
    def logging(self) -> Dict[str, Any]:
        return self.get_section('logging')
    
    @property
    def inference(self) -> Dict[str, Any]:
        return self.get_section('inference')
    
    @property
    def api(self) -> Dict[str, Any]:
        return self.get_section('api')
    
    @property
    def hardware(self) -> Dict[str, Any]:
        return self.get_section('hardware')


def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    return Config(config_path)