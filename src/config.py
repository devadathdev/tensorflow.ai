"""Configuration management for TensorVision AI."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Configuration manager using YAML files with environment variable overrides."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._config: Dict[str, Any] = {}
        self._load_config(config_path)

    def _load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.environ.get(
                'TENSORVISION_CONFIG',
                str(Path(__file__).parent.parent.parent / 'config.yaml')
            )
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
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
                self._config.setdefault(section, {})[key] = converter(value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        value: Any = self._config
        for part in key.split('.'):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section."""
        return self._config.get(section, {})

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation."""
        keys = key.split('.')
        config = self._config
        for part in keys[:-1]:
            if part not in config or not isinstance(config[part], dict):
                config[part] = {}
            config = config[part]
        config[keys[-1]] = value

    def save(self, path: Optional[str] = None) -> None:
        """Save current configuration to YAML."""
        save_path = Path(path) if path else self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self._config, f, default_flow_style=False, sort_keys=False)

    @property
    def dataset(self): return self.get_section('dataset')
    @property
    def augmentation(self): return self.get_section('augmentation')
    @property
    def model(self): return self.get_section('model')
    @property
    def training(self): return self.get_section('training')
    @property
    def output(self): return self.get_section('output')
    @property
    def logging(self): return self.get_section('logging')
    @property
    def inference(self): return self.get_section('inference')
    @property
    def api(self): return self.get_section('api')
    @property
    def hardware(self): return self.get_section('hardware')


def get_config(config_path: Optional[str] = None) -> Config:
    """Return an independent configuration instance for the requested path."""
    return Config(config_path)
