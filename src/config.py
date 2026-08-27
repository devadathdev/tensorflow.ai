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
            cls._load_config(config_path)
        return cls._instance

    @classmethod
    def _load_config(cls, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.environ.get(
                'TENSORVISION_CONFIG',
                str(Path(__file__).resolve().parent.parent / 'config.yaml')
            )

        cls.config_path = Path(config_path)

        if cls.config_path.exists():
            with open(cls.config_path, 'r', encoding='utf-8') as f:
                cls._config = yaml.safe_load(f) or {}
        else:
            cls._config = {}

        cls._apply_env_overrides()

    @classmethod
    def _apply_env_overrides(cls) -> None:
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
                cls._config.setdefault(section, {})[key] = converter(value)

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
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self._config
        for key_part in keys[:-1]:
            config = config.setdefault(key_part, {})
        config[keys[-1]] = value

    def save(self, path: Optional[str] = None) -> None:
        """Save current configuration to YAML."""
        save_path = Path(path) if path else self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self._config, f, default_flow_style=False, sort_keys=False)

    @property
    def dataset(self) -> Dict[str, Any]: return self.get_section('dataset')
    @property
    def augmentation(self) -> Dict[str, Any]: return self.get_section('augmentation')
    @property
    def model(self) -> Dict[str, Any]: return self.get_section('model')
    @property
    def training(self) -> Dict[str, Any]: return self.get_section('training')
    @property
    def output(self) -> Dict[str, Any]: return self.get_section('output')
    @property
    def logging(self) -> Dict[str, Any]: return self.get_section('logging')
    @property
    def inference(self) -> Dict[str, Any]: return self.get_section('inference')
    @property
    def api(self) -> Dict[str, Any]: return self.get_section('api')
    @property
    def hardware(self) -> Dict[str, Any]: return self.get_section('hardware')


def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    return Config(config_path)
