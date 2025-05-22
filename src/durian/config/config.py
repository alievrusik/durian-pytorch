import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: str):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
            
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Loaded configuration from {self.config_path}")
        return config
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
        
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self.config.update(updates)
        
    def save(self, path: str = None) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Path to save configuration to (defaults to original path)
        """
        if path is None:
            path = self.config_path
            
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        logger.info(f"Saved configuration to {path}")
        
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Model configuration dictionary
        """
        return self.config.get("model", {})
        
    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration.
        
        Returns:
            Training configuration dictionary
        """
        return self.config.get("training", {})
        
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration.
        
        Returns:
            Data configuration dictionary
        """
        return self.config.get("data", {})
        
    @property
    def audio_config(self) -> Dict[str, Any]:
        """Get audio configuration.
        
        Returns:
            Audio configuration dictionary
        """
        return self.config.get("audio", {})
        
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration.
        
        Returns:
            Logging configuration dictionary
        """
        return self.config.get("logging", {}) 