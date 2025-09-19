#!/usr/bin/env python3
"""
DAWN Telemetry Configuration System
===================================

Comprehensive configuration management for DAWN's telemetry system.
Provides runtime configurability, environment variable support, and
predefined profiles for different deployment scenarios.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class TelemetryProfile(Enum):
    """Predefined telemetry configuration profiles."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    DEBUG = "debug"
    MINIMAL = "minimal"
    HIGH_PERFORMANCE = "high_performance"

@dataclass
class BufferConfig:
    """Telemetry buffer configuration."""
    max_size: int = 10000
    auto_flush_interval: float = 30.0
    auto_flush_enabled: bool = True
    flush_on_shutdown: bool = True

@dataclass
class FilteringConfig:
    """Telemetry filtering configuration."""
    min_level: str = "INFO"  # DEBUG, INFO, WARN, ERROR, CRITICAL
    subsystem_filters: Dict[str, Dict[str, Any]] = None
    component_filters: Dict[str, Dict[str, Any]] = None
    event_type_filters: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.subsystem_filters is None:
            self.subsystem_filters = {}
        if self.component_filters is None:
            self.component_filters = {}
        if self.event_type_filters is None:
            self.event_type_filters = {}

@dataclass
class OutputConfig:
    """Telemetry output configuration."""
    enabled_formats: List[str] = None  # json, csv, prometheus, influxdb
    output_directory: str = "runtime/telemetry"
    file_rotation_size_mb: int = 100
    max_files: int = 10
    compress_rotated: bool = True
    
    def __post_init__(self):
        if self.enabled_formats is None:
            self.enabled_formats = ["json"]

@dataclass
class SystemMetricsConfig:
    """System metrics collection configuration."""
    enabled: bool = True
    collection_interval: float = 10.0
    collect_cpu: bool = True
    collect_memory: bool = True
    collect_disk: bool = True
    collect_network: bool = False
    collect_process_metrics: bool = True

@dataclass
class PerformanceConfig:
    """Performance monitoring configuration."""
    max_overhead_percent: float = 2.0
    disable_on_high_load: bool = True
    high_load_cpu_threshold: float = 0.85
    high_load_memory_threshold: float = 0.90
    performance_sampling_rate: float = 1.0  # 0.0 to 1.0

@dataclass
class SecurityConfig:
    """Security and privacy configuration."""
    scrub_sensitive_data: bool = True
    encrypt_at_rest: bool = False
    audit_access: bool = False
    max_retention_days: int = 30
    auto_purge_enabled: bool = True
    
    # Patterns to scrub from telemetry data
    sensitive_patterns: List[str] = None
    
    def __post_init__(self):
        if self.sensitive_patterns is None:
            self.sensitive_patterns = [
                r"password", r"token", r"key", r"secret", r"credential",
                r"api_key", r"auth", r"session_id"
            ]

class TelemetryConfig:
    """
    Main telemetry configuration manager.
    
    Supports loading from:
    - Configuration files (JSON)
    - Environment variables
    - Predefined profiles
    - Runtime updates
    """
    
    def __init__(self, config_file: Optional[str] = None, profile: Optional[str] = None):
        """
        Initialize telemetry configuration.
        
        Args:
            config_file: Path to JSON configuration file
            profile: Name of predefined profile to use
        """
        # Initialize with defaults
        self.enabled = True
        self.buffer = BufferConfig()
        self.filtering = FilteringConfig()
        self.output = OutputConfig()
        self.system_metrics = SystemMetricsConfig()
        self.performance = PerformanceConfig()
        self.security = SecurityConfig()
        
        # Load configuration
        if profile:
            self.load_profile(profile)
        elif config_file:
            self.load_from_file(config_file)
        else:
            self.load_from_environment()
        
        # Ensure output directory exists
        self._create_output_directory()
        
        logger.info(f"🔧 Telemetry configuration initialized")
    
    def load_profile(self, profile_name: str) -> None:
        """Load a predefined configuration profile."""
        if profile_name not in TELEMETRY_PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(TELEMETRY_PROFILES.keys())}")
        
        profile_config = TELEMETRY_PROFILES[profile_name]
        self._update_from_dict(profile_config)
        
        logger.info(f"Loaded telemetry profile: {profile_name}")
    
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from JSON file."""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                self._update_from_dict(config_data)
                logger.info(f"Configuration loaded from {config_file}")
            else:
                logger.warning(f"Configuration file not found: {config_file}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
    
    def load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # Main settings
        self.enabled = self._get_bool_env("DAWN_TELEMETRY_ENABLED", True)
        
        # Buffer settings
        self.buffer.max_size = self._get_int_env("DAWN_TELEMETRY_BUFFER_SIZE", 10000)
        self.buffer.auto_flush_interval = self._get_float_env("DAWN_TELEMETRY_FLUSH_INTERVAL", 30.0)
        self.buffer.auto_flush_enabled = self._get_bool_env("DAWN_TELEMETRY_AUTO_FLUSH", True)
        
        # Filtering settings
        self.filtering.min_level = os.getenv("DAWN_TELEMETRY_LEVEL", "INFO")
        
        # Output settings
        output_formats = os.getenv("DAWN_TELEMETRY_FORMATS", "json")
        self.output.enabled_formats = [f.strip() for f in output_formats.split(",")]
        self.output.output_directory = os.getenv("DAWN_TELEMETRY_DIR", "runtime/telemetry")
        self.output.file_rotation_size_mb = self._get_int_env("DAWN_TELEMETRY_FILE_SIZE_MB", 100)
        
        # System metrics settings
        self.system_metrics.enabled = self._get_bool_env("DAWN_TELEMETRY_SYSTEM_METRICS", True)
        self.system_metrics.collection_interval = self._get_float_env("DAWN_TELEMETRY_SYSTEM_INTERVAL", 10.0)
        
        # Performance settings
        self.performance.max_overhead_percent = self._get_float_env("DAWN_TELEMETRY_MAX_OVERHEAD", 2.0)
        self.performance.disable_on_high_load = self._get_bool_env("DAWN_TELEMETRY_DISABLE_HIGH_LOAD", True)
        
        logger.info("Configuration loaded from environment variables")
    
    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        if "enabled" in config_data:
            self.enabled = config_data["enabled"]
        
        if "buffer" in config_data:
            self._update_dataclass(self.buffer, config_data["buffer"])
        
        if "filtering" in config_data:
            self._update_dataclass(self.filtering, config_data["filtering"])
        
        if "output" in config_data:
            self._update_dataclass(self.output, config_data["output"])
        
        if "system_metrics" in config_data:
            self._update_dataclass(self.system_metrics, config_data["system_metrics"])
        
        if "performance" in config_data:
            self._update_dataclass(self.performance, config_data["performance"])
        
        if "security" in config_data:
            self._update_dataclass(self.security, config_data["security"])
    
    def _update_dataclass(self, dataclass_instance, update_dict: Dict[str, Any]) -> None:
        """Update dataclass fields from dictionary."""
        for key, value in update_dict.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
    
    def _get_bool_env(self, env_var: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(env_var)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def _get_int_env(self, env_var: str, default: int) -> int:
        """Get integer environment variable."""
        try:
            return int(os.getenv(env_var, str(default)))
        except ValueError:
            return default
    
    def _get_float_env(self, env_var: str, default: float) -> float:
        """Get float environment variable."""
        try:
            return float(os.getenv(env_var, str(default)))
        except ValueError:
            return default
    
    def _create_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        try:
            Path(self.output.output_directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create output directory {self.output.output_directory}: {e}")
    
    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to JSON file."""
        try:
            config_data = {
                "enabled": self.enabled,
                "buffer": asdict(self.buffer),
                "filtering": asdict(self.filtering),
                "output": asdict(self.output),
                "system_metrics": asdict(self.system_metrics),
                "performance": asdict(self.performance),
                "security": asdict(self.security)
            }
            
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "enabled": self.enabled,
            "buffer": asdict(self.buffer),
            "filtering": asdict(self.filtering),
            "output": asdict(self.output),
            "system_metrics": asdict(self.system_metrics),
            "performance": asdict(self.performance),
            "security": asdict(self.security)
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Validate buffer configuration
        if self.buffer.max_size <= 0:
            issues.append("buffer.max_size must be positive")
        
        if self.buffer.auto_flush_interval <= 0:
            issues.append("buffer.auto_flush_interval must be positive")
        
        # Validate filtering configuration
        valid_levels = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]
        if self.filtering.min_level not in valid_levels:
            issues.append(f"filtering.min_level must be one of: {valid_levels}")
        
        # Validate output configuration
        valid_formats = ["json", "csv", "prometheus", "influxdb"]
        for fmt in self.output.enabled_formats:
            if fmt not in valid_formats:
                issues.append(f"output format '{fmt}' not supported. Valid: {valid_formats}")
        
        if self.output.file_rotation_size_mb <= 0:
            issues.append("output.file_rotation_size_mb must be positive")
        
        if self.output.max_files <= 0:
            issues.append("output.max_files must be positive")
        
        # Validate system metrics configuration
        if self.system_metrics.collection_interval <= 0:
            issues.append("system_metrics.collection_interval must be positive")
        
        # Validate performance configuration
        if not 0.0 <= self.performance.max_overhead_percent <= 100.0:
            issues.append("performance.max_overhead_percent must be between 0.0 and 100.0")
        
        if not 0.0 <= self.performance.performance_sampling_rate <= 1.0:
            issues.append("performance.performance_sampling_rate must be between 0.0 and 1.0")
        
        if not 0.0 <= self.performance.high_load_cpu_threshold <= 1.0:
            issues.append("performance.high_load_cpu_threshold must be between 0.0 and 1.0")
        
        # Validate security configuration
        if self.security.max_retention_days <= 0:
            issues.append("security.max_retention_days must be positive")
        
        return issues
    
    def get_runtime_summary(self) -> Dict[str, Any]:
        """Get runtime configuration summary."""
        return {
            "enabled": self.enabled,
            "min_level": self.filtering.min_level,
            "buffer_size": self.buffer.max_size,
            "output_formats": self.output.enabled_formats,
            "output_directory": self.output.output_directory,
            "system_metrics_enabled": self.system_metrics.enabled,
            "max_overhead_percent": self.performance.max_overhead_percent,
            "auto_flush_interval": self.buffer.auto_flush_interval
        }

# Predefined configuration profiles
TELEMETRY_PROFILES = {
    "development": {
        "enabled": True,
        "buffer": {
            "max_size": 5000,
            "auto_flush_interval": 10.0,
            "auto_flush_enabled": True
        },
        "filtering": {
            "min_level": "DEBUG"
        },
        "output": {
            "enabled_formats": ["json"],
            "file_rotation_size_mb": 50,
            "max_files": 5
        },
        "system_metrics": {
            "enabled": True,
            "collection_interval": 5.0,
            "collect_process_metrics": True
        },
        "performance": {
            "max_overhead_percent": 5.0,
            "disable_on_high_load": False,
            "performance_sampling_rate": 1.0
        }
    },
    
    "production": {
        "enabled": True,
        "buffer": {
            "max_size": 20000,
            "auto_flush_interval": 60.0,
            "auto_flush_enabled": True
        },
        "filtering": {
            "min_level": "INFO"
        },
        "output": {
            "enabled_formats": ["json", "prometheus"],
            "file_rotation_size_mb": 200,
            "max_files": 20,
            "compress_rotated": True
        },
        "system_metrics": {
            "enabled": True,
            "collection_interval": 30.0,
            "collect_network": True
        },
        "performance": {
            "max_overhead_percent": 1.0,
            "disable_on_high_load": True,
            "performance_sampling_rate": 0.1
        },
        "security": {
            "scrub_sensitive_data": True,
            "audit_access": True,
            "max_retention_days": 90
        }
    },
    
    "debug": {
        "enabled": True,
        "buffer": {
            "max_size": 2000,
            "auto_flush_interval": 5.0
        },
        "filtering": {
            "min_level": "DEBUG"
        },
        "output": {
            "enabled_formats": ["json"],
            "file_rotation_size_mb": 10,
            "max_files": 3
        },
        "system_metrics": {
            "enabled": True,
            "collection_interval": 1.0,
            "collect_process_metrics": True
        },
        "performance": {
            "max_overhead_percent": 10.0,
            "performance_sampling_rate": 1.0
        }
    },
    
    "minimal": {
        "enabled": True,
        "buffer": {
            "max_size": 1000,
            "auto_flush_interval": 120.0
        },
        "filtering": {
            "min_level": "ERROR"
        },
        "output": {
            "enabled_formats": ["json"],
            "file_rotation_size_mb": 20,
            "max_files": 2
        },
        "system_metrics": {
            "enabled": False
        },
        "performance": {
            "max_overhead_percent": 0.5,
            "disable_on_high_load": True,
            "performance_sampling_rate": 0.01
        }
    },
    
    "high_performance": {
        "enabled": True,
        "buffer": {
            "max_size": 50000,
            "auto_flush_interval": 300.0
        },
        "filtering": {
            "min_level": "WARN"
        },
        "output": {
            "enabled_formats": ["json"],
            "file_rotation_size_mb": 500,
            "max_files": 50
        },
        "system_metrics": {
            "enabled": True,
            "collection_interval": 60.0
        },
        "performance": {
            "max_overhead_percent": 0.5,
            "disable_on_high_load": True,
            "high_load_cpu_threshold": 0.7,
            "performance_sampling_rate": 0.05
        }
    }
}

def load_telemetry_config(config_file: str = None, profile: str = None) -> TelemetryConfig:
    """
    Load telemetry configuration from file, profile, or environment.
    
    Args:
        config_file: Path to JSON configuration file
        profile: Name of predefined profile
        
    Returns:
        TelemetryConfig instance
    """
    # Check environment for profile override
    env_profile = os.getenv("DAWN_TELEMETRY_PROFILE")
    if env_profile and not profile:
        profile = env_profile
    
    return TelemetryConfig(config_file=config_file, profile=profile)

if __name__ == "__main__":
    # Demo configuration system
    print("🔧 DAWN Telemetry Configuration System")
    
    # Load default configuration
    config = TelemetryConfig()
    
    print(f"\n📊 Default Configuration Summary:")
    summary = config.get_runtime_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print(f"\n❌ Configuration Issues:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print(f"\n✅ Configuration is valid")
    
    # Show available profiles
    print(f"\n📋 Available Profiles:")
    for profile_name in TELEMETRY_PROFILES.keys():
        print(f"   • {profile_name}")
    
    # Demo profile loading
    prod_config = TelemetryConfig(profile="production")
    print(f"\n🏭 Production Profile Summary:")
    prod_summary = prod_config.get_runtime_summary()
    for key, value in prod_summary.items():
        print(f"   {key}: {value}")
    
    print(f"\n🔧 Telemetry configuration system ready")
