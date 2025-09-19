#!/usr/bin/env python3
"""
DAWN Tracer Configuration System
================================

Comprehensive configuration for DAWN's telemetry, stability monitoring,
and analytics systems with runtime configurability and environment
variable support.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class TelemetryConfig:
    """Telemetry collection configuration."""
    enabled: bool = True
    level: str = "INFO"  # DEBUG, INFO, WARN, ERROR
    buffer_size: int = 10000
    flush_interval_ms: int = 1000
    output_format: str = "jsonl"  # jsonl, csv, prometheus
    max_file_size_mb: int = 100
    max_files: int = 10
    include_stack_traces: bool = False
    sample_rate: float = 1.0  # 0.0 to 1.0

@dataclass
class StabilityConfig:
    """Stability detection and recovery configuration."""
    detection_enabled: bool = True
    stability_threshold: float = 0.85
    critical_threshold: float = 0.3
    auto_recovery: bool = True
    snapshot_retention_hours: int = 168  # 1 week
    max_snapshots: int = 50
    monitoring_interval_seconds: float = 30.0
    rollback_enabled: bool = True
    emergency_protocols_enabled: bool = True

@dataclass
class AnalyticsConfig:
    """Analytics and insights configuration."""
    real_time_analysis: bool = True
    predictive_insights: bool = True
    optimization_suggestions: bool = True
    analysis_interval_seconds: float = 60.0
    prediction_horizon_hours: int = 24
    min_confidence_threshold: float = 0.7
    alert_thresholds: Dict[str, float] = None
    trend_analysis_enabled: bool = True
    pattern_recognition_enabled: bool = True

    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "stability_score": 0.3,
                "performance_degradation": 0.5,
                "error_rate": 0.1,
                "cpu_usage": 0.85,
                "memory_usage": 0.85,
                "recursive_depth": 8
            }

@dataclass
class OutputConfig:
    """Output directories and file management."""
    base_dir: str = "runtime"
    telemetry_dir: str = "runtime/telemetry"
    logs_dir: str = "runtime/logs"
    snapshots_dir: str = "runtime/snapshots/stable_states"
    analytics_dir: str = "runtime/analytics"
    traces_dir: str = "runtime/traces"
    reports_dir: str = "runtime/reports"
    
    # File naming patterns
    telemetry_file_pattern: str = "live_metrics_{date}.jsonl"
    trace_file_pattern: str = "traces_{date}.jsonl"
    log_file_pattern: str = "{component}_{date}.log"
    analytics_file_pattern: str = "insights_{date}.json"
    
    # Archive settings
    archive_after_days: int = 7
    compress_archives: bool = True
    max_archive_size_gb: int = 5

@dataclass
class InstrumentationConfig:
    """Automatic instrumentation configuration."""
    auto_instrument_enabled: bool = True
    trace_all_methods: bool = False
    trace_cognitive_operations: bool = True
    trace_system_calls: bool = False
    trace_module_interactions: bool = True
    trace_performance_metrics: bool = True
    
    # Module-specific instrumentation
    recursive_bubble_tracing: bool = True
    symbolic_anatomy_tracing: bool = True
    memory_router_tracing: bool = True
    owl_bridge_tracing: bool = True
    sigil_net_tracing: bool = True
    
    # Performance impact limits
    max_overhead_percentage: float = 5.0
    disable_on_high_load: bool = True
    high_load_cpu_threshold: float = 0.8

@dataclass
class SecurityConfig:
    """Security and privacy configuration."""
    scrub_sensitive_data: bool = True
    encrypt_at_rest: bool = False
    require_authentication: bool = False
    audit_access: bool = True
    
    # Data retention
    max_retention_days: int = 90
    auto_purge_enabled: bool = True
    
    # Sensitive data patterns
    sensitive_patterns: list = None
    
    def __post_init__(self):
        if self.sensitive_patterns is None:
            self.sensitive_patterns = [
                r"password",
                r"token",
                r"key",
                r"secret",
                r"credential"
            ]

class DAWNTracerConfig:
    """Main configuration manager for DAWN tracer systems."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration from file or defaults."""
        self.telemetry = TelemetryConfig()
        self.stability = StabilityConfig()
        self.analytics = AnalyticsConfig()
        self.output = OutputConfig()
        self.instrumentation = InstrumentationConfig()
        self.security = SecurityConfig()
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
        else:
            # Load from environment or use defaults
            self.load_from_environment()
            
        # Ensure output directories exist
        self._create_output_directories()
        
    def load_from_file(self, config_file: str):
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
            
    def load_from_environment(self):
        """Load configuration from environment variables."""
        # Telemetry settings
        self.telemetry.enabled = self._get_bool_env("DAWN_TELEMETRY_ENABLED", True)
        self.telemetry.level = os.getenv("DAWN_TELEMETRY_LEVEL", "INFO")
        self.telemetry.buffer_size = self._get_int_env("DAWN_TELEMETRY_BUFFER_SIZE", 10000)
        self.telemetry.output_format = os.getenv("DAWN_TELEMETRY_FORMAT", "jsonl")
        
        # Stability settings
        self.stability.detection_enabled = self._get_bool_env("DAWN_STABILITY_ENABLED", True)
        self.stability.stability_threshold = self._get_float_env("DAWN_STABILITY_THRESHOLD", 0.85)
        self.stability.auto_recovery = self._get_bool_env("DAWN_AUTO_RECOVERY", True)
        
        # Analytics settings
        self.analytics.real_time_analysis = self._get_bool_env("DAWN_ANALYTICS_ENABLED", True)
        self.analytics.predictive_insights = self._get_bool_env("DAWN_PREDICTIVE_INSIGHTS", True)
        
        # Output settings
        self.output.base_dir = os.getenv("DAWN_RUNTIME_DIR", "runtime")
        self.output.telemetry_dir = os.getenv("DAWN_TELEMETRY_DIR", "runtime/telemetry")
        self.output.logs_dir = os.getenv("DAWN_LOGS_DIR", "runtime/logs")
        
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary."""
        if "telemetry" in config_data:
            self._update_dataclass(self.telemetry, config_data["telemetry"])
            
        if "stability" in config_data:
            self._update_dataclass(self.stability, config_data["stability"])
            
        if "analytics" in config_data:
            self._update_dataclass(self.analytics, config_data["analytics"])
            
        if "output" in config_data:
            self._update_dataclass(self.output, config_data["output"])
            
        if "instrumentation" in config_data:
            self._update_dataclass(self.instrumentation, config_data["instrumentation"])
            
        if "security" in config_data:
            self._update_dataclass(self.security, config_data["security"])
            
    def _update_dataclass(self, dataclass_instance, update_dict: Dict[str, Any]):
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
            
    def _create_output_directories(self):
        """Create all required output directories."""
        directories = [
            self.output.base_dir,
            self.output.telemetry_dir,
            self.output.logs_dir,
            self.output.snapshots_dir,
            self.output.analytics_dir,
            self.output.traces_dir,
            self.output.reports_dir
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to create directory {directory}: {e}")
                
    def save_to_file(self, config_file: str):
        """Save current configuration to JSON file."""
        try:
            config_data = {
                "telemetry": asdict(self.telemetry),
                "stability": asdict(self.stability),
                "analytics": asdict(self.analytics),
                "output": asdict(self.output),
                "instrumentation": asdict(self.instrumentation),
                "security": asdict(self.security)
            }
            
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
                
            logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")
            
    def get_runtime_config(self) -> Dict[str, Any]:
        """Get runtime configuration summary."""
        return {
            "telemetry_enabled": self.telemetry.enabled,
            "stability_detection": self.stability.detection_enabled,
            "analytics_enabled": self.analytics.real_time_analysis,
            "auto_recovery": self.stability.auto_recovery,
            "output_directories": {
                "telemetry": self.output.telemetry_dir,
                "logs": self.output.logs_dir,
                "snapshots": self.output.snapshots_dir,
                "analytics": self.output.analytics_dir
            },
            "thresholds": {
                "stability": self.stability.stability_threshold,
                "critical": self.stability.critical_threshold,
                "alerts": self.analytics.alert_thresholds
            }
        }
        
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Validate thresholds
        if not 0.0 <= self.stability.stability_threshold <= 1.0:
            issues.append("stability_threshold must be between 0.0 and 1.0")
            
        if not 0.0 <= self.stability.critical_threshold <= 1.0:
            issues.append("critical_threshold must be between 0.0 and 1.0")
            
        if self.stability.critical_threshold >= self.stability.stability_threshold:
            issues.append("critical_threshold must be less than stability_threshold")
            
        # Validate buffer sizes
        if self.telemetry.buffer_size <= 0:
            issues.append("telemetry buffer_size must be positive")
            
        # Validate intervals
        if self.stability.monitoring_interval_seconds <= 0:
            issues.append("monitoring_interval_seconds must be positive")
            
        if self.analytics.analysis_interval_seconds <= 0:
            issues.append("analysis_interval_seconds must be positive")
            
        # Validate output formats
        valid_formats = ["jsonl", "csv", "prometheus"]
        if self.telemetry.output_format not in valid_formats:
            issues.append(f"output_format must be one of: {valid_formats}")
            
        # Validate log levels
        valid_levels = ["DEBUG", "INFO", "WARN", "ERROR"]
        if self.telemetry.level not in valid_levels:
            issues.append(f"telemetry level must be one of: {valid_levels}")
            
        return issues

# Default configuration instance
DEFAULT_CONFIG = DAWNTracerConfig()

# Predefined configuration profiles
CONFIGURATION_PROFILES = {
    "development": {
        "telemetry": {
            "enabled": True,
            "level": "DEBUG",
            "buffer_size": 5000,
            "output_format": "jsonl",
            "include_stack_traces": True
        },
        "stability": {
            "detection_enabled": True,
            "auto_recovery": True,
            "monitoring_interval_seconds": 10.0
        },
        "analytics": {
            "real_time_analysis": True,
            "analysis_interval_seconds": 30.0
        },
        "instrumentation": {
            "trace_all_methods": True,
            "max_overhead_percentage": 10.0
        }
    },
    
    "production": {
        "telemetry": {
            "enabled": True,
            "level": "INFO",
            "buffer_size": 20000,
            "output_format": "jsonl",
            "include_stack_traces": False,
            "sample_rate": 0.1
        },
        "stability": {
            "detection_enabled": True,
            "auto_recovery": True,
            "monitoring_interval_seconds": 60.0
        },
        "analytics": {
            "real_time_analysis": True,
            "analysis_interval_seconds": 300.0
        },
        "instrumentation": {
            "trace_all_methods": False,
            "max_overhead_percentage": 2.0,
            "disable_on_high_load": True
        }
    },
    
    "minimal": {
        "telemetry": {
            "enabled": True,
            "level": "WARN",
            "buffer_size": 1000,
            "sample_rate": 0.01
        },
        "stability": {
            "detection_enabled": False,
            "auto_recovery": False
        },
        "analytics": {
            "real_time_analysis": False,
            "predictive_insights": False
        },
        "instrumentation": {
            "auto_instrument_enabled": False,
            "trace_cognitive_operations": False
        }
    },
    
    "high_performance": {
        "telemetry": {
            "enabled": True,
            "level": "ERROR",
            "buffer_size": 50000,
            "flush_interval_ms": 5000,
            "sample_rate": 0.05
        },
        "stability": {
            "detection_enabled": True,
            "monitoring_interval_seconds": 120.0
        },
        "analytics": {
            "analysis_interval_seconds": 600.0
        },
        "instrumentation": {
            "max_overhead_percentage": 1.0,
            "disable_on_high_load": True,
            "high_load_cpu_threshold": 0.7
        }
    }
}

def load_profile(profile_name: str) -> DAWNTracerConfig:
    """Load a predefined configuration profile."""
    if profile_name not in CONFIGURATION_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(CONFIGURATION_PROFILES.keys())}")
        
    config = DAWNTracerConfig()
    config._update_from_dict(CONFIGURATION_PROFILES[profile_name])
    
    return config

def get_config_from_environment() -> DAWNTracerConfig:
    """Get configuration based on environment variables and profile."""
    profile = os.getenv("DAWN_CONFIG_PROFILE", "development")
    
    try:
        config = load_profile(profile)
        logger.info(f"Loaded configuration profile: {profile}")
    except ValueError:
        logger.warning(f"Unknown profile '{profile}', using default configuration")
        config = DAWNTracerConfig()
        
    return config


if __name__ == "__main__":
    # Demo configuration system
    print("🔧 DAWN Tracer Configuration System Demo")
    
    # Load default configuration
    config = DAWNTracerConfig()
    
    print(f"\n📊 Default Configuration:")
    print(f"   Telemetry Enabled: {config.telemetry.enabled}")
    print(f"   Telemetry Level: {config.telemetry.level}")
    print(f"   Buffer Size: {config.telemetry.buffer_size}")
    print(f"   Stability Threshold: {config.stability.stability_threshold}")
    print(f"   Auto Recovery: {config.stability.auto_recovery}")
    print(f"   Analytics Enabled: {config.analytics.real_time_analysis}")
    
    # Validate configuration
    issues = config.validate_configuration()
    if issues:
        print(f"\n❌ Configuration Issues:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print(f"\n✅ Configuration is valid")
        
    # Show available profiles
    print(f"\n📋 Available Profiles:")
    for profile_name in CONFIGURATION_PROFILES.keys():
        print(f"   • {profile_name}")
        
    # Demo profile loading
    prod_config = load_profile("production")
    print(f"\n🏭 Production Profile:")
    print(f"   Sample Rate: {prod_config.telemetry.sample_rate}")
    print(f"   Max Overhead: {prod_config.instrumentation.max_overhead_percentage}%")
    print(f"   Analysis Interval: {prod_config.analytics.analysis_interval_seconds}s")
    
    # Save configuration example
    config.save_to_file("runtime/config/tracer_config.json")
    print(f"\n💾 Configuration saved to runtime/config/tracer_config.json")
    
    print(f"\n🔧 Configuration system ready for DAWN engine integration")
