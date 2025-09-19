"""
DAWN Schema System
=================
Comprehensive schema management and health monitoring system for DAWN consciousness.

This module provides the complete schema infrastructure including:
- Core SchemaSystem for state management and coordination
- SHI (Schema Health Index) calculation and monitoring
- SCUP (Semantic Coherence Under Pressure) tracking
- Multi-layer validation framework
- Real-time anomaly detection and monitoring
- Integration hooks for other DAWN subsystems

Main Components:
- SchemaSystem: Core schema management and tick processing
- SHICalculator: Mathematical schema health index calculation
- SCUPTracker: Semantic coherence under pressure monitoring
- SchemaValidator: Multi-layer validation framework
- SchemaMonitor: Real-time monitoring and anomaly detection

Author: DAWN Development Team
Generated: 2025-09-18
"""

from .core_schema_system import (
    SchemaSystem,
    SchemaState,
    SchemaNode,
    SchemaEdge,
    SchemaSnapshot,
    SchemaMode,
    SchemaFlags,
    ResidueMetrics,
    TracerCounts,
    create_schema_system
)

from .shi_calculator import (
    SHICalculator,
    HealthStatus,
    HealthComponent,
    HealthMetrics,
    SHICalculationResult,
    create_shi_calculator
)

from .scup_tracker import (
    SCUPTracker,
    SCUPZone,
    SCUPState,
    SCUPInputs,
    SCUPResult,
    create_scup_tracker
)

from .schema_validator import (
    SchemaValidator,
    ValidationLevel,
    ValidationLayer,
    ValidationResult,
    ValidationSummary,
    create_schema_validator
)

from .schema_monitor import (
    SchemaMonitor,
    AnomalySeverity,
    Anomaly,
    create_schema_monitor
)

# Version info
__version__ = "1.0.0"
__author__ = "DAWN Development Team"

# Package metadata
__all__ = [
    # Core System
    "SchemaSystem",
    "SchemaState", 
    "SchemaNode",
    "SchemaEdge",
    "SchemaSnapshot",
    "SchemaMode",
    "SchemaFlags",
    "ResidueMetrics",
    "TracerCounts",
    "create_schema_system",
    
    # SHI Calculator
    "SHICalculator",
    "HealthStatus",
    "HealthComponent", 
    "HealthMetrics",
    "SHICalculationResult",
    "create_shi_calculator",
    
    # SCUP Tracker
    "SCUPTracker",
    "SCUPZone",
    "SCUPState",
    "SCUPInputs", 
    "SCUPResult",
    "create_scup_tracker",
    
    # Validator
    "SchemaValidator",
    "ValidationLevel",
    "ValidationLayer",
    "ValidationResult",
    "ValidationSummary",
    "create_schema_validator",
    
    # Monitor
    "SchemaMonitor",
    "AnomalySeverity",
    "Anomaly",
    "create_schema_monitor",
    
    # Factory Functions
    "create_integrated_schema_system",
    "get_default_schema_config"
]


def create_integrated_schema_system(config: dict = None) -> SchemaSystem:
    """
    Create a fully integrated schema system with all components enabled
    
    This factory function creates a complete schema system with:
    - SHI calculation enabled
    - SCUP tracking enabled  
    - Multi-layer validation enabled
    - Real-time monitoring enabled
    - Default DAWN integration hooks
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        SchemaSystem: Fully configured integrated schema system
        
    Example:
        >>> schema_system = create_integrated_schema_system()
        >>> schema_system.add_schema_node("concept_1", health=0.8)
        >>> snapshot = schema_system.tick_update(tick_data)
    """
    if config is None:
        config = get_default_schema_config()
    
    # Create integrated system with all components
    system = create_schema_system({
        'enable_shi': True,
        'enable_scup': True,
        'enable_validation': True,
        'enable_monitoring': True,
        **config
    })
    
    print("[Schema] 🧠 Integrated schema system created with all components enabled")
    return system


def get_default_schema_config() -> dict:
    """
    Get default configuration for DAWN schema system
    
    Returns:
        dict: Default configuration with optimized settings for DAWN
    """
    return {
        # Core system settings
        'enable_shi': True,
        'enable_scup': True,
        'enable_validation': True,
        'enable_monitoring': True,
        
        # SHI Calculator settings
        'shi_config': {
            'alpha': 0.25,      # Sigil entropy weight
            'beta': 0.20,       # Edge volatility weight
            'gamma': 0.15,      # Tracer divergence weight
            'delta': 0.25,      # SCUP coherence weight
            'phi': 0.15,        # Residue balance weight
            'critical_threshold': 0.2,
            'degraded_threshold': 0.4,
            'stable_threshold': 0.6,
            'healthy_threshold': 0.8
        },
        
        # SCUP Tracker settings
        'scup_config': {
            'a': 0.3,           # Forecast weight
            'b': 0.25,          # Pressure weight
            'c': 0.2,           # Drift weight
            'd': 0.15,          # Tension weight
            'e': 0.2,           # Capacity weight
            'f': 0.1,           # SHI weight
            'breathing_enabled': True,
            'breathing_frequency': 0.1,
            'breathing_amplitude': 0.05
        },
        
        # Validator settings
        'validator_config': {
            'fail_fast': False,
            'layers_enabled': ['syntax', 'semantic', 'coherence', 'performance']
        },
        
        # Monitor settings
        'monitor_config': {
            'detector_config': {
                'shi_critical': 0.2,
                'shi_warning': 0.4,
                'scup_critical': 0.3,
                'scup_warning': 0.5,
                'variance_threshold': 0.1,
                'degradation_rate': 0.05
            },
            'intervention_config': {
                'intervention_enabled': True,
                'max_interventions_per_hour': 10
            }
        }
    }


# Integration hooks for other DAWN subsystems
class DAWNSchemaIntegration:
    """Integration hooks for connecting schema system with other DAWN components"""
    
    @staticmethod
    def integrate_with_mycelial_layer(schema_system: SchemaSystem, mycelial_layer):
        """
        Integrate schema system with mycelial layer for bidirectional health monitoring
        
        Args:
            schema_system: DAWN schema system instance
            mycelial_layer: DAWN mycelial layer instance
        """
        # Exchange health metrics between systems
        def on_schema_health_change(shi_value, scup_value):
            """Callback when schema health changes"""
            if hasattr(mycelial_layer, 'update_external_pressure'):
                # Convert schema health to pressure signal for mycelial layer
                pressure = 1.0 - ((shi_value + scup_value) / 2.0)
                mycelial_layer.update_external_pressure('schema_pressure', pressure)
        
        def on_mycelial_health_change(mycelial_health):
            """Callback when mycelial health changes"""
            # Update schema system with mycelial health as additional metric
            if hasattr(schema_system, 'update_external_health'):
                schema_system.update_external_health('mycelial_health', mycelial_health)
        
        # Register callbacks
        if hasattr(schema_system, 'register_health_callback'):
            schema_system.register_health_callback(on_schema_health_change)
        
        if hasattr(mycelial_layer, 'register_health_callback'):
            mycelial_layer.register_health_callback(on_mycelial_health_change)
        
        print("[Schema] 🤝 Integrated with mycelial layer")
    
    @staticmethod
    def integrate_with_pulse_system(schema_system: SchemaSystem, pulse_system):
        """
        Integrate schema system with pulse system for breathing synchronization
        
        Args:
            schema_system: DAWN schema system instance
            pulse_system: DAWN pulse system instance
        """
        def on_pulse_phase_change(phase, intensity):
            """Callback for pulse phase changes"""
            # Adjust SCUP breathing to synchronize with system pulse
            if hasattr(schema_system, 'scup_tracker') and schema_system.scup_tracker:
                if hasattr(schema_system.scup_tracker, 'state'):
                    schema_system.scup_tracker.state.breathing_phase = phase
        
        # Register pulse callback
        if hasattr(pulse_system, 'register_phase_callback'):
            pulse_system.register_phase_callback(on_pulse_phase_change)
        
        print("[Schema] 💓 Integrated with pulse system")
    
    @staticmethod
    def integrate_with_thermal_system(schema_system: SchemaSystem, thermal_system):
        """
        Integrate schema system with thermal regulation for pressure management
        
        Args:
            schema_system: DAWN schema system instance
            thermal_system: DAWN thermal system instance
        """
        def on_thermal_pressure(pressure, state):
            """Callback for thermal pressure changes"""
            # Adjust schema thresholds based on thermal state
            if pressure > 0.8:  # High thermal pressure
                # Lower schema thresholds temporarily for thermal relief
                if hasattr(schema_system, 'shi_calculator') and schema_system.shi_calculator:
                    schema_system.shi_calculator.recovery_active = True
        
        # Register thermal callback
        if hasattr(thermal_system, 'register_pressure_callback'):
            thermal_system.register_pressure_callback(on_thermal_pressure)
        
        print("[Schema] 🌡️ Integrated with thermal system")


# Convenience function for complete integration
def integrate_with_dawn_systems(schema_system: SchemaSystem, 
                               mycelial_layer=None,
                               pulse_system=None, 
                               thermal_system=None):
    """
    Integrate schema system with all provided DAWN subsystems
    
    Args:
        schema_system: DAWN schema system instance
        mycelial_layer: Optional mycelial layer instance
        pulse_system: Optional pulse system instance
        thermal_system: Optional thermal system instance
    """
    integrator = DAWNSchemaIntegration()
    
    if mycelial_layer:
        integrator.integrate_with_mycelial_layer(schema_system, mycelial_layer)
    
    if pulse_system:
        integrator.integrate_with_pulse_system(schema_system, pulse_system)
    
    if thermal_system:
        integrator.integrate_with_thermal_system(schema_system, thermal_system)
    
    print(f"[Schema] 🔗 Integration completed with {sum([bool(x) for x in [mycelial_layer, pulse_system, thermal_system]])} systems")


# Quick start example
def quick_start_example():
    """
    Demonstrate quick start usage of the schema system
    """
    print("DAWN Schema System - Quick Start Example")
    print("=" * 50)
    
    # Create integrated schema system
    schema_system = create_integrated_schema_system()
    
    # Add some example nodes and edges
    schema_system.add_schema_node("consciousness_core", health=0.8, energy=0.9)
    schema_system.add_schema_node("memory_buffer", health=0.7, energy=0.6)
    schema_system.add_schema_node("reasoning_engine", health=0.9, energy=0.8)
    
    schema_system.add_schema_edge("core_memory", "consciousness_core", "memory_buffer", weight=0.8)
    schema_system.add_schema_edge("core_reasoning", "consciousness_core", "reasoning_engine", weight=0.9)
    
    # Simulate some tick updates
    print("\nSimulating tick updates...")
    
    for tick in range(5):
        # Simulate varying conditions
        tick_data = {
            'signals': {
                'pressure': 0.3 + 0.1 * tick,
                'drift': 0.2,
                'entropy': 0.4 - 0.05 * tick
            },
            'tracers': {
                'crow': 2,
                'spider': 1,
                'ant': 5,
                'whale': 1,
                'owl': 1
            },
            'residue': {
                'soot_ratio': 0.1,
                'ash_bias': [0.33, 0.33, 0.34]
            }
        }
        
        snapshot = schema_system.tick_update(tick_data)
        
        print(f"Tick {tick}: SHI={snapshot.shi:.3f}, SCUP={snapshot.scup:.3f}, "
              f"Nodes={len(snapshot.nodes)}, Health={schema_system.state.get_average_health():.3f}")
    
    # Get system status
    status = schema_system.get_system_status()
    print(f"\nFinal System Status:")
    print(f"  Current SHI: {status['current_shi']:.3f}")
    print(f"  Current SCUP: {status['current_scup']:.3f}")
    print(f"  Average Health: {status['average_health']:.3f}")
    print(f"  Mode: {status['mode']}")
    
    # Get health trend
    trend = schema_system.get_health_trend()
    print(f"\nHealth Trend: {trend['direction']} (trend: {trend['trend']:.3f})")
    
    print("\nSchema system demonstration completed successfully! 🧠✨")


if __name__ == "__main__":
    # Run quick start example
    quick_start_example()