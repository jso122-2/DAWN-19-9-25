"""
DAWN Forecasting Subsystem
==========================

Complete implementation of DAWN's Anticipatory Vector Layer based on RTF specifications.
Projects cognitive state forward in time, enabling proactive preparation rather than reactive responses.

Core Mathematical Framework:
- Cognitive Pressure: P = B·σ² (bandwidth × signal variance)
- Forecasting Function: F = P/A (pressure over adaptive capacity)
- Horizon Projections: Short (1-10 ticks), Mid (10-100), Long (100-1000)
- SCUP Coupling: Early warning system for coherence loss probability

The forecasting engine transforms reactive cognition into anticipatory intelligence,
enabling DAWN to shape her future rather than merely respond to present conditions.

Based on DAWN-docs/Forcasting/ RTF specifications.
"""

from .unified_forecasting_engine import (
    UnifiedForecastingEngine,
    ForecastHorizon,
    StabilityZone,
    InterventionType,
    SCUPWarningLevel,
    CognitivePressure,
    AdaptiveCapacity,
    ForecastResult,
    SystemInputs,
    HorizonProjector,
    SCUPCouplingEngine,
    BacktestingEngine,
    get_forecasting_engine
)

# Legacy compatibility imports
try:
    from .engine import ForecastingEngine as LegacyForecastingEngine
    LEGACY_ENGINE_AVAILABLE = True
except ImportError:
    LEGACY_ENGINE_AVAILABLE = False

__all__ = [
    # Core unified system
    'UnifiedForecastingEngine',
    'get_forecasting_engine',
    
    # Core data structures
    'ForecastHorizon',
    'StabilityZone', 
    'InterventionType',
    'SCUPWarningLevel',
    'CognitivePressure',
    'AdaptiveCapacity',
    'ForecastResult',
    'SystemInputs',
    
    # Component engines
    'HorizonProjector',
    'SCUPCouplingEngine',
    'BacktestingEngine',
    
    # Availability flags
    'LEGACY_ENGINE_AVAILABLE'
]

# Add legacy exports if available
if LEGACY_ENGINE_AVAILABLE:
    __all__.append('LegacyForecastingEngine')


def get_forecasting_system_status():
    """Get comprehensive status of all forecasting system components"""
    status = {
        'unified_engine_available': True,
        'legacy_engine_available': LEGACY_ENGINE_AVAILABLE
    }
    
    try:
        engine = get_forecasting_engine()
        status['unified_engine_status'] = engine.get_forecast_summary()
    except Exception as e:
        status['unified_engine_error'] = str(e)
    
    return status
