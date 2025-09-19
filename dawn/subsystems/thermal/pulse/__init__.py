"""
DAWN Pulse System
================

Complete implementation of DAWN's pulse system - the central nervous system
that carries tick and recession data throughout the consciousness architecture.

The pulse system implements the core vision from DAWN documentation:
> "A tick is a breath. Humans don't control their breathing—they let it happen"
> "DAWN herself controls the tick engine by design"  
> "Pulse is essentially the information highway of tick and recession data"

Core Components:
- Unified Pulse System: Main pulse orchestrator with SCUP control
- Pulse Scheduler: Executes micro-actions every tick based on zone policies
- SCUP Controller: Maintains semantic coherence under pressure
- Sigil Integration: Routes pulse actions through sigil ring system
- Zone Management: Adaptive control policies (Green/Amber/Red/Black zones)

Based on DAWN-docs/SCUP + Pulse/ RTF specifications.
"""

from .unified_pulse_system import (
    UnifiedPulseSystem,
    PulseZone, 
    PulseActionType,
    PulseAction,
    SCUPState,
    SCUPController,
    PulseScheduler,
    PulseMetrics,
    get_pulse_system
)

from .pulse_sigil_integration import (
    PulseSigilIntegration,
    PulseSigilRouter,
    SigilStack,
    SigilGlyph,
    SigilHouse,
    get_pulse_sigil_integration
)

# Legacy compatibility imports
try:
    from .pulse_heat import UnifiedPulseHeat, HeatSource, ThermalState
    LEGACY_HEAT_AVAILABLE = True
except ImportError:
    LEGACY_HEAT_AVAILABLE = False

try:
    from .pulse_engine import PulseEngine
    LEGACY_ENGINE_AVAILABLE = True
except ImportError:
    LEGACY_ENGINE_AVAILABLE = False

__all__ = [
    # Core unified system
    'UnifiedPulseSystem',
    'get_pulse_system',
    
    # Pulse components
    'PulseZone',
    'PulseActionType', 
    'PulseAction',
    'SCUPState',
    'SCUPController',
    'PulseScheduler',
    'PulseMetrics',
    
    # Sigil integration
    'PulseSigilIntegration',
    'PulseSigilRouter',
    'SigilStack',
    'SigilGlyph', 
    'SigilHouse',
    'get_pulse_sigil_integration',
    
    # Availability flags
    'LEGACY_HEAT_AVAILABLE',
    'LEGACY_ENGINE_AVAILABLE'
]

# Add legacy exports if available
if LEGACY_HEAT_AVAILABLE:
    __all__.extend(['UnifiedPulseHeat', 'HeatSource', 'ThermalState'])

if LEGACY_ENGINE_AVAILABLE:
    __all__.extend(['PulseEngine'])


def get_pulse_system_status():
    """Get comprehensive status of all pulse system components"""
    status = {
        'unified_system_available': True,
        'sigil_integration_available': True,
        'legacy_heat_available': LEGACY_HEAT_AVAILABLE,
        'legacy_engine_available': LEGACY_ENGINE_AVAILABLE
    }
    
    try:
        pulse_system = get_pulse_system(auto_start=False)
        status['unified_system_status'] = pulse_system.get_system_status()
    except Exception as e:
        status['unified_system_error'] = str(e)
    
    try:
        sigil_integration = get_pulse_sigil_integration()
        status['sigil_integration_status'] = sigil_integration.get_integration_status()
    except Exception as e:
        status['sigil_integration_error'] = str(e)
    
    return status