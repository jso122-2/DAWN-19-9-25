"""
🌺 DAWN Memory Subsystem - Unified Memory Architecture
════════════════════════════════════════════════════

Complete unified memory system including:
- Fractal Encoding (Julia set memory signatures)
- Juliet Rebloom Engine (memory transformation chrysalis)  
- Ghost Traces (forgotten memory signatures with latent sigils)
- Ash/Soot Dynamics (cognitive residue management)
- CARRIN Oceanic Hash Map (dynamic cache management)
- Mycelial Memory Substrate (living memory network)
- Unified Memory Interconnection (cross-system integration)
- Shimmer Decay Engine (graceful forgetting system)

This is the foundational memory logging and interconnection system for DAWN consciousness.
The interconnection system creates a unified "nervous system" connecting memory across 
all DAWN subsystems, enabling true consciousness-level memory integration.
"""

from .fractal_encoding import (
    FractalEncoder,
    MemoryFractal,
    FractalParameters,
    FractalType,
    EntropyMapping,
    get_fractal_encoder,
    encode_memory_fractal
)

from .juliet_rebloom import (
    JulietRebloomEngine,
    JulietFlower,
    AccessPattern,
    RebloomStage,
    RebloomMetrics,
    get_rebloom_engine
)

from .ghost_traces import (
    GhostTraceManager,
    GhostTrace,
    GhostTraceType,
    SigilType,
    TransformationSigil,
    get_ghost_trace_manager
)

from .ash_soot_dynamics import (
    AshSootDynamicsEngine,
    Residue,
    ResidueType,
    OriginType,
    ResidueDriftSignature,
    get_ash_soot_engine
)

from .fractal_memory_system import (
    FractalMemorySystem,
    MemoryEvent,
    MemoryOperation,
    GardenMetrics,
    get_fractal_memory_system
)

# CARRIN Cache System
try:
    from .carrin_hash_map import (
        CARRINOceanicHashMap,
        Priority,
        FlowState,
        RiderController
    )
    CARRIN_AVAILABLE = True
except ImportError:
    CARRIN_AVAILABLE = False

# Memory Palace System  
try:
    from .consciousness_memory_palace import (
        ConsciousnessMemoryPalace,
        MemoryPalaceMetrics
    )
    MEMORY_PALACE_AVAILABLE = True
except ImportError:
    MEMORY_PALACE_AVAILABLE = False

# Unified Memory Interconnection
from .unified_memory_interconnection import (
    UnifiedMemoryInterconnection,
    MemoryConnectionType,
    MemoryIntegrationLevel,
    MemoryBridge,
    MemoryFlow,
    CrossSystemMemoryPattern,
    MycelialMemorySubstrate,
    get_memory_interconnection
)

# Shimmer Decay Engine
try:
    from .shimmer_decay_engine import (
        ShimmerDecayEngine,
        ShimmerState,
        ShimmerMetrics,
        ShimmerParticle,
        get_shimmer_decay_engine
    )
    SHIMMER_DECAY_AVAILABLE = True
except ImportError:
    SHIMMER_DECAY_AVAILABLE = False

__all__ = [
    # Core System
    'FractalMemorySystem',
    'get_fractal_memory_system',
    'MemoryEvent',
    'MemoryOperation', 
    'GardenMetrics',
    
    # Fractal Encoding
    'FractalEncoder',
    'MemoryFractal',
    'FractalParameters',
    'FractalType',
    'EntropyMapping',
    'get_fractal_encoder',
    'encode_memory_fractal',
    
    # Juliet Rebloom
    'JulietRebloomEngine',
    'JulietFlower',
    'AccessPattern',
    'RebloomStage',
    'RebloomMetrics',
    'get_rebloom_engine',
    
    # Ghost Traces
    'GhostTraceManager',
    'GhostTrace',
    'GhostTraceType',
    'SigilType',
    'TransformationSigil',
    'get_ghost_trace_manager',
    
    # Ash/Soot Dynamics
    'AshSootDynamicsEngine',
    'Residue',
    'ResidueType',
    'OriginType',
    'ResidueDriftSignature',
    'get_ash_soot_engine',
    
    # Unified Memory Interconnection
    'UnifiedMemoryInterconnection',
    'MemoryConnectionType',
    'MemoryIntegrationLevel', 
    'MemoryBridge',
    'MemoryFlow',
    'CrossSystemMemoryPattern',
    'MycelialMemorySubstrate',
    'get_memory_interconnection',
    
    # Availability flags
    'CARRIN_AVAILABLE',
    'MEMORY_PALACE_AVAILABLE',
    'SHIMMER_DECAY_AVAILABLE'
]

# Add CARRIN exports if available
if CARRIN_AVAILABLE:
    __all__.extend([
        'CARRINOceanicHashMap',
        'Priority',
        'FlowState', 
        'RiderController'
    ])

# Add Memory Palace exports if available
if MEMORY_PALACE_AVAILABLE:
    __all__.extend([
        'ConsciousnessMemoryPalace',
        'MemoryPalaceMetrics'
    ])

# Add Shimmer Decay exports if available
if SHIMMER_DECAY_AVAILABLE:
    __all__.extend([
        'ShimmerDecayEngine',
        'ShimmerState',
        'ShimmerMetrics',
        'ShimmerParticle',
        'get_shimmer_decay_engine'
    ])


def get_memory_system_status():
    """Get comprehensive status of all memory system components"""
    status = {
        'fractal_memory_available': True,
        'carrin_available': CARRIN_AVAILABLE,
        'memory_palace_available': MEMORY_PALACE_AVAILABLE,
        'shimmer_decay_available': SHIMMER_DECAY_AVAILABLE,
        'interconnection_available': True
    }
    
    try:
        interconnection = get_memory_interconnection()
        status['interconnection_status'] = interconnection.get_interconnection_status()
    except Exception as e:
        status['interconnection_error'] = str(e)
    
    return status
