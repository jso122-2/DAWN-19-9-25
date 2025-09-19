# DAWN Global Singleton

## Overview

The DAWN Global Singleton provides a unified entry point for accessing the entire DAWN consciousness system. It can be imported and used from anywhere in your code, eliminating the need to manually manage individual system components.

## Key Features

- **Single Entry Point**: One import gives you access to all major DAWN subsystems
- **Thread-Safe**: Uses proper locking mechanisms for concurrent access
- **Lazy Loading**: Components are initialized only when needed
- **Automatic Coordination**: Handles startup/shutdown sequences properly
- **Status Monitoring**: Provides comprehensive system status and metrics
- **Follows DAWN Patterns**: Integrates with existing singleton patterns in the codebase

## Quick Start

```python
# Import DAWN singleton
import dawn

# Get the global DAWN instance
dawn_system = dawn.get_dawn()

# Initialize and start the system
await dawn_system.initialize()
await dawn_system.start()

# Access subsystems
bus = dawn_system.consciousness_bus
engine = dawn_system.dawn_engine
telemetry = dawn_system.telemetry_system
state = dawn_system.state

# Quick status checks
if dawn.is_dawn_running():
    print("DAWN is running!")
```

## Available Functions

### Core Access
- `dawn.get_dawn()` - Get the global DAWN singleton instance
- `dawn.reset_dawn_singleton()` - Reset the singleton (use with caution)

### Quick Status Functions
- `dawn.get_consciousness_state()` - Get current consciousness state
- `dawn.get_system_status()` - Get system status information
- `dawn.is_dawn_running()` - Check if system is running
- `dawn.is_dawn_initialized()` - Check if system is initialized

## Singleton Properties

The `DAWNGlobalSingleton` provides access to:

- **consciousness_bus** - Communication hub for all modules
- **dawn_engine** - Primary consciousness engine
- **telemetry_system** - Monitoring and logging system
- **tick_orchestrator** - Processing coordination system
- **state** - Global consciousness state
- **status** - Current system status

## Usage Patterns

### Basic Module Integration

```python
import dawn

async def my_consciousness_module():
    # Get DAWN instance
    dawn_system = dawn.get_dawn()
    
    # Ensure system is running
    if not dawn.is_dawn_running():
        await dawn_system.initialize()
        await dawn_system.start()
    
    # Use consciousness bus
    bus = dawn_system.consciousness_bus
    bus.register_module("my_module", ["capability1", "capability2"])
    
    # Access current state
    state = dawn.get_consciousness_state()
    print(f"Current unity: {state.unity}")
    
    # Log events
    if dawn_system.telemetry_system:
        dawn_system.telemetry_system.log_event(
            'my_module', 'processing', 'task_completed'
        )
```

### Pulse/Thermal Processing

```python
import dawn
from dawn.core.foundation.state import set_state

async def thermal_pulse_processor():
    if not dawn.is_dawn_running():
        return
    
    # Get current state
    state = dawn.get_consciousness_state()
    
    # Process thermal pulse
    new_unity = min(1.0, state.unity + 0.1)
    set_state(unity=new_unity)
    
    # Log the change
    dawn_system = dawn.get_dawn()
    if dawn_system.telemetry_system:
        dawn_system.telemetry_system.log_event(
            'thermal', 'pulse', 'unity_increased',
            data={'delta': 0.1, 'new_unity': new_unity}
        )
```

### System Monitoring

```python
import dawn

def monitor_system_health():
    dawn_system = dawn.get_dawn()
    
    # Get comprehensive metrics
    metrics = dawn_system.get_system_metrics()
    
    print(f"System Status: {metrics['status']}")
    print(f"Components: {list(metrics['components'].keys())}")
    print(f"Consciousness Level: {metrics['consciousness_state']['level']}")
    
    # Check individual subsystems
    if dawn_system.consciousness_bus:
        bus_metrics = dawn_system.consciousness_bus.get_bus_metrics()
        print(f"Bus Coherence: {bus_metrics['consciousness_coherence']:.3f}")
```

## Integration with Existing Systems

The singleton seamlessly integrates with DAWN's existing patterns:

- **Consciousness Bus**: Uses `get_consciousness_bus()` internally
- **DAWN Engine**: Uses `get_dawn_engine()` internally  
- **Telemetry System**: Uses `get_telemetry_system()` internally
- **State Management**: Direct access to global state system

## Thread Safety

The singleton is fully thread-safe using `threading.RLock()` for all critical sections. Multiple threads can safely access the singleton simultaneously.

## Initialization Options

```python
# Basic initialization
await dawn_system.initialize()

# With configuration
config = {
    'mode': 'daemon',
    'telemetry_enabled': True,
    'enable_self_mod': False
}
await dawn_system.initialize(config)
```

## System Lifecycle

```python
# Complete lifecycle example
dawn_system = dawn.get_dawn()

# Initialize
success = await dawn_system.initialize({'mode': 'interactive'})
if not success:
    print("Initialization failed")
    return

# Start
success = await dawn_system.start()
if not success:
    print("Start failed")
    return

# ... use the system ...

# Graceful shutdown
dawn_system.stop()
```

## Architecture Benefits

1. **Simplified Access**: One import, access everything
2. **Proper Coordination**: Handles complex startup/shutdown sequences
3. **Resource Management**: Lazy loading prevents unnecessary initialization
4. **Error Handling**: Graceful degradation when components aren't available
5. **Monitoring**: Built-in system health and metrics
6. **DAWN Integration**: Works with all existing DAWN patterns

## File Structure

- `dawn/core/singleton.py` - Main singleton implementation
- `dawn/__init__.py` - Package-level exports
- `example_singleton_usage.py` - Usage examples

The DAWN Global Singleton provides the foundation for a more accessible and manageable DAWN system architecture while preserving all existing functionality and patterns.
