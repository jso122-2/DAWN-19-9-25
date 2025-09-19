"""
DAWN Foundation Systems
======================

Core foundational components providing:
- Base interfaces for all DAWN modules
- Event system for inter-module communication
- Module registry and lifecycle management
- Plugin architecture support
"""

from .base_engine import BaseEngine, EngineState
from .base_module import BaseModule, ModuleCapability
from .event_system import EventSystem, Event, EventBus
from .registry import ModuleRegistry, CapabilityRegistry
from .lifecycle import ModuleLifecycle, LifecycleState
from .state import get_state, set_state, clamp, label_for, update_unity_delta, update_awareness_delta, get_state_summary, reset_state
from .snapshot import snapshot, restore

__all__ = [
    'BaseEngine', 'EngineState',
    'BaseModule', 'ModuleCapability', 
    'EventSystem', 'Event', 'EventBus',
    'ModuleRegistry', 'CapabilityRegistry',
    'ModuleLifecycle', 'LifecycleState',
    'get_state', 'set_state', 'clamp', 'label_for', 'update_unity_delta', 'update_awareness_delta', 'get_state_summary', 'reset_state',
    'snapshot', 'restore'
]
