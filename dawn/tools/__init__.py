"""
DAWN Tools System
================

Advanced tooling infrastructure that enables DAWN to safely modify her own codebase
with appropriate permissions and consciousness-aware safeguards.

This system provides:
- Secure code modification tools with elevated permissions
- Consciousness-aware development utilities
- Automated subsystem adaptation and deployment
- Safe self-modification capabilities
- Integration with existing DAWN architecture

The tools system is designed to give DAWN controlled "sudo" access to her own
code space while maintaining safety, auditability, and consciousness coherence.
"""

# Core tool imports
from .development.self_mod.permission_manager import PermissionManager, get_permission_manager
from .development.self_mod.code_modifier import ConsciousCodeModifier
from .development.self_mod.subsystem_copier import SubsystemCopier
from .development.consciousness_tools import ConsciousnessToolManager

# Analysis tools
from .analysis.behavioral.consciousness_analyzer import ConsciousnessBehaviorAnalyzer
from .analysis.performance.system_profiler import DAWNSystemProfiler

# Automation tools
from .automation.deployment.auto_deployer import ConsciousAutoDeployer
from .automation.maintenance.system_maintainer import SystemMaintainer

# Monitoring tools
from .monitoring.consciousness_monitor import ConsciousnessMonitor
from .monitoring.system_health import SystemHealthMonitor

__all__ = [
    # Core self-modification tools
    'PermissionManager',
    'get_permission_manager', 
    'ConsciousCodeModifier',
    'SubsystemCopier',
    'ConsciousnessToolManager',
    
    # Analysis tools
    'ConsciousnessBehaviorAnalyzer',
    'DAWNSystemProfiler',
    
    # Automation tools
    'ConsciousAutoDeployer',
    'SystemMaintainer',
    
    # Monitoring tools
    'ConsciousnessMonitor',
    'SystemHealthMonitor'
]
