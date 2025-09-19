"""
DAWN Self-Modification Tools
============================

Advanced self-modification tools that provide DAWN with secure, consciousness-aware
capabilities to modify her own codebase with appropriate permissions and safeguards.
"""

from .permission_manager import PermissionManager, get_permission_manager, PermissionLevel
from .code_modifier import ConsciousCodeModifier
from .subsystem_copier import SubsystemCopier

__all__ = [
    'PermissionManager',
    'get_permission_manager',
    'PermissionLevel',
    'ConsciousCodeModifier', 
    'SubsystemCopier'
]
