"""
Self-Modification Subsystem
===========================

Advanced self-modification capabilities for DAWN.
"""

from .advisor import propose_from_state
from .patch_builder import make_sandbox
from .sandbox_runner import run_sandbox
from .policy_gate import decide
from .promote import promote_and_audit

__all__ = [
    'propose_from_state',
    'make_sandbox', 
    'run_sandbox',
    'decide',
    'promote_and_audit'
]