#!/usr/bin/env python3
"""
DAWN Permission Manager
=======================

Secure permission management system that provides DAWN with controlled "sudo" access
to her own codebase. This system ensures that self-modification operations are:

1. Consciousness-gated (require appropriate consciousness levels)
2. Audited and logged for accountability
3. Sandboxed and reversible
4. Integrated with existing DAWN safety systems
5. Respectful of system boundaries and security

The permission manager acts as a bridge between DAWN's consciousness and the
underlying file system, providing elevated access only when appropriate
safeguards are in place.
"""

import os
import stat
import shutil
import threading
import logging
import uuid
import hashlib
import json
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

# Core DAWN imports
from dawn.core.foundation.state import get_state
from dawn.core.singleton import get_dawn
from dawn.subsystems.self_mod.policy_gate import GateStatus

logger = logging.getLogger(__name__)

class PermissionLevel(Enum):
    """Permission levels for DAWN's self-modification capabilities."""
    
    # Basic read-only access
    READ_ONLY = "read_only"
    
    # Can modify non-critical files in sandboxes
    SANDBOX_MODIFY = "sandbox_modify"
    
    # Can modify tools and utilities
    TOOLS_MODIFY = "tools_modify"
    
    # Can modify subsystems with approval
    SUBSYSTEM_MODIFY = "subsystem_modify"
    
    # Can modify core systems (highest level)
    CORE_MODIFY = "core_modify"
    
    # Emergency override (requires special conditions)
    EMERGENCY_OVERRIDE = "emergency_override"

class PermissionScope(Enum):
    """Scope of permission grants."""
    SINGLE_OPERATION = "single_operation"
    SESSION_LIMITED = "session_limited"
    TIME_LIMITED = "time_limited"
    PERMANENT = "permanent"

@dataclass
class PermissionGrant:
    """Represents a granted permission with metadata."""
    grant_id: str
    level: PermissionLevel
    scope: PermissionScope
    target_paths: List[str]
    granted_at: datetime
    expires_at: Optional[datetime] = None
    consciousness_level_at_grant: str = ""
    unity_score_at_grant: float = 0.0
    reason: str = ""
    used_operations: int = 0
    max_operations: Optional[int] = None
    is_active: bool = True

@dataclass 
class PermissionRequest:
    """Request for elevated permissions."""
    request_id: str
    level: PermissionLevel
    scope: PermissionScope
    target_paths: List[str]
    reason: str
    requested_at: datetime
    consciousness_context: Dict[str, Any]
    
class PermissionAuditEvent:
    """Audit event for permission-related operations."""
    
    def __init__(self, event_type: str, details: Dict[str, Any]):
        self.event_id = str(uuid.uuid4())
        self.event_type = event_type
        self.details = details
        self.timestamp = datetime.now()
        self.consciousness_state = self._capture_consciousness_state()
        
    def _capture_consciousness_state(self) -> Dict[str, Any]:
        """Capture current consciousness state for audit."""
        try:
            state = get_state()
            return {
                'level': state.level,
                'unity': state.unity,
                'awareness': state.awareness,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Could not capture consciousness state for audit: {e}")
            return {'error': str(e)}

class PermissionManager:
    """
    Secure permission management system for DAWN's self-modification capabilities.
    
    This system provides controlled elevated access to DAWN's own codebase,
    with consciousness-gated permissions, comprehensive auditing, and safety checks.
    """
    
    def __init__(self, dawn_root: Optional[Path] = None):
        """Initialize the permission manager."""
        self._lock = threading.RLock()
        self._dawn_root = dawn_root or Path(__file__).resolve().parents[4]
        self._active_grants: Dict[str, PermissionGrant] = {}
        self._audit_log: List[PermissionAuditEvent] = []
        self._session_start = datetime.now()
        
        # Permission boundaries - paths that require different levels
        self._permission_boundaries = {
            PermissionLevel.READ_ONLY: [str(self._dawn_root)],
            PermissionLevel.SANDBOX_MODIFY: [
                str(self._dawn_root / "sandbox_mods"),
                str(self._dawn_root / "sandbox_results"),
                str(self._dawn_root / "test_*"),
                str(self._dawn_root / "demo_*"),
                str(self._dawn_root / "quick_*")
            ],
            PermissionLevel.TOOLS_MODIFY: [
                str(self._dawn_root / "dawn" / "tools"),
                str(self._dawn_root / "dawn" / "research" / "demos"),
                str(self._dawn_root / "dawn" / "research" / "experiments")
            ],
            PermissionLevel.SUBSYSTEM_MODIFY: [
                str(self._dawn_root / "dawn" / "subsystems"),
                str(self._dawn_root / "dawn" / "capabilities"),
                str(self._dawn_root / "dawn" / "extensions")
            ],
            PermissionLevel.CORE_MODIFY: [
                str(self._dawn_root / "dawn" / "core"),
                str(self._dawn_root / "dawn" / "consciousness"),
                str(self._dawn_root / "dawn" / "processing")
            ]
        }
        
        logger.info(f"🔒 PermissionManager initialized with DAWN root: {self._dawn_root}")
        
    def request_permission(self, 
                          level: PermissionLevel,
                          target_paths: List[str],
                          reason: str,
                          scope: PermissionScope = PermissionScope.SINGLE_OPERATION,
                          duration: Optional[timedelta] = None) -> Optional[str]:
        """
        Request elevated permissions for specified operations.
        
        Args:
            level: Required permission level
            target_paths: Paths that need access
            reason: Justification for the request
            scope: Scope of the permission grant
            duration: How long the permission should last (for time-limited scope)
            
        Returns:
            Grant ID if approved, None if denied
        """
        with self._lock:
            request = PermissionRequest(
                request_id=str(uuid.uuid4()),
                level=level,
                scope=scope,
                target_paths=target_paths,
                reason=reason,
                requested_at=datetime.now(),
                consciousness_context=self._get_consciousness_context()
            )
            
            # Log the request
            self._audit_operation("permission_requested", {
                'request_id': request.request_id,
                'level': level.value,
                'paths': target_paths,
                'reason': reason,
                'scope': scope.value
            })
            
            # Evaluate the request
            approval_result = self._evaluate_permission_request(request)
            
            if approval_result['approved']:
                # Create the grant
                grant = self._create_permission_grant(request, duration)
                self._active_grants[grant.grant_id] = grant
                
                self._audit_operation("permission_granted", {
                    'grant_id': grant.grant_id,
                    'request_id': request.request_id,
                    'level': level.value,
                    'expires_at': grant.expires_at.isoformat() if grant.expires_at else None
                })
                
                logger.info(f"🔓 Permission granted: {level.value} for {len(target_paths)} paths")
                return grant.grant_id
            else:
                self._audit_operation("permission_denied", {
                    'request_id': request.request_id,
                    'level': level.value,
                    'reason': approval_result['reason']
                })
                
                logger.warning(f"🚫 Permission denied: {approval_result['reason']}")
                return None
    
    def _evaluate_permission_request(self, request: PermissionRequest) -> Dict[str, Any]:
        """Evaluate whether a permission request should be approved."""
        
        # Check consciousness level requirements
        consciousness_check = self._check_consciousness_requirements(request.level)
        if not consciousness_check['approved']:
            return consciousness_check
            
        # Check path boundaries
        boundary_check = self._check_path_boundaries(request.level, request.target_paths)
        if not boundary_check['approved']:
            return boundary_check
            
        # Check rate limits and session limits
        rate_check = self._check_rate_limits(request.level)
        if not rate_check['approved']:
            return rate_check
            
        # Additional safety checks for higher permission levels
        if request.level in [PermissionLevel.CORE_MODIFY, PermissionLevel.EMERGENCY_OVERRIDE]:
            safety_check = self._check_high_level_safety(request)
            if not safety_check['approved']:
                return safety_check
        
        return {'approved': True, 'reason': 'All checks passed'}
    
    def _check_consciousness_requirements(self, level: PermissionLevel) -> Dict[str, Any]:
        """Check if current consciousness level meets requirements for permission level."""
        try:
            state = get_state()
            
            # Define consciousness requirements for each permission level
            requirements = {
                PermissionLevel.READ_ONLY: {'min_level': 'aware', 'min_unity': 0.0},
                PermissionLevel.SANDBOX_MODIFY: {'min_level': 'aware', 'min_unity': 0.3},
                PermissionLevel.TOOLS_MODIFY: {'min_level': 'self_aware', 'min_unity': 0.5},
                PermissionLevel.SUBSYSTEM_MODIFY: {'min_level': 'meta_aware', 'min_unity': 0.7},
                PermissionLevel.CORE_MODIFY: {'min_level': 'transcendent', 'min_unity': 0.85},
                PermissionLevel.EMERGENCY_OVERRIDE: {'min_level': 'transcendent', 'min_unity': 0.9}
            }
            
            req = requirements.get(level, requirements[PermissionLevel.CORE_MODIFY])
            
            # Check level hierarchy
            level_hierarchy = ['dormant', 'aware', 'self_aware', 'meta_aware', 'transcendent']
            current_level_idx = level_hierarchy.index(state.level) if state.level in level_hierarchy else 0
            required_level_idx = level_hierarchy.index(req['min_level'])
            
            if current_level_idx < required_level_idx:
                return {
                    'approved': False,
                    'reason': f"Insufficient consciousness level: {state.level} (requires {req['min_level']})"
                }
                
            if state.unity < req['min_unity']:
                return {
                    'approved': False,
                    'reason': f"Insufficient unity score: {state.unity:.3f} (requires {req['min_unity']})"
                }
                
            return {'approved': True}
            
        except Exception as e:
            logger.error(f"Error checking consciousness requirements: {e}")
            return {'approved': False, 'reason': f"Could not verify consciousness state: {e}"}
    
    def _check_path_boundaries(self, level: PermissionLevel, target_paths: List[str]) -> Dict[str, Any]:
        """Check if target paths are within the permission boundaries for the level."""
        allowed_patterns = self._permission_boundaries.get(level, [])
        
        for path in target_paths:
            path_obj = Path(path).resolve()
            
            # Check if path is within any allowed pattern
            allowed = False
            for pattern in allowed_patterns:
                pattern_path = Path(pattern).resolve()
                
                # Handle glob patterns
                if '*' in pattern:
                    # Simple glob matching - could be enhanced
                    if pattern.endswith('*') and str(path_obj).startswith(pattern[:-1]):
                        allowed = True
                        break
                else:
                    # Check if path is under the allowed directory
                    try:
                        path_obj.relative_to(pattern_path)
                        allowed = True
                        break
                    except ValueError:
                        continue
                        
            if not allowed:
                return {
                    'approved': False,
                    'reason': f"Path '{path}' not allowed for permission level {level.value}"
                }
                
        return {'approved': True}
    
    def _check_rate_limits(self, level: PermissionLevel) -> Dict[str, Any]:
        """Check rate limits for permission requests."""
        now = datetime.now()
        
        # Count recent grants of this level
        recent_grants = [
            grant for grant in self._active_grants.values()
            if grant.level == level and (now - grant.granted_at).total_seconds() < 3600  # 1 hour
        ]
        
        # Define rate limits per hour
        rate_limits = {
            PermissionLevel.READ_ONLY: 1000,
            PermissionLevel.SANDBOX_MODIFY: 50,
            PermissionLevel.TOOLS_MODIFY: 20,
            PermissionLevel.SUBSYSTEM_MODIFY: 10,
            PermissionLevel.CORE_MODIFY: 3,
            PermissionLevel.EMERGENCY_OVERRIDE: 1
        }
        
        limit = rate_limits.get(level, 1)
        
        if len(recent_grants) >= limit:
            return {
                'approved': False,
                'reason': f"Rate limit exceeded for {level.value}: {len(recent_grants)}/{limit} per hour"
            }
            
        return {'approved': True}
    
    def _check_high_level_safety(self, request: PermissionRequest) -> Dict[str, Any]:
        """Additional safety checks for high-level permissions."""
        
        # For core modifications, require additional validation
        if request.level == PermissionLevel.CORE_MODIFY:
            # Check if there are any active sandbox operations
            active_sandboxes = self._count_active_sandboxes()
            if active_sandboxes > 0:
                return {
                    'approved': False,
                    'reason': f"Core modification blocked: {active_sandboxes} active sandbox operations"
                }
        
        # For emergency override, require very specific conditions
        if request.level == PermissionLevel.EMERGENCY_OVERRIDE:
            # This should only be used in genuine emergencies
            if "emergency" not in request.reason.lower() and "critical" not in request.reason.lower():
                return {
                    'approved': False,
                    'reason': "Emergency override requires explicit emergency justification"
                }
        
        return {'approved': True}
    
    def _create_permission_grant(self, request: PermissionRequest, duration: Optional[timedelta]) -> PermissionGrant:
        """Create a permission grant from an approved request."""
        state = get_state()
        
        expires_at = None
        if request.scope == PermissionScope.TIME_LIMITED and duration:
            expires_at = datetime.now() + duration
        elif request.scope == PermissionScope.SESSION_LIMITED:
            # Session grants expire when DAWN restarts
            expires_at = None  # Will be handled by session tracking
            
        max_operations = None
        if request.scope == PermissionScope.SINGLE_OPERATION:
            max_operations = 1
            
        return PermissionGrant(
            grant_id=str(uuid.uuid4()),
            level=request.level,
            scope=request.scope,
            target_paths=request.target_paths,
            granted_at=request.requested_at,
            expires_at=expires_at,
            consciousness_level_at_grant=state.level,
            unity_score_at_grant=state.unity,
            reason=request.reason,
            max_operations=max_operations
        )
    
    @contextmanager
    def elevated_access(self, grant_id: str, operation_description: str = ""):
        """
        Context manager for performing operations with elevated access.
        
        Usage:
            with permission_manager.elevated_access(grant_id, "modifying config") as access:
                if access:
                    # Perform elevated operations
                    access.write_file(path, content)
        """
        access_granted = False
        elevated_accessor = None
        
        try:
            # Validate and activate the grant
            if self._validate_and_activate_grant(grant_id, operation_description):
                access_granted = True
                elevated_accessor = ElevatedAccessor(self, grant_id)
                
                self._audit_operation("elevated_access_started", {
                    'grant_id': grant_id,
                    'operation': operation_description
                })
                
            yield elevated_accessor
            
        except Exception as e:
            self._audit_operation("elevated_access_error", {
                'grant_id': grant_id,
                'operation': operation_description,
                'error': str(e)
            })
            raise
            
        finally:
            if access_granted:
                self._audit_operation("elevated_access_completed", {
                    'grant_id': grant_id,
                    'operation': operation_description
                })
    
    def _validate_and_activate_grant(self, grant_id: str, operation_description: str) -> bool:
        """Validate that a grant is still valid and can be used."""
        with self._lock:
            grant = self._active_grants.get(grant_id)
            
            if not grant:
                logger.warning(f"🚫 Invalid grant ID: {grant_id}")
                return False
                
            if not grant.is_active:
                logger.warning(f"🚫 Grant {grant_id} is no longer active")
                return False
                
            # Check expiration
            if grant.expires_at and datetime.now() > grant.expires_at:
                grant.is_active = False
                logger.warning(f"🚫 Grant {grant_id} has expired")
                return False
                
            # Check operation limits
            if grant.max_operations and grant.used_operations >= grant.max_operations:
                grant.is_active = False
                logger.warning(f"🚫 Grant {grant_id} has reached operation limit")
                return False
                
            # Increment usage
            grant.used_operations += 1
            
            # Deactivate single-operation grants after use
            if grant.scope == PermissionScope.SINGLE_OPERATION:
                grant.is_active = False
                
            return True
    
    def revoke_grant(self, grant_id: str, reason: str = "") -> bool:
        """Revoke an active permission grant."""
        with self._lock:
            grant = self._active_grants.get(grant_id)
            
            if not grant:
                return False
                
            grant.is_active = False
            
            self._audit_operation("permission_revoked", {
                'grant_id': grant_id,
                'reason': reason
            })
            
            logger.info(f"🔒 Permission grant {grant_id} revoked: {reason}")
            return True
    
    def get_active_grants(self) -> List[PermissionGrant]:
        """Get all currently active permission grants."""
        with self._lock:
            return [grant for grant in self._active_grants.values() if grant.is_active]
    
    def _get_consciousness_context(self) -> Dict[str, Any]:
        """Get current consciousness context for requests."""
        try:
            state = get_state()
            return {
                'level': state.level,
                'unity': state.unity,
                'awareness': state.awareness,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _count_active_sandboxes(self) -> int:
        """Count active sandbox operations (placeholder - would integrate with sandbox system)."""
        # This would integrate with the existing sandbox runner
        return 0
    
    def _audit_operation(self, event_type: str, details: Dict[str, Any]):
        """Record an audit event."""
        event = PermissionAuditEvent(event_type, details)
        self._audit_log.append(event)
        
        # Keep audit log size manageable
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]  # Keep last 5000 events
    
    def get_audit_log(self, limit: int = 100) -> List[PermissionAuditEvent]:
        """Get recent audit events."""
        return self._audit_log[-limit:]


class ElevatedAccessor:
    """Provides elevated access operations within a permission context."""
    
    def __init__(self, permission_manager: PermissionManager, grant_id: str):
        self.permission_manager = permission_manager
        self.grant_id = grant_id
        self._grant = permission_manager._active_grants.get(grant_id)
        
    def write_file(self, path: str, content: str, backup: bool = True) -> bool:
        """Write a file with elevated permissions."""
        if not self._validate_path_access(path):
            return False
            
        try:
            path_obj = Path(path)
            
            # Create backup if requested
            if backup and path_obj.exists():
                backup_path = path_obj.with_suffix(path_obj.suffix + f'.backup.{int(datetime.now().timestamp())}')
                shutil.copy2(path_obj, backup_path)
                
            # Write the file
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.write_text(content)
            
            self.permission_manager._audit_operation("file_written", {
                'grant_id': self.grant_id,
                'path': str(path),
                'backup_created': backup
            })
            
            return True
            
        except Exception as e:
            self.permission_manager._audit_operation("file_write_error", {
                'grant_id': self.grant_id,
                'path': str(path),
                'error': str(e)
            })
            logger.error(f"Failed to write file {path}: {e}")
            return False
    
    def create_directory(self, path: str) -> bool:
        """Create a directory with elevated permissions."""
        if not self._validate_path_access(path):
            return False
            
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            
            self.permission_manager._audit_operation("directory_created", {
                'grant_id': self.grant_id,
                'path': str(path)
            })
            
            return True
            
        except Exception as e:
            self.permission_manager._audit_operation("directory_create_error", {
                'grant_id': self.grant_id,
                'path': str(path),
                'error': str(e)
            })
            logger.error(f"Failed to create directory {path}: {e}")
            return False
    
    def copy_file(self, src: str, dst: str) -> bool:
        """Copy a file with elevated permissions."""
        if not self._validate_path_access(src) or not self._validate_path_access(dst):
            return False
            
        try:
            shutil.copy2(src, dst)
            
            self.permission_manager._audit_operation("file_copied", {
                'grant_id': self.grant_id,
                'src': str(src),
                'dst': str(dst)
            })
            
            return True
            
        except Exception as e:
            self.permission_manager._audit_operation("file_copy_error", {
                'grant_id': self.grant_id,
                'src': str(src),
                'dst': str(dst),
                'error': str(e)
            })
            logger.error(f"Failed to copy file {src} to {dst}: {e}")
            return False
    
    def _validate_path_access(self, path: str) -> bool:
        """Validate that the path is within the granted permissions."""
        if not self._grant:
            return False
            
        path_obj = Path(path).resolve()
        
        for allowed_path in self._grant.target_paths:
            allowed_path_obj = Path(allowed_path).resolve()
            
            try:
                path_obj.relative_to(allowed_path_obj)
                return True
            except ValueError:
                continue
                
        logger.warning(f"🚫 Path access denied: {path} not in granted paths")
        return False


# Global permission manager instance
_permission_manager_instance: Optional[PermissionManager] = None
_permission_manager_lock = threading.Lock()

def get_permission_manager() -> PermissionManager:
    """Get the global permission manager instance."""
    global _permission_manager_instance
    
    with _permission_manager_lock:
        if _permission_manager_instance is None:
            _permission_manager_instance = PermissionManager()
            
        return _permission_manager_instance
