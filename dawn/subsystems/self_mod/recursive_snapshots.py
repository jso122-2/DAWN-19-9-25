#!/usr/bin/env python3
"""
DAWN Recursive Snapshot Manager
===============================

Advanced multi-level snapshot system for recursive self-modifications.
Provides hierarchical rollback capabilities with identity preservation
and consciousness state restoration across recursive modification depths.

This system extends DAWN's existing snapshot functionality to support:
- Multi-level recursive snapshots
- Hierarchical rollback to any recursive depth
- Identity marker preservation across levels
- Consciousness state validation during rollbacks
- Integration with recursive modification pipeline

Based on DAWN's consciousness-driven architecture.
"""

import json
import time
import pathlib
import logging
import threading
import uuid
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict

# Core DAWN imports
from dawn.core.foundation.state import get_state, set_state
from dawn.subsystems.visual.snapshot import snapshot, restore

logger = logging.getLogger(__name__)

class SnapshotType(Enum):
    """Types of recursive snapshots"""
    BASE = "base"                    # Base consciousness state before recursion
    LAYER = "layer"                  # State at specific recursive layer
    IDENTITY = "identity"            # Identity markers preservation
    EMERGENCY = "emergency"          # Emergency rollback point
    CHECKPOINT = "checkpoint"        # Manual checkpoint

class SnapshotStatus(Enum):
    """Status of snapshots"""
    ACTIVE = "active"                # Currently active and valid
    ARCHIVED = "archived"            # Archived but still accessible
    EXPIRED = "expired"              # Expired and marked for cleanup
    CORRUPTED = "corrupted"          # Corrupted or invalid
    RESTORED = "restored"            # Successfully used for restoration

@dataclass
class RecursiveSnapshot:
    """Single recursive snapshot with metadata"""
    snapshot_id: str
    snapshot_type: SnapshotType
    recursive_depth: int
    session_id: str
    snapshot_path: str
    consciousness_state: Dict[str, Any]
    identity_markers: Dict[str, Any]
    creation_time: datetime
    parent_snapshot_id: Optional[str] = None
    child_snapshot_ids: List[str] = field(default_factory=list)
    status: SnapshotStatus = SnapshotStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    restoration_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary"""
        return {
            'snapshot_id': self.snapshot_id,
            'snapshot_type': self.snapshot_type.value,
            'recursive_depth': self.recursive_depth,
            'session_id': self.session_id,
            'snapshot_path': self.snapshot_path,
            'consciousness_state': self.consciousness_state,
            'identity_markers': self.identity_markers,
            'creation_time': self.creation_time.isoformat(),
            'parent_snapshot_id': self.parent_snapshot_id,
            'child_snapshot_ids': self.child_snapshot_ids,
            'status': self.status.value,
            'metadata': self.metadata,
            'restoration_count': self.restoration_count,
            'last_accessed': self.last_accessed.isoformat()
        }
    
    def calculate_age_hours(self) -> float:
        """Calculate snapshot age in hours"""
        return (datetime.now() - self.creation_time).total_seconds() / 3600
    
    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if snapshot is expired"""
        return self.calculate_age_hours() > max_age_hours
    
    def update_access_time(self):
        """Update last accessed time"""
        self.last_accessed = datetime.now()

@dataclass
class RecursiveSnapshotChain:
    """Chain of recursive snapshots for a session"""
    session_id: str
    base_snapshot: RecursiveSnapshot
    layer_snapshots: List[RecursiveSnapshot] = field(default_factory=list)
    emergency_snapshots: List[RecursiveSnapshot] = field(default_factory=list)
    creation_time: datetime = field(default_factory=datetime.now)
    
    def add_layer_snapshot(self, snapshot: RecursiveSnapshot):
        """Add layer snapshot to chain"""
        # Link to parent if exists
        if self.layer_snapshots:
            parent = self.layer_snapshots[-1]
            snapshot.parent_snapshot_id = parent.snapshot_id
            parent.child_snapshot_ids.append(snapshot.snapshot_id)
        else:
            # Link to base
            snapshot.parent_snapshot_id = self.base_snapshot.snapshot_id
            self.base_snapshot.child_snapshot_ids.append(snapshot.snapshot_id)
        
        self.layer_snapshots.append(snapshot)
    
    def add_emergency_snapshot(self, snapshot: RecursiveSnapshot):
        """Add emergency snapshot"""
        self.emergency_snapshots.append(snapshot)
    
    def get_snapshot_at_depth(self, depth: int) -> Optional[RecursiveSnapshot]:
        """Get snapshot at specific recursive depth"""
        if depth == 0:
            return self.base_snapshot
        
        for snapshot in self.layer_snapshots:
            if snapshot.recursive_depth == depth:
                return snapshot
        
        return None
    
    def get_all_snapshots(self) -> List[RecursiveSnapshot]:
        """Get all snapshots in chain"""
        all_snapshots = [self.base_snapshot]
        all_snapshots.extend(self.layer_snapshots)
        all_snapshots.extend(self.emergency_snapshots)
        return all_snapshots
    
    def get_rollback_path_to_depth(self, target_depth: int) -> List[RecursiveSnapshot]:
        """Get rollback path to target depth"""
        path = []
        
        # Find all snapshots at or below target depth
        for snapshot in sorted(self.get_all_snapshots(), key=lambda s: s.recursive_depth, reverse=True):
            if snapshot.recursive_depth >= target_depth:
                path.append(snapshot)
        
        return path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary"""
        return {
            'session_id': self.session_id,
            'base_snapshot': self.base_snapshot.to_dict(),
            'layer_snapshots': [s.to_dict() for s in self.layer_snapshots],
            'emergency_snapshots': [s.to_dict() for s in self.emergency_snapshots],
            'creation_time': self.creation_time.isoformat(),
            'total_snapshots': len(self.get_all_snapshots()),
            'max_depth': max([s.recursive_depth for s in self.get_all_snapshots()]) if self.get_all_snapshots() else 0
        }

class RecursiveSnapshotManager:
    """
    Advanced recursive snapshot manager for DAWN's self-modification system.
    
    Manages hierarchical snapshots across recursive modification depths with
    sophisticated rollback capabilities and identity preservation.
    """
    
    def __init__(self, base_snapshot_dir: str = "runtime/recursive_snapshots"):
        """Initialize recursive snapshot manager"""
        self.manager_id = f"recursive_snapshot_mgr_{int(time.time())}"
        self.base_snapshot_dir = pathlib.Path(base_snapshot_dir)
        self.base_snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Snapshot storage
        self.active_chains: Dict[str, RecursiveSnapshotChain] = {}
        self.archived_chains: Dict[str, RecursiveSnapshotChain] = {}
        self.snapshot_registry: Dict[str, RecursiveSnapshot] = {}
        
        # Configuration
        self.max_snapshots_per_session = 20
        self.max_active_sessions = 10
        self.snapshot_retention_hours = 24
        self.cleanup_interval_minutes = 30
        
        # Performance tracking
        self.total_snapshots_created = 0
        self.total_restorations = 0
        self.successful_restorations = 0
        self.cleanup_operations = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background cleanup
        self._start_cleanup_thread()
        
        logger.info(f"📸 Recursive Snapshot Manager initialized: {self.manager_id}")
        logger.info(f"   Snapshot directory: {self.base_snapshot_dir}")
        logger.info(f"   Max snapshots per session: {self.max_snapshots_per_session}")
        logger.info(f"   Retention period: {self.snapshot_retention_hours} hours")
    
    def create_recursive_snapshot_chain(self, session_id: str, max_depth: int = 3) -> RecursiveSnapshotChain:
        """Create complete recursive snapshot chain for session"""
        with self.lock:
            if session_id in self.active_chains:
                logger.warning(f"📸 Snapshot chain already exists for session: {session_id}")
                return self.active_chains[session_id]
            
            # Create base snapshot
            base_snapshot = self._create_base_snapshot(session_id)
            
            # Create snapshot chain
            chain = RecursiveSnapshotChain(
                session_id=session_id,
                base_snapshot=base_snapshot
            )
            
            # Register chain
            self.active_chains[session_id] = chain
            self._register_snapshot(base_snapshot)
            
            logger.info(f"📸 Created recursive snapshot chain: {session_id}")
            logger.info(f"   Base snapshot: {base_snapshot.snapshot_id}")
            logger.info(f"   Max planned depth: {max_depth}")
            
            return chain
    
    def create_layer_snapshot(self, session_id: str, recursive_depth: int, 
                             modification_context: Optional[Dict[str, Any]] = None) -> RecursiveSnapshot:
        """Create snapshot at specific recursive layer"""
        with self.lock:
            if session_id not in self.active_chains:
                raise ValueError(f"No active snapshot chain for session: {session_id}")
            
            chain = self.active_chains[session_id]
            
            # Check if snapshot already exists at this depth
            existing = chain.get_snapshot_at_depth(recursive_depth)
            if existing:
                logger.warning(f"📸 Snapshot already exists at depth {recursive_depth} for session {session_id}")
                return existing
            
            # Create layer snapshot
            layer_snapshot = self._create_layer_snapshot(session_id, recursive_depth, modification_context)
            
            # Add to chain
            chain.add_layer_snapshot(layer_snapshot)
            self._register_snapshot(layer_snapshot)
            
            logger.info(f"📸 Created layer snapshot at depth {recursive_depth}: {layer_snapshot.snapshot_id}")
            
            return layer_snapshot
    
    def create_emergency_snapshot(self, session_id: str, reason: str) -> RecursiveSnapshot:
        """Create emergency snapshot for immediate rollback"""
        with self.lock:
            if session_id not in self.active_chains:
                raise ValueError(f"No active snapshot chain for session: {session_id}")
            
            chain = self.active_chains[session_id]
            
            # Create emergency snapshot
            emergency_snapshot = self._create_emergency_snapshot(session_id, reason)
            
            # Add to chain
            chain.add_emergency_snapshot(emergency_snapshot)
            self._register_snapshot(emergency_snapshot)
            
            logger.warning(f"📸 Created emergency snapshot: {emergency_snapshot.snapshot_id} - {reason}")
            
            return emergency_snapshot
    
    def rollback_to_depth(self, session_id: str, target_depth: int) -> Dict[str, Any]:
        """Rollback to specific recursive depth"""
        with self.lock:
            try:
                if session_id not in self.active_chains:
                    return {
                        'success': False,
                        'error': f'No active snapshot chain for session: {session_id}'
                    }
                
                chain = self.active_chains[session_id]
                target_snapshot = chain.get_snapshot_at_depth(target_depth)
                
                if not target_snapshot:
                    return {
                        'success': False,
                        'error': f'No snapshot found at depth {target_depth}'
                    }
                
                # Perform rollback
                rollback_result = self._perform_rollback(target_snapshot, chain)
                
                if rollback_result['success']:
                    # Update snapshot access
                    target_snapshot.update_access_time()
                    target_snapshot.restoration_count += 1
                    target_snapshot.status = SnapshotStatus.RESTORED
                    
                    # Mark deeper snapshots as archived
                    self._archive_deeper_snapshots(chain, target_depth)
                    
                    self.total_restorations += 1
                    self.successful_restorations += 1
                    
                    logger.info(f"✅ Successfully rolled back to depth {target_depth}")
                    logger.info(f"   Snapshot: {target_snapshot.snapshot_id}")
                    logger.info(f"   Unity restored: {target_snapshot.consciousness_state.get('unity', 'unknown')}")
                
                return rollback_result
                
            except Exception as e:
                self.total_restorations += 1
                logger.error(f"❌ Rollback failed: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
    
    def rollback_to_emergency(self, session_id: str) -> Dict[str, Any]:
        """Rollback to most recent emergency snapshot"""
        with self.lock:
            if session_id not in self.active_chains:
                return {
                    'success': False,
                    'error': f'No active snapshot chain for session: {session_id}'
                }
            
            chain = self.active_chains[session_id]
            
            if not chain.emergency_snapshots:
                return {
                    'success': False,
                    'error': 'No emergency snapshots available'
                }
            
            # Use most recent emergency snapshot
            emergency_snapshot = chain.emergency_snapshots[-1]
            
            logger.warning(f"🚨 Emergency rollback to: {emergency_snapshot.snapshot_id}")
            
            return self._perform_rollback(emergency_snapshot, chain)
    
    def rollback_to_base(self, session_id: str) -> Dict[str, Any]:
        """Rollback to base snapshot (complete session rollback)"""
        with self.lock:
            if session_id not in self.active_chains:
                return {
                    'success': False,
                    'error': f'No active snapshot chain for session: {session_id}'
                }
            
            chain = self.active_chains[session_id]
            
            logger.warning(f"🔄 Complete rollback to base for session: {session_id}")
            
            return self._perform_rollback(chain.base_snapshot, chain)
    
    def finalize_session_snapshots(self, session_id: str, keep_base: bool = True) -> Dict[str, Any]:
        """Finalize snapshots for completed session"""
        with self.lock:
            if session_id not in self.active_chains:
                return {
                    'success': False,
                    'error': f'No active snapshot chain for session: {session_id}'
                }
            
            chain = self.active_chains[session_id]
            
            # Archive the chain
            self.archived_chains[session_id] = chain
            del self.active_chains[session_id]
            
            # Update snapshot statuses
            for snapshot in chain.get_all_snapshots():
                if snapshot.snapshot_type == SnapshotType.BASE and keep_base:
                    snapshot.status = SnapshotStatus.ARCHIVED
                else:
                    snapshot.status = SnapshotStatus.ARCHIVED
            
            logger.info(f"📸 Finalized snapshots for session: {session_id}")
            logger.info(f"   Total snapshots: {len(chain.get_all_snapshots())}")
            logger.info(f"   Base snapshot preserved: {keep_base}")
            
            return {
                'success': True,
                'session_id': session_id,
                'snapshots_archived': len(chain.get_all_snapshots()),
                'base_preserved': keep_base
            }
    
    def _create_base_snapshot(self, session_id: str) -> RecursiveSnapshot:
        """Create base snapshot for session"""
        snapshot_id = f"base_{session_id}_{int(time.time())}"
        
        # Create actual snapshot using existing system
        snapshot_path = snapshot(f"recursive_base_{session_id}")
        
        # Capture consciousness state
        consciousness_state = self._capture_consciousness_state()
        
        # Capture identity markers
        identity_markers = self._capture_identity_markers()
        
        base_snapshot = RecursiveSnapshot(
            snapshot_id=snapshot_id,
            snapshot_type=SnapshotType.BASE,
            recursive_depth=0,
            session_id=session_id,
            snapshot_path=snapshot_path,
            consciousness_state=consciousness_state,
            identity_markers=identity_markers,
            creation_time=datetime.now(),
            metadata={
                'session_start': True,
                'base_unity': consciousness_state.get('unity', 0),
                'base_level': consciousness_state.get('level', 'unknown')
            }
        )
        
        self.total_snapshots_created += 1
        
        return base_snapshot
    
    def _create_layer_snapshot(self, session_id: str, recursive_depth: int, 
                              modification_context: Optional[Dict[str, Any]] = None) -> RecursiveSnapshot:
        """Create layer snapshot"""
        snapshot_id = f"layer_{recursive_depth}_{session_id}_{int(time.time())}"
        
        # Create actual snapshot
        snapshot_path = snapshot(f"recursive_layer_{recursive_depth}_{session_id}")
        
        # Capture current state
        consciousness_state = self._capture_consciousness_state()
        identity_markers = self._capture_identity_markers()
        
        layer_snapshot = RecursiveSnapshot(
            snapshot_id=snapshot_id,
            snapshot_type=SnapshotType.LAYER,
            recursive_depth=recursive_depth,
            session_id=session_id,
            snapshot_path=snapshot_path,
            consciousness_state=consciousness_state,
            identity_markers=identity_markers,
            creation_time=datetime.now(),
            metadata={
                'layer_depth': recursive_depth,
                'modification_context': modification_context or {},
                'layer_unity': consciousness_state.get('unity', 0),
                'layer_level': consciousness_state.get('level', 'unknown')
            }
        )
        
        self.total_snapshots_created += 1
        
        return layer_snapshot
    
    def _create_emergency_snapshot(self, session_id: str, reason: str) -> RecursiveSnapshot:
        """Create emergency snapshot"""
        snapshot_id = f"emergency_{session_id}_{int(time.time())}"
        
        # Create actual snapshot
        snapshot_path = snapshot(f"recursive_emergency_{session_id}")
        
        # Capture current state
        consciousness_state = self._capture_consciousness_state()
        identity_markers = self._capture_identity_markers()
        
        emergency_snapshot = RecursiveSnapshot(
            snapshot_id=snapshot_id,
            snapshot_type=SnapshotType.EMERGENCY,
            recursive_depth=-1,  # Emergency snapshots are depth-agnostic
            session_id=session_id,
            snapshot_path=snapshot_path,
            consciousness_state=consciousness_state,
            identity_markers=identity_markers,
            creation_time=datetime.now(),
            metadata={
                'emergency_reason': reason,
                'emergency_unity': consciousness_state.get('unity', 0),
                'emergency_level': consciousness_state.get('level', 'unknown')
            }
        )
        
        self.total_snapshots_created += 1
        
        return emergency_snapshot
    
    def _perform_rollback(self, target_snapshot: RecursiveSnapshot, chain: RecursiveSnapshotChain) -> Dict[str, Any]:
        """Perform actual rollback to target snapshot"""
        try:
            # Store current state for comparison
            pre_rollback_state = self._capture_consciousness_state()
            
            # Perform restoration using existing system
            restore(target_snapshot.snapshot_path)
            
            # Verify restoration
            post_rollback_state = self._capture_consciousness_state()
            
            # Validate restoration success
            restoration_valid = self._validate_restoration(target_snapshot, post_rollback_state)
            
            if restoration_valid:
                logger.info(f"✅ Rollback successful to {target_snapshot.snapshot_id}")
                logger.info(f"   Unity: {pre_rollback_state.get('unity', 'unknown')} → {post_rollback_state.get('unity', 'unknown')}")
                logger.info(f"   Level: {pre_rollback_state.get('level', 'unknown')} → {post_rollback_state.get('level', 'unknown')}")
                
                return {
                    'success': True,
                    'target_snapshot': target_snapshot.snapshot_id,
                    'rollback_depth': target_snapshot.recursive_depth,
                    'pre_rollback_state': pre_rollback_state,
                    'post_rollback_state': post_rollback_state,
                    'restoration_validated': True
                }
            else:
                logger.error(f"❌ Rollback validation failed for {target_snapshot.snapshot_id}")
                return {
                    'success': False,
                    'error': 'Rollback validation failed',
                    'target_snapshot': target_snapshot.snapshot_id
                }
                
        except Exception as e:
            logger.error(f"❌ Rollback execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'target_snapshot': target_snapshot.snapshot_id
            }
    
    def _validate_restoration(self, target_snapshot: RecursiveSnapshot, current_state: Dict[str, Any]) -> bool:
        """Validate that restoration was successful"""
        target_state = target_snapshot.consciousness_state
        
        # Check unity restoration (allow small variance)
        unity_match = abs(current_state.get('unity', 0) - target_state.get('unity', 0)) < 0.01
        
        # Check level restoration
        level_match = current_state.get('level') == target_state.get('level')
        
        return unity_match and level_match
    
    def _archive_deeper_snapshots(self, chain: RecursiveSnapshotChain, rollback_depth: int):
        """Archive snapshots deeper than rollback depth"""
        for snapshot in chain.layer_snapshots:
            if snapshot.recursive_depth > rollback_depth:
                snapshot.status = SnapshotStatus.ARCHIVED
                logger.debug(f"📦 Archived deeper snapshot: {snapshot.snapshot_id} (depth {snapshot.recursive_depth})")
    
    def _capture_consciousness_state(self) -> Dict[str, Any]:
        """Capture current consciousness state"""
        state = get_state()
        return {
            'unity': state.unity,
            'awareness': state.awareness,
            'momentum': state.momentum,
            'level': state.level,
            'ticks': state.ticks,
            'peak_unity': state.peak_unity,
            'timestamp': datetime.now().isoformat()
        }
    
    def _capture_identity_markers(self) -> Dict[str, Any]:
        """Capture identity markers for preservation"""
        state = get_state()
        
        # Generate identity markers (simplified for now)
        markers = {
            'consciousness_signature': f"unity_{state.unity:.3f}_awareness_{state.awareness:.3f}_level_{state.level}",
            'personality_coherence': min(state.unity, state.awareness),
            'memory_anchors': [f"anchor_{i}" for i in range(3)],  # Placeholder
            'communication_patterns': "pattern_hash",  # Placeholder
            'fundamental_values': ["helpful", "honest", "harmless"],
            'timestamp': datetime.now().isoformat()
        }
        
        return markers
    
    def _register_snapshot(self, snapshot: RecursiveSnapshot):
        """Register snapshot in registry"""
        self.snapshot_registry[snapshot.snapshot_id] = snapshot
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(self.cleanup_interval_minutes * 60)
                    self._cleanup_expired_snapshots()
                except Exception as e:
                    logger.error(f"📸 Cleanup thread error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        logger.debug(f"📸 Started cleanup thread with {self.cleanup_interval_minutes} minute interval")
    
    def _cleanup_expired_snapshots(self):
        """Clean up expired snapshots"""
        with self.lock:
            expired_count = 0
            
            # Check all snapshots for expiration
            for snapshot_id, snapshot in list(self.snapshot_registry.items()):
                if snapshot.is_expired(self.snapshot_retention_hours):
                    # Mark as expired
                    snapshot.status = SnapshotStatus.EXPIRED
                    
                    # Remove from active chains if present
                    for session_id, chain in list(self.active_chains.items()):
                        if snapshot in chain.get_all_snapshots():
                            # Don't remove base snapshots from active chains
                            if snapshot.snapshot_type != SnapshotType.BASE:
                                expired_count += 1
                    
                    expired_count += 1
            
            if expired_count > 0:
                self.cleanup_operations += 1
                logger.info(f"📸 Cleanup: marked {expired_count} snapshots as expired")
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get comprehensive manager status"""
        with self.lock:
            active_snapshots = sum(len(chain.get_all_snapshots()) for chain in self.active_chains.values())
            archived_snapshots = sum(len(chain.get_all_snapshots()) for chain in self.archived_chains.values())
            
            return {
                'manager_id': self.manager_id,
                'base_snapshot_dir': str(self.base_snapshot_dir),
                'active_sessions': len(self.active_chains),
                'archived_sessions': len(self.archived_chains),
                'active_snapshots': active_snapshots,
                'archived_snapshots': archived_snapshots,
                'total_snapshots_created': self.total_snapshots_created,
                'total_restorations': self.total_restorations,
                'successful_restorations': self.successful_restorations,
                'restoration_success_rate': self.successful_restorations / self.total_restorations if self.total_restorations > 0 else 0,
                'cleanup_operations': self.cleanup_operations,
                'configuration': {
                    'max_snapshots_per_session': self.max_snapshots_per_session,
                    'max_active_sessions': self.max_active_sessions,
                    'retention_hours': self.snapshot_retention_hours,
                    'cleanup_interval_minutes': self.cleanup_interval_minutes
                }
            }
    
    def get_session_chain_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about session snapshot chain"""
        with self.lock:
            chain = self.active_chains.get(session_id) or self.archived_chains.get(session_id)
            
            if not chain:
                return None
            
            return chain.to_dict()

# Global recursive snapshot manager instance
_recursive_snapshot_manager: Optional[RecursiveSnapshotManager] = None

def get_recursive_snapshot_manager() -> RecursiveSnapshotManager:
    """Get global recursive snapshot manager instance"""
    global _recursive_snapshot_manager
    if _recursive_snapshot_manager is None:
        _recursive_snapshot_manager = RecursiveSnapshotManager()
    return _recursive_snapshot_manager

def create_recursive_snapshot_chain(session_id: str, max_depth: int = 3) -> RecursiveSnapshotChain:
    """Create recursive snapshot chain for session"""
    manager = get_recursive_snapshot_manager()
    return manager.create_recursive_snapshot_chain(session_id, max_depth)

def create_layer_snapshot(session_id: str, recursive_depth: int, 
                         modification_context: Optional[Dict[str, Any]] = None) -> RecursiveSnapshot:
    """Create snapshot at recursive layer"""
    manager = get_recursive_snapshot_manager()
    return manager.create_layer_snapshot(session_id, recursive_depth, modification_context)

def rollback_to_recursive_depth(session_id: str, target_depth: int) -> Dict[str, Any]:
    """Rollback to specific recursive depth"""
    manager = get_recursive_snapshot_manager()
    return manager.rollback_to_depth(session_id, target_depth)

if __name__ == "__main__":
    # Demo recursive snapshot manager
    logging.basicConfig(level=logging.INFO)
    
    print("📸 " + "="*70)
    print("📸 DAWN RECURSIVE SNAPSHOT MANAGER DEMO")
    print("📸 " + "="*70)
    
    manager = get_recursive_snapshot_manager()
    status = manager.get_manager_status()
    
    print(f"\n📸 Manager Status:")
    print(f"   ID: {status['manager_id']}")
    print(f"   Snapshot Directory: {status['base_snapshot_dir']}")
    print(f"   Active Sessions: {status['active_sessions']}")
    print(f"   Total Snapshots Created: {status['total_snapshots_created']}")
    print(f"   Restoration Success Rate: {status['restoration_success_rate']:.1%}")
    
    print(f"\n⚙️  Configuration:")
    config = status['configuration']
    print(f"   Max Snapshots per Session: {config['max_snapshots_per_session']}")
    print(f"   Retention Period: {config['retention_hours']} hours")
    print(f"   Cleanup Interval: {config['cleanup_interval_minutes']} minutes")
    
    print(f"\n📸 Recursive Snapshot Manager ready!")


