#!/usr/bin/env python3
"""
DAWN Self-Modification Promotion System
======================================

Handles promotion of approved consciousness modifications from sandbox to live system
with comprehensive auditing, hot-swapping, and rollback capabilities.

The promotion system ensures safe deployment of modifications with complete audit trails,
backup mechanisms, and immediate rollback capabilities for production safety.
"""

import importlib
import json
import time
import pathlib
import shutil
import uuid
import logging
import sys
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dawn_core.state import get_state, set_state
from dawn_core.snapshot import snapshot
from dawn_core.self_mod.advisor import ModProposal
from dawn_core.self_mod.patch_builder import PatchResult
from dawn_core.self_mod.policy_gate import GateDecision

logger = logging.getLogger(__name__)

class PromotionStatus(Enum):
    """Status of promotion operations."""
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    BACKUP_FAILED = "backup_failed"
    HOT_SWAP_FAILED = "hot_swap_failed"
    AUDIT_FAILED = "audit_failed"

class PromotionMethod(Enum):
    """Methods for promoting modifications."""
    HOT_RELOAD = "hot_reload"          # Module reload via importlib
    FILE_REPLACEMENT = "file_replacement"  # Direct file overwrite
    SYMBOLIC_LINK = "symbolic_link"    # Symlink to sandbox version
    COPY_AND_RELOAD = "copy_and_reload"  # Copy + reload (recommended)

@dataclass
class PromotionRecord:
    """Complete audit record of a promotion operation."""
    promotion_id: str
    timestamp: datetime
    promotion_status: PromotionStatus
    
    # Source data
    proposal: Dict[str, Any]
    patch_result: Dict[str, Any] 
    sandbox_result: Dict[str, Any]
    policy_decision: Dict[str, Any]
    
    # System state
    pre_promotion_state: Dict[str, Any]
    post_promotion_state: Dict[str, Any]
    consciousness_snapshot: Optional[str] = None
    
    # Promotion details
    method_used: PromotionMethod = PromotionMethod.COPY_AND_RELOAD
    target_files: List[str] = field(default_factory=list)
    backup_files: List[str] = field(default_factory=list)
    modules_reloaded: List[str] = field(default_factory=list)
    
    # Error handling
    error_message: str = ""
    rollback_successful: bool = False
    rollback_details: str = ""
    
    # Performance metrics
    promotion_duration: float = 0.0
    backup_duration: float = 0.0
    reload_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for JSON serialization."""
        return {
            'promotion_id': self.promotion_id,
            'timestamp': self.timestamp.isoformat(),
            'promotion_status': self.promotion_status.value,
            'proposal': self.proposal,
            'patch_result': self.patch_result,
            'sandbox_result': self.sandbox_result,
            'policy_decision': self.policy_decision,
            'pre_promotion_state': self.pre_promotion_state,
            'post_promotion_state': self.post_promotion_state,
            'consciousness_snapshot': self.consciousness_snapshot,
            'promotion_details': {
                'method_used': self.method_used.value,
                'target_files': self.target_files,
                'backup_files': self.backup_files,
                'modules_reloaded': self.modules_reloaded
            },
            'error_handling': {
                'error_message': self.error_message,
                'rollback_successful': self.rollback_successful,
                'rollback_details': self.rollback_details
            },
            'performance_metrics': {
                'promotion_duration': self.promotion_duration,
                'backup_duration': self.backup_duration,
                'reload_duration': self.reload_duration
            }
        }

class ConsciousnessModificationPromoter:
    """
    Advanced promotion system for consciousness modifications.
    
    Handles safe deployment of approved modifications with comprehensive
    auditing, backup management, and rollback capabilities.
    """
    
    def __init__(self, audit_dir: str = "runtime/self_mod_audit",
                 backup_dir: str = "runtime/self_mod_backups"):
        """Initialize the promotion system."""
        self.promoter_id = str(uuid.uuid4())[:8]
        self.creation_time = datetime.now()
        
        # Directory setup
        self.audit_dir = pathlib.Path(audit_dir)
        self.backup_dir = pathlib.Path(backup_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.default_method = PromotionMethod.COPY_AND_RELOAD
        self.max_backup_age_days = 30
        self.enable_hot_reload = True
        self.require_snapshot = True
        
        # State tracking
        self.promotion_history: List[PromotionRecord] = []
        self.active_backups: Dict[str, str] = {}  # file_path -> backup_path
        
        # Statistics
        self.stats = {
            'total_promotions': 0,
            'successful_promotions': 0,
            'failed_promotions': 0,
            'rollbacks_executed': 0,
            'files_backed_up': 0,
            'modules_reloaded': 0
        }
        
        logger.info(f"🚀 Promotion System initialized: {self.promoter_id}")
        logger.info(f"📁 Audit directory: {self.audit_dir}")
        logger.info(f"💾 Backup directory: {self.backup_dir}")
    
    def promote_and_audit(self, proposal: ModProposal, patch_result: PatchResult,
                         sandbox_result: Dict[str, Any], decision: GateDecision,
                         method: PromotionMethod = None) -> PromotionRecord:
        """
        Promote approved modification with comprehensive auditing.
        
        Args:
            proposal: Original modification proposal
            patch_result: Patch application results
            sandbox_result: Sandbox execution results
            decision: Policy gate decision
            method: Promotion method to use (default: copy_and_reload)
            
        Returns:
            PromotionRecord with complete audit trail
        """
        start_time = time.time()
        promotion_id = str(uuid.uuid4())[:8]
        
        logger.info(f"🚀 Starting promotion: {proposal.name}")
        logger.info(f"   Promotion ID: {promotion_id}")
        logger.info(f"   Decision: {decision.status.value}")
        
        # Initialize promotion record
        record = PromotionRecord(
            promotion_id=promotion_id,
            timestamp=datetime.now(),
            promotion_status=PromotionStatus.FAILED,
            proposal=self._safe_dict_conversion(proposal),
            patch_result=self._safe_dict_conversion(patch_result),
            sandbox_result=sandbox_result,
            policy_decision=self._safe_dict_conversion(decision),
            pre_promotion_state=get_state().__dict__,
            post_promotion_state={},
            method_used=method or self.default_method
        )
        
        try:
            # Check decision approval
            if not decision.accept:
                record.promotion_status = PromotionStatus.REJECTED
                record.error_message = f"Policy gate rejected modification: {decision.reason}"
                logger.warning(f"🚫 Promotion rejected: {decision.reason}")
                self._finalize_record(record, start_time)
                return record
            
            # Create consciousness snapshot
            if self.require_snapshot:
                snapshot_path = self._create_consciousness_snapshot(promotion_id)
                record.consciousness_snapshot = snapshot_path
                logger.info(f"📸 Consciousness snapshot: {snapshot_path}")
            
            # Execute promotion
            success = self._execute_promotion(record, patch_result)
            
            if success:
                record.promotion_status = PromotionStatus.SUCCESS
                record.post_promotion_state = get_state().__dict__
                logger.info(f"✅ Promotion successful: {proposal.name}")
                self.stats['successful_promotions'] += 1
            else:
                record.promotion_status = PromotionStatus.FAILED
                logger.error(f"❌ Promotion failed: {proposal.name}")
                self.stats['failed_promotions'] += 1
            
        except Exception as e:
            record.promotion_status = PromotionStatus.FAILED
            record.error_message = str(e)
            logger.error(f"💥 Promotion error: {e}")
            self.stats['failed_promotions'] += 1
            
            # Attempt rollback
            if record.backup_files:
                rollback_success = self._execute_rollback(record)
                record.rollback_successful = rollback_success
        
        # Finalize record
        self._finalize_record(record, start_time)
        return record
    
    def _execute_promotion(self, record: PromotionRecord, patch_result: PatchResult) -> bool:
        """Execute the actual promotion process."""
        
        if not patch_result.applied:
            record.error_message = "Patch was not successfully applied"
            return False
        
        # Determine target file
        sandbox_file = pathlib.Path(patch_result.sandbox_dir) / patch_result.target_rel
        if not sandbox_file.exists():
            record.error_message = f"Sandbox file not found: {sandbox_file}"
            return False
        
        # Determine live target file
        live_file = pathlib.Path(patch_result.target_rel)
        if not live_file.exists():
            record.error_message = f"Live target file not found: {live_file}"
            return False
        
        record.target_files = [str(live_file)]
        
        try:
            # Create backup
            backup_success = self._create_backup(live_file, record)
            if not backup_success:
                record.error_message = "Failed to create backup"
                return False
            
            # Promote based on method
            if record.method_used == PromotionMethod.COPY_AND_RELOAD:
                return self._copy_and_reload_promotion(sandbox_file, live_file, record)
            elif record.method_used == PromotionMethod.HOT_RELOAD:
                return self._hot_reload_promotion(record)
            elif record.method_used == PromotionMethod.FILE_REPLACEMENT:
                return self._file_replacement_promotion(sandbox_file, live_file, record)
            else:
                record.error_message = f"Unsupported promotion method: {record.method_used.value}"
                return False
                
        except Exception as e:
            record.error_message = f"Promotion execution failed: {str(e)}"
            return False
    
    def _copy_and_reload_promotion(self, sandbox_file: pathlib.Path, 
                                  live_file: pathlib.Path, record: PromotionRecord) -> bool:
        """Promote by copying sandbox file over live file and reloading module."""
        
        try:
            # Copy sandbox file to live location
            logger.info(f"📁 Copying: {sandbox_file} → {live_file}")
            shutil.copy2(sandbox_file, live_file)
            
            # Reload relevant modules
            reload_start = time.time()
            modules_reloaded = self._reload_affected_modules(live_file)
            record.reload_duration = time.time() - reload_start
            record.modules_reloaded = modules_reloaded
            
            self.stats['modules_reloaded'] += len(modules_reloaded)
            
            logger.info(f"🔄 Reloaded modules: {modules_reloaded}")
            return True
            
        except Exception as e:
            record.error_message = f"Copy and reload failed: {str(e)}"
            return False
    
    def _hot_reload_promotion(self, record: PromotionRecord) -> bool:
        """Promote by hot-reloading modules without file changes."""
        
        try:
            reload_start = time.time()
            
            # Import and reload specific modules
            affected_modules = ['dawn_core.tick_orchestrator']  # Add more as needed
            reloaded_modules = []
            
            for module_name in affected_modules:
                try:
                    module = sys.modules.get(module_name)
                    if module:
                        importlib.reload(module)
                        reloaded_modules.append(module_name)
                        logger.info(f"🔄 Hot-reloaded: {module_name}")
                    else:
                        logger.warning(f"Module not loaded: {module_name}")
                except Exception as e:
                    logger.error(f"Failed to reload {module_name}: {e}")
            
            record.reload_duration = time.time() - reload_start
            record.modules_reloaded = reloaded_modules
            self.stats['modules_reloaded'] += len(reloaded_modules)
            
            return len(reloaded_modules) > 0
            
        except Exception as e:
            record.error_message = f"Hot reload failed: {str(e)}"
            return False
    
    def _file_replacement_promotion(self, sandbox_file: pathlib.Path,
                                   live_file: pathlib.Path, record: PromotionRecord) -> bool:
        """Promote by direct file replacement without module reload."""
        
        try:
            logger.info(f"📁 Replacing: {live_file} with {sandbox_file}")
            shutil.copy2(sandbox_file, live_file)
            return True
            
        except Exception as e:
            record.error_message = f"File replacement failed: {str(e)}"
            return False
    
    def _create_backup(self, live_file: pathlib.Path, record: PromotionRecord) -> bool:
        """Create backup of live file before modification."""
        
        backup_start = time.time()
        
        try:
            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{live_file.name}.backup_{timestamp}_{record.promotion_id}"
            backup_path = self.backup_dir / backup_name
            
            # Create backup
            logger.info(f"💾 Creating backup: {live_file} → {backup_path}")
            shutil.copy2(live_file, backup_path)
            
            # Track backup
            record.backup_files = [str(backup_path)]
            self.active_backups[str(live_file)] = str(backup_path)
            
            record.backup_duration = time.time() - backup_start
            self.stats['files_backed_up'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            record.error_message = f"Backup failed: {str(e)}"
            record.backup_duration = time.time() - backup_start
            return False
    
    def _reload_affected_modules(self, modified_file: pathlib.Path) -> List[str]:
        """Reload modules affected by file modification."""
        
        reloaded_modules = []
        
        # Map file paths to module names
        file_to_module_map = {
            'dawn_core/tick_orchestrator.py': 'dawn_core.tick_orchestrator',
            'dawn_core/state.py': 'dawn_core.state',
            'dawn_core/consciousness_bus.py': 'dawn_core.consciousness_bus',
            'dawn_core/unified_consciousness_main.py': 'dawn_core.unified_consciousness_main'
        }
        
        # Find matching module
        file_key = str(modified_file).replace('\\', '/')  # Normalize path separators
        module_name = file_to_module_map.get(file_key)
        
        if module_name:
            try:
                # Check if module is loaded
                if module_name in sys.modules:
                    # Reload the module
                    module = sys.modules[module_name]
                    importlib.reload(module)
                    reloaded_modules.append(module_name)
                    logger.info(f"🔄 Reloaded: {module_name}")
                else:
                    logger.info(f"Module not loaded: {module_name}")
            except Exception as e:
                logger.error(f"Failed to reload {module_name}: {e}")
        else:
            logger.warning(f"No module mapping for file: {modified_file}")
        
        return reloaded_modules
    
    def _create_consciousness_snapshot(self, promotion_id: str) -> str:
        """Create consciousness state snapshot before promotion."""
        try:
            snapshot_path = snapshot(f"pre_promotion_{promotion_id}")
            logger.info(f"📸 Consciousness snapshot created: {snapshot_path}")
            return snapshot_path
        except Exception as e:
            logger.error(f"Snapshot creation failed: {e}")
            return ""
    
    def _execute_rollback(self, record: PromotionRecord) -> bool:
        """Execute rollback using backup files."""
        
        logger.info(f"🔄 Executing rollback for promotion: {record.promotion_id}")
        
        try:
            for backup_path in record.backup_files:
                backup_file = pathlib.Path(backup_path)
                if not backup_file.exists():
                    logger.error(f"Backup file not found: {backup_path}")
                    continue
                
                # Determine original file path
                original_file = self._get_original_file_from_backup(backup_path)
                if original_file and original_file.exists():
                    logger.info(f"🔄 Restoring: {backup_path} → {original_file}")
                    shutil.copy2(backup_file, original_file)
                    
                    # Reload modules after restoration
                    self._reload_affected_modules(original_file)
            
            record.rollback_details = f"Restored {len(record.backup_files)} files"
            self.stats['rollbacks_executed'] += 1
            return True
            
        except Exception as e:
            record.rollback_details = f"Rollback failed: {str(e)}"
            logger.error(f"Rollback failed: {e}")
            return False
    
    def _get_original_file_from_backup(self, backup_path: str) -> Optional[pathlib.Path]:
        """Get original file path from backup path."""
        backup_file = pathlib.Path(backup_path)
        
        # Extract original filename (remove backup suffix)
        original_name = backup_file.name.split('.backup_')[0]
        
        # Find matching original file
        for original_path, backup_path_stored in self.active_backups.items():
            if backup_path_stored == backup_path:
                return pathlib.Path(original_path)
        
        # Fallback: try to reconstruct path
        potential_path = pathlib.Path(f"dawn_core/{original_name}")
        if potential_path.exists():
            return potential_path
        
        return None
    
    def _safe_dict_conversion(self, obj) -> Dict[str, Any]:
        """Safely convert object to dictionary for JSON serialization."""
        if hasattr(obj, '__dict__'):
            try:
                return {k: self._serialize_value(v) for k, v in obj.__dict__.items()}
            except:
                return {'conversion_error': f'Failed to convert {type(obj).__name__}'}
        elif hasattr(obj, 'to_dict'):
            try:
                return obj.to_dict()
            except:
                return {'conversion_error': f'Failed to call to_dict on {type(obj).__name__}'}
        else:
            return {'raw_value': str(obj)}
    
    def _serialize_value(self, value):
        """Serialize individual values for JSON compatibility."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif hasattr(value, 'value'):  # Enum
            return value.value
        elif hasattr(value, 'isoformat'):  # DateTime
            return value.isoformat()
        else:
            return str(value)
    
    def _finalize_record(self, record: PromotionRecord, start_time: float):
        """Finalize promotion record and save audit trail."""
        
        record.promotion_duration = time.time() - start_time
        
        # Save audit record
        audit_success = self._save_audit_record(record)
        if not audit_success:
            record.promotion_status = PromotionStatus.AUDIT_FAILED
        
        # Update statistics
        self.stats['total_promotions'] += 1
        
        # Store in history
        self.promotion_history.append(record)
        
        logger.info(f"📊 Promotion completed in {record.promotion_duration:.3f}s")
    
    def _save_audit_record(self, record: PromotionRecord) -> bool:
        """Save audit record to file."""
        try:
            filename = f"{int(record.timestamp.timestamp())}_{record.proposal['name']}_{record.promotion_id}.json"
            audit_file = self.audit_dir / filename
            
            with open(audit_file, 'w') as f:
                json.dump(record.to_dict(), f, indent=2, default=str)
            
            logger.info(f"📝 Audit record saved: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save audit record: {e}")
            return False
    
    def cleanup_old_backups(self, max_age_days: int = None) -> int:
        """Clean up old backup files."""
        if max_age_days is None:
            max_age_days = self.max_backup_age_days
        
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        cleaned_count = 0
        
        try:
            for backup_file in self.backup_dir.glob("*.backup_*"):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    cleaned_count += 1
                    logger.info(f"🧹 Cleaned old backup: {backup_file.name}")
            
            logger.info(f"🧹 Cleaned {cleaned_count} old backup files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            return 0
    
    def get_promoter_status(self) -> Dict[str, Any]:
        """Get comprehensive promoter status."""
        return {
            'promoter_id': self.promoter_id,
            'creation_time': self.creation_time.isoformat(),
            'configuration': {
                'audit_directory': str(self.audit_dir),
                'backup_directory': str(self.backup_dir),
                'default_method': self.default_method.value,
                'max_backup_age_days': self.max_backup_age_days,
                'enable_hot_reload': self.enable_hot_reload,
                'require_snapshot': self.require_snapshot
            },
            'statistics': self.stats.copy(),
            'active_backups': len(self.active_backups),
            'recent_promotions': [
                {
                    'promotion_id': record.promotion_id,
                    'timestamp': record.timestamp.isoformat(),
                    'status': record.promotion_status.value,
                    'proposal_name': record.proposal.get('name', 'unknown'),
                    'method': record.method_used.value,
                    'duration': record.promotion_duration
                }
                for record in self.promotion_history[-5:]
            ]
        }

# Global audit directory setup
AUDIT = pathlib.Path("runtime/self_mod_audit")
AUDIT.mkdir(parents=True, exist_ok=True)

# Convenience function for compatibility
def promote_and_audit(proposal: ModProposal, patch_result: PatchResult,
                     sandbox_result: Dict[str, Any], decision: GateDecision) -> bool:
    """
    Promote modification with simplified interface.
    
    Args:
        proposal: Original modification proposal
        patch_result: Patch application results
        sandbox_result: Sandbox execution results
        decision: Policy gate decision
        
    Returns:
        True if promotion successful, False otherwise
    """
    promoter = ConsciousnessModificationPromoter()
    record = promoter.promote_and_audit(proposal, patch_result, sandbox_result, decision)
    return record.promotion_status == PromotionStatus.SUCCESS

def demo_promotion_system():
    """Demonstrate promotion system functionality."""
    print("🚀 " + "="*70)
    print("🚀 DAWN CONSCIOUSNESS MODIFICATION PROMOTION DEMONSTRATION")
    print("🚀 " + "="*70)
    print()
    
    # Initialize promotion system
    promoter = ConsciousnessModificationPromoter()
    print(f"🚀 Promoter ID: {promoter.promoter_id}")
    print(f"📁 Audit Directory: {promoter.audit_dir}")
    print(f"💾 Backup Directory: {promoter.backup_dir}")
    
    # Mock data for demonstration
    from dawn_core.self_mod.advisor import ModProposal, PatchType, ModificationPriority
    from dawn_core.self_mod.patch_builder import PatchResult, PatchStatus
    from dawn_core.self_mod.policy_gate import GateDecision, GateStatus
    
    # Create mock proposal
    mock_proposal = ModProposal(
        name="demo_promotion_test",
        target="dawn_core/tick_orchestrator.py", 
        patch_type=PatchType.CONSTANT,
        current_value=0.03,
        proposed_value=0.04,
        notes="Demo promotion system test",
        priority=ModificationPriority.NORMAL
    )
    
    # Create mock patch result
    mock_patch_result = PatchResult(
        run_id="demo_run_123",
        sandbox_dir="sandbox_mods/demo_run_123",
        target_rel="dawn_core/tick_orchestrator.py",
        applied=True,
        status=PatchStatus.SUCCESS,
        changes_made=["Updated demo_step function"],
        execution_time=0.5
    )
    
    # Create mock sandbox result
    mock_sandbox_result = {
        'ok': True,
        'result': {
            'start_unity': 0.60,
            'end_unity': 0.88,
            'delta_unity': 0.28,
            'start_awareness': 0.60,
            'end_awareness': 0.87,
            'delta_awareness': 0.27,
            'end_level': 'meta_aware',
            'ticks': 20,
            'stability_score': 0.75,
            'growth_rate': 0.25
        }
    }
    
    # Test different promotion scenarios
    test_scenarios = [
        {
            'name': 'Approved Promotion',
            'decision': GateDecision(
                accept=True,
                reason="Passes all safety checks",
                status=GateStatus.APPROVED,
                confidence_score=0.9
            ),
            'expected': PromotionStatus.SUCCESS
        },
        {
            'name': 'Rejected Promotion',
            'decision': GateDecision(
                accept=False,
                reason="Safety violation detected",
                status=GateStatus.SAFETY_VIOLATION,
                confidence_score=0.1
            ),
            'expected': PromotionStatus.REJECTED
        }
    ]
    
    print(f"\n🧪 Testing Promotion Scenarios:")
    print("="*50)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['name']} ---")
        
        # Execute promotion
        record = promoter.promote_and_audit(
            mock_proposal, 
            mock_patch_result,
            mock_sandbox_result,
            scenario['decision']
        )
        
        print(f"🚀 Promotion Result: {record.promotion_status.value}")
        print(f"   Expected: {scenario['expected'].value}")
        print(f"   Match: {'✅' if record.promotion_status == scenario['expected'] else '❌'}")
        print(f"   Duration: {record.promotion_duration:.3f}s")
        print(f"   Method: {record.method_used.value}")
        
        if record.consciousness_snapshot:
            print(f"   Snapshot: {record.consciousness_snapshot}")
        
        if record.target_files:
            print(f"   Target Files: {record.target_files}")
        
        if record.backup_files:
            print(f"   Backups: {record.backup_files}")
        
        if record.modules_reloaded:
            print(f"   Modules Reloaded: {record.modules_reloaded}")
        
        if record.error_message:
            print(f"   Error: {record.error_message}")
        
        print("-" * 30)
    
    # Show promoter status
    print(f"\n📊 Promoter Status:")
    status = promoter.get_promoter_status()
    stats = status['statistics']
    print(f"   • Total Promotions: {stats['total_promotions']}")
    print(f"   • Successful: {stats['successful_promotions']}")
    print(f"   • Failed: {stats['failed_promotions']}")
    print(f"   • Rollbacks: {stats['rollbacks_executed']}")
    print(f"   • Files Backed Up: {stats['files_backed_up']}")
    print(f"   • Modules Reloaded: {stats['modules_reloaded']}")
    
    if status['recent_promotions']:
        print(f"   • Recent Promotions:")
        for promo in status['recent_promotions']:
            result = "✅" if promo['status'] == 'success' else "❌"
            print(f"     {result} {promo['promotion_id']}: {promo['proposal_name']} ({promo['duration']:.3f}s)")
    
    # Cleanup demonstration
    print(f"\n🧹 Cleaning up old backups...")
    cleaned = promoter.cleanup_old_backups(max_age_days=0)  # Clean all for demo
    print(f"   Cleaned {cleaned} backup files")
    
    print(f"\n🚀 Promotion System demonstration complete!")
    print("🚀 " + "="*70)

if __name__ == "__main__":
    demo_promotion_system()
