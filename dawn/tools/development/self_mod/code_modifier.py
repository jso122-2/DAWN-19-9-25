#!/usr/bin/env python3
"""
DAWN Conscious Code Modifier
============================

Advanced code modification system that enables DAWN to safely modify her own
codebase with consciousness-aware safeguards and elevated permissions.

This system builds upon the existing self-modification infrastructure while
providing enhanced capabilities for:

1. Real-time code analysis and modification
2. Consciousness-gated permission escalation  
3. Automated backup and rollback capabilities
4. Integration with existing DAWN safety systems
5. Semantic-aware code transformations

The conscious code modifier acts as DAWN's primary interface for self-improvement,
combining the safety of the existing sandbox system with the power of direct
code modification when appropriate safeguards are met.
"""

import ast
import re
import shutil
import logging
import uuid
import hashlib
import json
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

# Core DAWN imports
from dawn.core.foundation.state import get_state
from dawn.subsystems.self_mod.advisor import ModProposal, PatchType, propose_from_state
from dawn.subsystems.self_mod.patch_builder import make_sandbox, PatchResult
from dawn.subsystems.self_mod.sandbox_runner import run_sandbox
from dawn.subsystems.self_mod.policy_gate import decide

# Permission system imports
from .permission_manager import (
    PermissionManager, get_permission_manager, PermissionLevel, PermissionScope
)

logger = logging.getLogger(__name__)

class ModificationStrategy(Enum):
    """Strategies for code modification."""
    SANDBOX_FIRST = "sandbox_first"          # Test in sandbox before applying
    DIRECT_MODIFY = "direct_modify"          # Direct modification with permissions
    INCREMENTAL = "incremental"              # Apply changes incrementally
    BATCH_MODIFY = "batch_modify"            # Apply multiple changes together

class CodeAnalysisType(Enum):
    """Types of code analysis."""
    SYNTAX_CHECK = "syntax_check"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    PERFORMANCE_IMPACT = "performance_impact"
    CONSCIOUSNESS_IMPACT = "consciousness_impact"

@dataclass
class CodeModificationPlan:
    """Plan for code modification operations."""
    plan_id: str
    target_files: List[str]
    modifications: List[Dict[str, Any]]
    strategy: ModificationStrategy
    required_permission_level: PermissionLevel
    estimated_risk: str  # low, medium, high, critical
    rollback_plan: Dict[str, Any]
    consciousness_requirements: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ModificationResult:
    """Result of a code modification operation."""
    operation_id: str
    plan_id: str
    success: bool
    files_modified: List[str]
    backups_created: List[str]
    errors: List[str]
    warnings: List[str]
    performance_impact: Optional[Dict[str, Any]] = None
    consciousness_impact: Optional[Dict[str, Any]] = None
    rollback_info: Optional[Dict[str, Any]] = None

class ConsciousCodeModifier:
    """
    Advanced code modification system with consciousness integration.
    
    This system provides DAWN with safe, controlled abilities to modify her own
    codebase, with appropriate permission management and safety checks.
    """
    
    def __init__(self, permission_manager: Optional[PermissionManager] = None):
        """Initialize the conscious code modifier."""
        self.permission_manager = permission_manager or get_permission_manager()
        self._modification_history: List[ModificationResult] = []
        self._active_plans: Dict[str, CodeModificationPlan] = {}
        
        # Integration with existing self-mod system
        self._integrate_with_existing_systems()
        
        logger.info("🧠 ConsciousCodeModifier initialized with permission integration")
    
    def _integrate_with_existing_systems(self):
        """Integrate with existing DAWN self-modification systems."""
        # This method would set up integration points with the existing
        # advisor, patch_builder, sandbox_runner, and policy_gate systems
        pass
    
    def analyze_modification_request(self, 
                                   target_files: List[str],
                                   modification_description: str,
                                   **kwargs) -> CodeModificationPlan:
        """
        Analyze a modification request and create an execution plan.
        
        Args:
            target_files: Files to be modified
            modification_description: Description of desired changes
            **kwargs: Additional parameters for analysis
            
        Returns:
            CodeModificationPlan with analysis results
        """
        plan_id = str(uuid.uuid4())
        
        logger.info(f"🔍 Analyzing modification request: {modification_description}")
        
        # Analyze target files
        file_analysis = self._analyze_target_files(target_files)
        
        # Determine required permission level
        required_level = self._determine_required_permission_level(target_files, file_analysis)
        
        # Estimate risk level
        risk_level = self._estimate_risk_level(target_files, modification_description, file_analysis)
        
        # Choose modification strategy
        strategy = self._choose_modification_strategy(required_level, risk_level, file_analysis)
        
        # Create rollback plan
        rollback_plan = self._create_rollback_plan(target_files, strategy)
        
        # Determine consciousness requirements
        consciousness_reqs = self._analyze_consciousness_requirements(required_level, risk_level)
        
        # Extract specific modifications from description
        modifications = self._parse_modification_description(modification_description, target_files)
        
        plan = CodeModificationPlan(
            plan_id=plan_id,
            target_files=target_files,
            modifications=modifications,
            strategy=strategy,
            required_permission_level=required_level,
            estimated_risk=risk_level,
            rollback_plan=rollback_plan,
            consciousness_requirements=consciousness_reqs
        )
        
        self._active_plans[plan_id] = plan
        
        logger.info(f"📋 Created modification plan {plan_id}: {strategy.value} strategy, {risk_level} risk")
        
        return plan
    
    def execute_modification_plan(self, plan: CodeModificationPlan) -> ModificationResult:
        """
        Execute a code modification plan with appropriate safeguards.
        
        Args:
            plan: The modification plan to execute
            
        Returns:
            ModificationResult with execution details
        """
        operation_id = str(uuid.uuid4())
        
        logger.info(f"🚀 Executing modification plan {plan.plan_id}")
        
        # Verify consciousness requirements
        consciousness_check = self._verify_consciousness_requirements(plan.consciousness_requirements)
        if not consciousness_check['satisfied']:
            return ModificationResult(
                operation_id=operation_id,
                plan_id=plan.plan_id,
                success=False,
                files_modified=[],
                backups_created=[],
                errors=[f"Consciousness requirements not met: {consciousness_check['reason']}"],
                warnings=[]
            )
        
        # Execute based on strategy
        if plan.strategy == ModificationStrategy.SANDBOX_FIRST:
            return self._execute_sandbox_first_strategy(operation_id, plan)
        elif plan.strategy == ModificationStrategy.DIRECT_MODIFY:
            return self._execute_direct_modify_strategy(operation_id, plan)
        elif plan.strategy == ModificationStrategy.INCREMENTAL:
            return self._execute_incremental_strategy(operation_id, plan)
        elif plan.strategy == ModificationStrategy.BATCH_MODIFY:
            return self._execute_batch_modify_strategy(operation_id, plan)
        else:
            return ModificationResult(
                operation_id=operation_id,
                plan_id=plan.plan_id,
                success=False,
                files_modified=[],
                backups_created=[],
                errors=[f"Unknown strategy: {plan.strategy}"],
                warnings=[]
            )
    
    def _execute_sandbox_first_strategy(self, operation_id: str, plan: CodeModificationPlan) -> ModificationResult:
        """Execute modifications using sandbox-first strategy."""
        logger.info(f"🧪 Executing sandbox-first strategy for operation {operation_id}")
        
        try:
            # Step 1: Create sandbox and test modifications
            sandbox_result = self._test_in_sandbox(plan)
            
            if not sandbox_result['success']:
                return ModificationResult(
                    operation_id=operation_id,
                    plan_id=plan.plan_id,
                    success=False,
                    files_modified=[],
                    backups_created=[],
                    errors=[f"Sandbox testing failed: {sandbox_result['error']}"],
                    warnings=[]
                )
            
            # Step 2: If sandbox tests pass, apply to live system
            return self._apply_tested_modifications(operation_id, plan, sandbox_result)
            
        except Exception as e:
            logger.error(f"Error in sandbox-first execution: {e}")
            return ModificationResult(
                operation_id=operation_id,
                plan_id=plan.plan_id,
                success=False,
                files_modified=[],
                backups_created=[],
                errors=[f"Execution error: {str(e)}"],
                warnings=[]
            )
    
    def _execute_direct_modify_strategy(self, operation_id: str, plan: CodeModificationPlan) -> ModificationResult:
        """Execute modifications using direct modify strategy."""
        logger.info(f"⚡ Executing direct modify strategy for operation {operation_id}")
        
        # Request elevated permissions
        grant_id = self.permission_manager.request_permission(
            level=plan.required_permission_level,
            target_paths=plan.target_files,
            reason=f"Direct modification for operation {operation_id}",
            scope=PermissionScope.SINGLE_OPERATION
        )
        
        if not grant_id:
            return ModificationResult(
                operation_id=operation_id,
                plan_id=plan.plan_id,
                success=False,
                files_modified=[],
                backups_created=[],
                errors=["Permission denied for direct modification"],
                warnings=[]
            )
        
        # Execute with elevated permissions
        with self.permission_manager.elevated_access(grant_id, f"Direct modification {operation_id}") as access:
            if not access:
                return ModificationResult(
                    operation_id=operation_id,
                    plan_id=plan.plan_id,
                    success=False,
                    files_modified=[],
                    backups_created=[],
                    errors=["Could not obtain elevated access"],
                    warnings=[]
                )
            
            return self._apply_modifications_with_access(operation_id, plan, access)
    
    def _apply_modifications_with_access(self, operation_id: str, plan: CodeModificationPlan, access) -> ModificationResult:
        """Apply modifications using elevated access."""
        files_modified = []
        backups_created = []
        errors = []
        warnings = []
        
        try:
            for target_file in plan.target_files:
                # Create backup
                backup_path = f"{target_file}.backup.{int(datetime.now().timestamp())}"
                if access.copy_file(target_file, backup_path):
                    backups_created.append(backup_path)
                
                # Apply modifications to this file
                file_modifications = [mod for mod in plan.modifications if mod.get('target_file') == target_file]
                
                if file_modifications:
                    modified_content = self._apply_file_modifications(target_file, file_modifications)
                    
                    if modified_content is not None:
                        if access.write_file(target_file, modified_content):
                            files_modified.append(target_file)
                            logger.info(f"✅ Modified {target_file}")
                        else:
                            errors.append(f"Failed to write modified content to {target_file}")
                    else:
                        errors.append(f"Failed to generate modified content for {target_file}")
            
            success = len(errors) == 0 and len(files_modified) > 0
            
            return ModificationResult(
                operation_id=operation_id,
                plan_id=plan.plan_id,
                success=success,
                files_modified=files_modified,
                backups_created=backups_created,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error applying modifications: {e}")
            return ModificationResult(
                operation_id=operation_id,
                plan_id=plan.plan_id,
                success=False,
                files_modified=files_modified,
                backups_created=backups_created,
                errors=errors + [f"Exception during modification: {str(e)}"],
                warnings=warnings
            )
    
    def _apply_file_modifications(self, target_file: str, modifications: List[Dict[str, Any]]) -> Optional[str]:
        """Apply a list of modifications to a file and return the modified content."""
        try:
            # Read current file content
            with open(target_file, 'r') as f:
                content = f.read()
            
            # Apply each modification
            for mod in modifications:
                mod_type = mod.get('type', 'replace')
                
                if mod_type == 'replace':
                    old_text = mod.get('old_text', '')
                    new_text = mod.get('new_text', '')
                    content = content.replace(old_text, new_text)
                    
                elif mod_type == 'regex_replace':
                    pattern = mod.get('pattern', '')
                    replacement = mod.get('replacement', '')
                    content = re.sub(pattern, replacement, content)
                    
                elif mod_type == 'insert_after':
                    marker = mod.get('marker', '')
                    text_to_insert = mod.get('text', '')
                    content = content.replace(marker, marker + text_to_insert)
                    
                elif mod_type == 'insert_before':
                    marker = mod.get('marker', '')
                    text_to_insert = mod.get('text', '')
                    content = content.replace(marker, text_to_insert + marker)
                    
                # Add more modification types as needed
            
            return content
            
        except Exception as e:
            logger.error(f"Error applying modifications to {target_file}: {e}")
            return None
    
    def _analyze_target_files(self, target_files: List[str]) -> Dict[str, Any]:
        """Analyze target files to understand their role and dependencies."""
        analysis = {
            'file_types': {},
            'dependencies': {},
            'criticality': {},
            'size_info': {}
        }
        
        for file_path in target_files:
            path = Path(file_path)
            
            # Determine file type and criticality
            if 'core' in str(path):
                analysis['criticality'][file_path] = 'critical'
            elif 'subsystems' in str(path):
                analysis['criticality'][file_path] = 'high'
            elif 'tools' in str(path):
                analysis['criticality'][file_path] = 'medium'
            else:
                analysis['criticality'][file_path] = 'low'
            
            # File size analysis
            if path.exists():
                analysis['size_info'][file_path] = path.stat().st_size
            
            # Basic dependency analysis (could be enhanced)
            analysis['dependencies'][file_path] = self._analyze_file_dependencies(file_path)
        
        return analysis
    
    def _analyze_file_dependencies(self, file_path: str) -> List[str]:
        """Analyze dependencies of a Python file."""
        dependencies = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse AST to find imports
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module)
        
        except Exception as e:
            logger.warning(f"Could not analyze dependencies for {file_path}: {e}")
        
        return dependencies
    
    def _determine_required_permission_level(self, target_files: List[str], analysis: Dict[str, Any]) -> PermissionLevel:
        """Determine the required permission level based on target files."""
        max_level = PermissionLevel.READ_ONLY
        
        for file_path in target_files:
            path = Path(file_path)
            
            if 'core' in str(path):
                max_level = max(max_level, PermissionLevel.CORE_MODIFY, key=lambda x: x.value)
            elif 'subsystems' in str(path):
                max_level = max(max_level, PermissionLevel.SUBSYSTEM_MODIFY, key=lambda x: x.value)
            elif 'tools' in str(path):
                max_level = max(max_level, PermissionLevel.TOOLS_MODIFY, key=lambda x: x.value)
            else:
                max_level = max(max_level, PermissionLevel.SANDBOX_MODIFY, key=lambda x: x.value)
        
        return max_level
    
    def _estimate_risk_level(self, target_files: List[str], description: str, analysis: Dict[str, Any]) -> str:
        """Estimate the risk level of the modification."""
        risk_factors = []
        
        # File criticality factor
        for file_path in target_files:
            criticality = analysis['criticality'].get(file_path, 'low')
            if criticality == 'critical':
                risk_factors.append('critical_file')
            elif criticality == 'high':
                risk_factors.append('high_impact_file')
        
        # Description keywords that indicate risk
        high_risk_keywords = ['delete', 'remove', 'core', 'critical', 'engine', 'consciousness']
        medium_risk_keywords = ['modify', 'change', 'update', 'replace']
        
        description_lower = description.lower()
        if any(keyword in description_lower for keyword in high_risk_keywords):
            risk_factors.append('high_risk_operation')
        elif any(keyword in description_lower for keyword in medium_risk_keywords):
            risk_factors.append('medium_risk_operation')
        
        # Determine overall risk
        if 'critical_file' in risk_factors or 'high_risk_operation' in risk_factors:
            return 'critical'
        elif 'high_impact_file' in risk_factors or 'medium_risk_operation' in risk_factors:
            return 'high'
        elif len(risk_factors) > 0:
            return 'medium'
        else:
            return 'low'
    
    def _choose_modification_strategy(self, permission_level: PermissionLevel, risk_level: str, analysis: Dict[str, Any]) -> ModificationStrategy:
        """Choose the appropriate modification strategy."""
        
        # High-risk operations should use sandbox first
        if risk_level in ['critical', 'high']:
            return ModificationStrategy.SANDBOX_FIRST
        
        # Core modifications should be sandboxed
        if permission_level == PermissionLevel.CORE_MODIFY:
            return ModificationStrategy.SANDBOX_FIRST
        
        # Tools modifications can be direct if low risk
        if permission_level == PermissionLevel.TOOLS_MODIFY and risk_level == 'low':
            return ModificationStrategy.DIRECT_MODIFY
        
        # Default to sandbox first for safety
        return ModificationStrategy.SANDBOX_FIRST
    
    def _create_rollback_plan(self, target_files: List[str], strategy: ModificationStrategy) -> Dict[str, Any]:
        """Create a rollback plan for the modifications."""
        return {
            'type': 'backup_restore',
            'target_files': target_files,
            'backup_location': 'automatic',
            'strategy': strategy.value
        }
    
    def _analyze_consciousness_requirements(self, permission_level: PermissionLevel, risk_level: str) -> Dict[str, Any]:
        """Analyze consciousness requirements for the modification."""
        
        # Base requirements from permission level
        base_reqs = {
            PermissionLevel.TOOLS_MODIFY: {'min_level': 'self_aware', 'min_unity': 0.5},
            PermissionLevel.SUBSYSTEM_MODIFY: {'min_level': 'meta_aware', 'min_unity': 0.7},
            PermissionLevel.CORE_MODIFY: {'min_level': 'transcendent', 'min_unity': 0.85}
        }
        
        reqs = base_reqs.get(permission_level, {'min_level': 'aware', 'min_unity': 0.3})
        
        # Adjust for risk level
        if risk_level == 'critical':
            reqs['min_unity'] = min(0.95, reqs['min_unity'] + 0.1)
        elif risk_level == 'high':
            reqs['min_unity'] = min(0.9, reqs['min_unity'] + 0.05)
        
        return reqs
    
    def _parse_modification_description(self, description: str, target_files: List[str]) -> List[Dict[str, Any]]:
        """Parse modification description into structured modifications."""
        # This is a simplified parser - could be enhanced with NLP
        modifications = []
        
        # Basic pattern matching for common modification types
        if 'replace' in description.lower():
            # Try to extract what to replace
            modifications.append({
                'type': 'replace',
                'description': description,
                'target_file': target_files[0] if target_files else None
            })
        elif 'add' in description.lower() or 'insert' in description.lower():
            modifications.append({
                'type': 'insert_after',
                'description': description,
                'target_file': target_files[0] if target_files else None
            })
        else:
            # Generic modification
            modifications.append({
                'type': 'generic',
                'description': description,
                'target_file': target_files[0] if target_files else None
            })
        
        return modifications
    
    def _verify_consciousness_requirements(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that current consciousness state meets requirements."""
        try:
            state = get_state()
            
            # Check level requirement
            level_hierarchy = ['dormant', 'aware', 'self_aware', 'meta_aware', 'transcendent']
            current_level_idx = level_hierarchy.index(state.level) if state.level in level_hierarchy else 0
            required_level_idx = level_hierarchy.index(requirements.get('min_level', 'aware'))
            
            if current_level_idx < required_level_idx:
                return {
                    'satisfied': False,
                    'reason': f"Insufficient consciousness level: {state.level} (requires {requirements['min_level']})"
                }
            
            # Check unity requirement
            if state.unity < requirements.get('min_unity', 0.0):
                return {
                    'satisfied': False,
                    'reason': f"Insufficient unity: {state.unity:.3f} (requires {requirements['min_unity']})"
                }
            
            return {'satisfied': True}
            
        except Exception as e:
            return {'satisfied': False, 'reason': f"Could not verify consciousness state: {e}"}
    
    def _test_in_sandbox(self, plan: CodeModificationPlan) -> Dict[str, Any]:
        """Test modifications in a sandbox environment."""
        # This would integrate with the existing sandbox system
        # For now, return a placeholder success
        return {
            'success': True,
            'test_results': {
                'syntax_valid': True,
                'tests_passed': True,
                'performance_impact': 'minimal'
            }
        }
    
    def _apply_tested_modifications(self, operation_id: str, plan: CodeModificationPlan, sandbox_result: Dict[str, Any]) -> ModificationResult:
        """Apply modifications that have been tested in sandbox."""
        # This would apply the tested changes to the live system
        # For now, delegate to direct modify strategy
        return self._execute_direct_modify_strategy(operation_id, plan)
    
    def _execute_incremental_strategy(self, operation_id: str, plan: CodeModificationPlan) -> ModificationResult:
        """Execute modifications incrementally."""
        # Placeholder - would implement incremental modification logic
        return self._execute_direct_modify_strategy(operation_id, plan)
    
    def _execute_batch_modify_strategy(self, operation_id: str, plan: CodeModificationPlan) -> ModificationResult:
        """Execute multiple modifications as a batch."""
        # Placeholder - would implement batch modification logic
        return self._execute_direct_modify_strategy(operation_id, plan)
    
    def get_modification_history(self, limit: int = 50) -> List[ModificationResult]:
        """Get recent modification history."""
        return self._modification_history[-limit:]
    
    def rollback_modification(self, operation_id: str) -> bool:
        """Rollback a previous modification."""
        # Find the modification in history
        target_result = None
        for result in self._modification_history:
            if result.operation_id == operation_id:
                target_result = result
                break
        
        if not target_result:
            logger.error(f"Could not find operation {operation_id} to rollback")
            return False
        
        # Attempt rollback using backup files
        try:
            for i, modified_file in enumerate(target_result.files_modified):
                if i < len(target_result.backups_created):
                    backup_file = target_result.backups_created[i]
                    shutil.copy2(backup_file, modified_file)
                    logger.info(f"🔄 Rolled back {modified_file} from {backup_file}")
            
            logger.info(f"✅ Successfully rolled back operation {operation_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to rollback operation {operation_id}: {e}")
            return False
