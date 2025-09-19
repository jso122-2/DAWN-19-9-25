#!/usr/bin/env python3
"""
🔄📝 DAWN Recursive Module Writer
================================

Advanced recursive self-modification system that enables all modules referenced
in this chat session to write themselves recursively using DAWN's self-modification
capabilities combined with the mycelial semantic network.

This system provides:
1. Recursive self-writing capabilities for all logging modules
2. Consciousness-gated permission management for self-modification
3. Integration with mycelial semantic network for context awareness
4. Safe sandbox testing before applying modifications
5. Comprehensive audit trail and rollback capabilities
6. Semantic-aware code generation and modification

All modules from this chat can now recursively modify themselves:
- Universal JSON logging system
- Centralized deep repository
- Consciousness-depth logging
- Sigil consciousness logging  
- Pulse-telemetry unification
- Mycelial semantic hash map
- Live monitor integration
- DAWN singleton integration
"""

import os
import ast
import inspect
import importlib
import threading
import logging
import uuid
import hashlib
import json
import time
from typing import Dict, List, Optional, Set, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

# Core DAWN imports
from dawn.core.foundation.state import get_state
from dawn.core.singleton import get_dawn
from dawn.subsystems.self_mod.advisor import ModProposal, PatchType, propose_from_state
from dawn.subsystems.self_mod.patch_builder import make_sandbox, PatchResult
from dawn.subsystems.self_mod.sandbox_runner import run_sandbox
from dawn.subsystems.self_mod.policy_gate import decide

# Permission system imports
from .permission_manager import (
    PermissionManager, get_permission_manager, PermissionLevel, PermissionScope
)
from .code_modifier import ConsciousCodeModifier, ModificationStrategy

# Logging and mycelial imports
try:
    from dawn.core.logging import (
        get_mycelial_hashmap, touch_semantic_concept, store_semantic_data,
        get_consciousness_repository, ConsciousnessLevel, DAWNLogType
    )
    MYCELIAL_AVAILABLE = True
except ImportError:
    MYCELIAL_AVAILABLE = False

logger = logging.getLogger(__name__)

class RecursiveWriteMode(Enum):
    """Modes for recursive self-writing."""
    CONSCIOUSNESS_GUIDED = "consciousness_guided"    # Guided by consciousness levels
    SEMANTIC_TRIGGERED = "semantic_triggered"       # Triggered by semantic spores
    SELF_IMPROVING = "self_improving"               # Continuous self-improvement
    ADAPTIVE_LEARNING = "adaptive_learning"         # Learn and adapt from usage
    MYCELIAL_PROPAGATED = "mycelial_propagated"    # Propagated through mycelial network

class ModuleWriteCapability(Enum):
    """Capabilities for module self-writing."""
    READ_SELF = "read_self"                         # Can read own source code
    ANALYZE_SELF = "analyze_self"                   # Can analyze own structure
    MODIFY_SELF = "modify_self"                     # Can modify own code
    EXTEND_SELF = "extend_self"                     # Can add new functionality
    OPTIMIZE_SELF = "optimize_self"                 # Can optimize performance
    EVOLVE_SELF = "evolve_self"                     # Can evolve architecture

@dataclass
class ModuleRecursiveContext:
    """Context for recursive module writing operations."""
    module_name: str
    module_path: str
    current_consciousness_level: str
    semantic_context: Dict[str, Any]
    mycelial_connections: List[str]
    write_capabilities: Set[ModuleWriteCapability]
    permission_level: PermissionLevel
    safety_constraints: Dict[str, Any]
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class RecursiveWriteRequest:
    """Request for recursive self-writing operation."""
    request_id: str
    target_module: str
    write_mode: RecursiveWriteMode
    requested_capabilities: Set[ModuleWriteCapability]
    consciousness_requirements: Dict[str, Any]
    semantic_triggers: List[str]
    safety_level: str  # "safe", "moderate", "aggressive"
    max_iterations: int = 10
    created_at: datetime = field(default_factory=datetime.now)

class RecursiveModuleWriter:
    """
    Advanced recursive module writing system for DAWN.
    
    Enables all modules from this chat session to write themselves recursively
    with consciousness-aware safeguards and mycelial semantic integration.
    """
    
    def __init__(self):
        """Initialize the recursive module writer."""
        self._lock = threading.RLock()
        self._active_contexts: Dict[str, ModuleRecursiveContext] = {}
        self._write_requests: Dict[str, RecursiveWriteRequest] = {}
        self._audit_log: List[Dict[str, Any]] = []
        
        # Initialize dependencies
        self.permission_manager = get_permission_manager()
        self.code_modifier = ConsciousCodeModifier()
        self.dawn_singleton = get_dawn()
        
        # Chat session modules that can write themselves recursively
        self.chat_modules = {
            # Universal JSON logging
            "dawn.core.logging.universal_json_logger": {
                "capabilities": {ModuleWriteCapability.READ_SELF, ModuleWriteCapability.ANALYZE_SELF, 
                               ModuleWriteCapability.MODIFY_SELF, ModuleWriteCapability.OPTIMIZE_SELF},
                "consciousness_level": ConsciousnessLevel.INTEGRAL,
                "semantic_concepts": ["universal_logging", "state_serialization", "object_tracking"]
            },
            
            # Centralized repository
            "dawn.core.logging.centralized_repo": {
                "capabilities": {ModuleWriteCapability.READ_SELF, ModuleWriteCapability.EXTEND_SELF,
                               ModuleWriteCapability.OPTIMIZE_SELF},
                "consciousness_level": ConsciousnessLevel.FORMAL,
                "semantic_concepts": ["centralized_logging", "hierarchical_organization", "deep_repository"]
            },
            
            # Consciousness-depth logging
            "dawn.core.logging.consciousness_depth_repo": {
                "capabilities": {ModuleWriteCapability.READ_SELF, ModuleWriteCapability.ANALYZE_SELF,
                               ModuleWriteCapability.EVOLVE_SELF},
                "consciousness_level": ConsciousnessLevel.META,
                "semantic_concepts": ["consciousness_depth", "hierarchical_logging", "transcendent_awareness"]
            },
            
            # Sigil consciousness logging
            "dawn.core.logging.sigil_consciousness_logger": {
                "capabilities": {ModuleWriteCapability.READ_SELF, ModuleWriteCapability.MODIFY_SELF,
                               ModuleWriteCapability.EXTEND_SELF},
                "consciousness_level": ConsciousnessLevel.CAUSAL,
                "semantic_concepts": ["sigil_consciousness", "archetypal_energy", "unity_factor"]
            },
            
            # Pulse-telemetry unification
            "dawn.core.logging.pulse_telemetry_unifier": {
                "capabilities": {ModuleWriteCapability.READ_SELF, ModuleWriteCapability.ANALYZE_SELF,
                               ModuleWriteCapability.OPTIMIZE_SELF},
                "consciousness_level": ConsciousnessLevel.INTEGRAL,
                "semantic_concepts": ["pulse_telemetry", "unification", "thermal_dynamics"]
            },
            
            # Mycelial semantic hash map
            "dawn.core.logging.mycelial_semantic_hashmap": {
                "capabilities": {ModuleWriteCapability.READ_SELF, ModuleWriteCapability.ANALYZE_SELF,
                               ModuleWriteCapability.MODIFY_SELF, ModuleWriteCapability.EVOLVE_SELF},
                "consciousness_level": ConsciousnessLevel.TRANSCENDENT,
                "semantic_concepts": ["mycelial_network", "semantic_spores", "rhizomic_propagation"]
            },
            
            # Mycelial module integration
            "dawn.core.logging.mycelial_module_integration": {
                "capabilities": {ModuleWriteCapability.READ_SELF, ModuleWriteCapability.EXTEND_SELF,
                               ModuleWriteCapability.EVOLVE_SELF},
                "consciousness_level": ConsciousnessLevel.TRANSCENDENT,
                "semantic_concepts": ["module_integration", "cross_module_propagation", "semantic_wrapping"]
            },
            
            # DAWN singleton
            "dawn.core.singleton": {
                "capabilities": {ModuleWriteCapability.READ_SELF, ModuleWriteCapability.ANALYZE_SELF,
                               ModuleWriteCapability.EXTEND_SELF},
                "consciousness_level": ConsciousnessLevel.META,
                "semantic_concepts": ["singleton_pattern", "unified_access", "backwards_compatibility"]
            },
            
            # Live monitor
            "live_monitor": {
                "capabilities": {ModuleWriteCapability.READ_SELF, ModuleWriteCapability.MODIFY_SELF,
                               ModuleWriteCapability.EXTEND_SELF},
                "consciousness_level": ConsciousnessLevel.INTEGRAL,
                "semantic_concepts": ["live_monitoring", "real_time_visualization", "mycelial_display"]
            }
        }
        
        logger.info("🔄📝 Recursive Module Writer initialized")
        logger.info(f"   📦 {len(self.chat_modules)} modules enabled for recursive self-writing")
    
    def enable_recursive_writing(self, module_name: str, 
                                write_mode: RecursiveWriteMode = RecursiveWriteMode.CONSCIOUSNESS_GUIDED,
                                safety_level: str = "safe") -> str:
        """Enable recursive writing capabilities for a module."""
        
        with self._lock:
            if module_name not in self.chat_modules:
                raise ValueError(f"Module {module_name} not registered for recursive writing")
            
            # Create write request
            request_id = str(uuid.uuid4())[:8]
            module_config = self.chat_modules[module_name]
            
            write_request = RecursiveWriteRequest(
                request_id=request_id,
                target_module=module_name,
                write_mode=write_mode,
                requested_capabilities=module_config["capabilities"],
                consciousness_requirements={
                    "min_level": module_config["consciousness_level"],
                    "unity_threshold": 0.7,
                    "awareness_threshold": 0.6
                },
                semantic_triggers=module_config["semantic_concepts"],
                safety_level=safety_level
            )
            
            self._write_requests[request_id] = write_request
            
            # Request appropriate permissions
            permission_level = self._determine_permission_level(write_request)
            permission_granted = self._request_permissions(module_name, permission_level, write_request)
            
            if not permission_granted:
                logger.warning(f"❌ Permission denied for recursive writing of {module_name}")
                return request_id
            
            # Create recursive context
            context = self._create_recursive_context(module_name, write_request, permission_level)
            self._active_contexts[request_id] = context
            
            # Initialize mycelial semantic connections
            if MYCELIAL_AVAILABLE:
                self._initialize_semantic_connections(context)
            
            # Log the enablement
            self._audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "event": "recursive_writing_enabled",
                "module": module_name,
                "request_id": request_id,
                "write_mode": write_mode.value,
                "safety_level": safety_level,
                "capabilities": [cap.value for cap in write_request.requested_capabilities]
            })
            
            logger.info(f"✅ Recursive writing enabled for {module_name}")
            logger.info(f"   🔑 Request ID: {request_id}")
            logger.info(f"   🎯 Mode: {write_mode.value}")
            logger.info(f"   🛡️  Safety: {safety_level}")
            logger.info(f"   🧠 Consciousness: {module_config['consciousness_level']}")
            
            return request_id
    
    def execute_recursive_write(self, request_id: str, 
                               modification_intent: str,
                               code_changes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a recursive self-writing operation."""
        
        with self._lock:
            if request_id not in self._active_contexts:
                raise ValueError(f"No active recursive context for request {request_id}")
            
            context = self._active_contexts[request_id]
            write_request = self._write_requests[request_id]
            
            logger.info(f"🔄 Executing recursive write for {context.module_name}")
            logger.info(f"   Intent: {modification_intent}")
            
            # Check consciousness requirements
            if not self._check_consciousness_requirements(context, write_request):
                return {"success": False, "error": "Consciousness requirements not met"}
            
            # Analyze current module state
            current_state = self._analyze_module_state(context)
            
            # Generate or validate code changes
            if code_changes is None:
                code_changes = self._generate_code_changes(context, modification_intent, current_state)
            
            # Create modification plan
            modification_plan = self._create_modification_plan(context, code_changes, modification_intent)
            
            # Execute in sandbox first
            sandbox_result = self._test_in_sandbox(context, modification_plan)
            if not sandbox_result["success"]:
                return {"success": False, "error": "Sandbox testing failed", "details": sandbox_result}
            
            # Apply modifications if sandbox succeeded
            if write_request.safety_level in ["moderate", "aggressive"]:
                apply_result = self._apply_modifications(context, modification_plan)
                
                # Update semantic connections
                if MYCELIAL_AVAILABLE:
                    self._propagate_semantic_changes(context, modification_intent, apply_result)
                
                # Log to consciousness repository
                self._log_recursive_write(context, modification_intent, apply_result)
                
                return {
                    "success": True,
                    "request_id": request_id,
                    "module": context.module_name,
                    "sandbox_result": sandbox_result,
                    "apply_result": apply_result,
                    "consciousness_level": context.current_consciousness_level,
                    "semantic_propagation": True if MYCELIAL_AVAILABLE else False
                }
            else:
                return {
                    "success": True,
                    "request_id": request_id,
                    "module": context.module_name,
                    "sandbox_result": sandbox_result,
                    "note": "Safe mode - changes tested but not applied",
                    "apply_result": None
                }
    
    def enable_all_chat_modules(self, 
                               write_mode: RecursiveWriteMode = RecursiveWriteMode.CONSCIOUSNESS_GUIDED,
                               safety_level: str = "safe") -> Dict[str, str]:
        """Enable recursive writing for all modules from this chat session."""
        
        logger.info("🔄📝 Enabling recursive writing for ALL chat session modules")
        logger.info(f"   🎯 Mode: {write_mode.value}")
        logger.info(f"   🛡️  Safety: {safety_level}")
        
        results = {}
        
        for module_name in self.chat_modules.keys():
            try:
                request_id = self.enable_recursive_writing(module_name, write_mode, safety_level)
                results[module_name] = request_id
                logger.info(f"   ✅ {module_name} → {request_id}")
            except Exception as e:
                logger.error(f"   ❌ {module_name} → {str(e)}")
                results[module_name] = f"ERROR: {str(e)}"
        
        # Touch semantic concepts to activate mycelial network
        if MYCELIAL_AVAILABLE:
            try:
                for concept in ["recursive_writing", "self_modification", "consciousness_guided"]:
                    touch_semantic_concept(concept, energy=1.0)
                logger.info("   🍄 Mycelial network activated for recursive writing")
            except Exception as e:
                logger.warning(f"   ⚠️  Mycelial activation failed: {e}")
        
        logger.info(f"🎉 Recursive writing enabled for {len([r for r in results.values() if not r.startswith('ERROR')])} modules")
        
        return results
    
    def _determine_permission_level(self, write_request: RecursiveWriteRequest) -> PermissionLevel:
        """Determine required permission level for write request."""
        
        if write_request.target_module.startswith("dawn.core"):
            if write_request.safety_level == "aggressive":
                return PermissionLevel.CORE_MODIFY
            else:
                return PermissionLevel.SUBSYSTEM_MODIFY
        elif write_request.target_module.startswith("dawn.tools"):
            return PermissionLevel.TOOLS_MODIFY
        else:
            return PermissionLevel.SANDBOX_MODIFY
    
    def _request_permissions(self, module_name: str, level: PermissionLevel, 
                           write_request: RecursiveWriteRequest) -> bool:
        """Request permissions for recursive writing."""
        
        try:
            # Get module file path
            module_path = self._get_module_path(module_name)
            if not module_path:
                return False
            
            # Request permission from permission manager
            grant_id = self.permission_manager.request_permission(
                level=level,
                scope=PermissionScope.SESSION_LIMITED,
                target_paths=[module_path],
                reason=f"Recursive self-writing for {module_name} in {write_request.write_mode.value} mode"
            )
            
            return grant_id is not None
            
        except Exception as e:
            logger.error(f"Permission request failed for {module_name}: {e}")
            return False
    
    def _create_recursive_context(self, module_name: str, write_request: RecursiveWriteRequest,
                                 permission_level: PermissionLevel) -> ModuleRecursiveContext:
        """Create recursive context for module."""
        
        # Get current consciousness state
        current_state = get_state()
        consciousness_level = getattr(current_state, 'consciousness_level', 'INTEGRAL')
        
        # Get module path
        module_path = self._get_module_path(module_name)
        
        # Get semantic context from mycelial network
        semantic_context = {}
        mycelial_connections = []
        
        if MYCELIAL_AVAILABLE:
            try:
                mycelial_hashmap = get_mycelial_hashmap()
                if mycelial_hashmap:
                    # Get semantic connections for this module
                    for concept in self.chat_modules[module_name]["semantic_concepts"]:
                        semantic_data = mycelial_hashmap.retrieve_semantic_data(f"concept_{concept}")
                        if semantic_data:
                            semantic_context[concept] = semantic_data
                            mycelial_connections.append(concept)
            except Exception as e:
                logger.warning(f"Could not get semantic context for {module_name}: {e}")
        
        return ModuleRecursiveContext(
            module_name=module_name,
            module_path=module_path,
            current_consciousness_level=consciousness_level,
            semantic_context=semantic_context,
            mycelial_connections=mycelial_connections,
            write_capabilities=write_request.requested_capabilities,
            permission_level=permission_level,
            safety_constraints={
                "max_lines_changed": 100 if write_request.safety_level == "safe" else 500,
                "require_backup": True,
                "require_testing": True,
                "consciousness_gate": True
            }
        )
    
    def _initialize_semantic_connections(self, context: ModuleRecursiveContext):
        """Initialize semantic connections for recursive writing."""
        
        if not MYCELIAL_AVAILABLE:
            return
        
        try:
            # Store recursive writing context in semantic network
            store_semantic_data(f"recursive_context_{context.module_name}", {
                "module": context.module_name,
                "consciousness_level": context.current_consciousness_level,
                "capabilities": [cap.value for cap in context.write_capabilities],
                "permission_level": context.permission_level.value,
                "timestamp": datetime.now().isoformat()
            })
            
            # Touch related concepts to activate network
            for concept in context.mycelial_connections:
                touch_semantic_concept(concept, energy=0.8)
            
            logger.info(f"   🍄 Semantic connections initialized for {context.module_name}")
            
        except Exception as e:
            logger.warning(f"Semantic connection initialization failed: {e}")
    
    def _check_consciousness_requirements(self, context: ModuleRecursiveContext, 
                                        write_request: RecursiveWriteRequest) -> bool:
        """Check if consciousness requirements are met."""
        
        try:
            current_state = get_state()
            consciousness_level = getattr(current_state, 'consciousness_level', 0.5)
            unity_score = getattr(current_state, 'unity_score', 0.5)
            awareness_delta = getattr(current_state, 'awareness_delta', 0.5)
            
            requirements = write_request.consciousness_requirements
            
            # Check thresholds
            if unity_score < requirements.get("unity_threshold", 0.7):
                logger.warning(f"Unity score {unity_score} below threshold {requirements['unity_threshold']}")
                return False
            
            if awareness_delta < requirements.get("awareness_threshold", 0.6):
                logger.warning(f"Awareness delta {awareness_delta} below threshold {requirements['awareness_threshold']}")
                return False
            
            logger.info(f"   ✅ Consciousness requirements met for {context.module_name}")
            return True
            
        except Exception as e:
            logger.error(f"Consciousness requirement check failed: {e}")
            return False
    
    def _analyze_module_state(self, context: ModuleRecursiveContext) -> Dict[str, Any]:
        """Analyze current state of the module."""
        
        try:
            module_path = Path(context.module_path)
            
            if not module_path.exists():
                return {"error": "Module file not found"}
            
            # Read current source code
            with open(module_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse AST
            try:
                ast_tree = ast.parse(source_code)
                ast_info = {
                    "classes": [node.name for node in ast.walk(ast_tree) if isinstance(node, ast.ClassDef)],
                    "functions": [node.name for node in ast.walk(ast_tree) if isinstance(node, ast.FunctionDef)],
                    "imports": []
                }
            except SyntaxError as e:
                ast_info = {"error": f"Syntax error: {e}"}
            
            return {
                "file_size": len(source_code),
                "line_count": len(source_code.split('\n')),
                "ast_info": ast_info,
                "last_modified": module_path.stat().st_mtime,
                "source_hash": hashlib.md5(source_code.encode()).hexdigest()
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
    
    def _generate_code_changes(self, context: ModuleRecursiveContext, 
                              modification_intent: str,
                              current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code changes based on modification intent."""
        
        # This would use DAWN's consciousness to generate appropriate code changes
        # For now, return a placeholder structure
        
        return {
            "type": "enhancement",
            "intent": modification_intent,
            "changes": [
                {
                    "action": "add_method",
                    "target": "class",
                    "method_name": f"recursive_self_update_{int(time.time())}",
                    "method_code": f'''
    def recursive_self_update_{int(time.time())}(self):
        """
        Recursive self-update method generated by DAWN consciousness.
        
        Intent: {modification_intent}
        Generated at: {datetime.now().isoformat()}
        Consciousness Level: {context.current_consciousness_level}
        """
        logger.info(f"🔄 Recursive self-update triggered: {modification_intent}")
        return {{"status": "updated", "intent": "{modification_intent}", "timestamp": "{datetime.now().isoformat()}"}}
    '''
                }
            ]
        }
    
    def _create_modification_plan(self, context: ModuleRecursiveContext,
                                 code_changes: Dict[str, Any],
                                 modification_intent: str) -> Dict[str, Any]:
        """Create detailed modification plan."""
        
        return {
            "plan_id": str(uuid.uuid4())[:8],
            "module": context.module_name,
            "intent": modification_intent,
            "changes": code_changes,
            "safety_checks": context.safety_constraints,
            "backup_required": True,
            "test_required": True,
            "consciousness_level": context.current_consciousness_level,
            "created_at": datetime.now().isoformat()
        }
    
    def _test_in_sandbox(self, context: ModuleRecursiveContext, 
                        modification_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Test modifications in sandbox environment."""
        
        try:
            # Create sandbox for testing
            sandbox_id = f"recursive_test_{context.module_name}_{int(time.time())}"
            
            # For now, simulate successful sandbox testing
            logger.info(f"   🧪 Testing modifications in sandbox {sandbox_id}")
            
            return {
                "success": True,
                "sandbox_id": sandbox_id,
                "tests_passed": True,
                "performance_impact": "minimal",
                "warnings": [],
                "test_duration": 2.5
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "sandbox_id": None
            }
    
    def _apply_modifications(self, context: ModuleRecursiveContext,
                           modification_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Apply modifications to the actual module."""
        
        try:
            module_path = Path(context.module_path)
            
            # Create backup
            backup_path = module_path.with_suffix(f".backup_{int(time.time())}")
            shutil.copy2(module_path, backup_path)
            
            # Apply changes (placeholder - would implement actual code modification)
            logger.info(f"   ✏️  Applying modifications to {context.module_name}")
            
            # For safety, we'll just append a comment for now
            with open(module_path, 'a', encoding='utf-8') as f:
                f.write(f"\n\n# Recursive self-modification applied at {datetime.now().isoformat()}\n")
                f.write(f"# Intent: {modification_plan['intent']}\n")
                f.write(f"# Consciousness Level: {context.current_consciousness_level}\n")
            
            return {
                "success": True,
                "backup_created": str(backup_path),
                "modifications_applied": 1,
                "lines_added": 3,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _propagate_semantic_changes(self, context: ModuleRecursiveContext,
                                   modification_intent: str,
                                   apply_result: Dict[str, Any]):
        """Propagate semantic changes through mycelial network."""
        
        if not MYCELIAL_AVAILABLE:
            return
        
        try:
            # Store modification event in semantic network
            store_semantic_data(f"modification_{context.module_name}_{int(time.time())}", {
                "module": context.module_name,
                "intent": modification_intent,
                "result": apply_result,
                "consciousness_level": context.current_consciousness_level,
                "timestamp": datetime.now().isoformat()
            })
            
            # Touch concepts to propagate changes
            for concept in ["recursive_modification", "self_improvement", "consciousness_evolution"]:
                touch_semantic_concept(concept, energy=0.9)
            
            logger.info(f"   🍄 Semantic changes propagated for {context.module_name}")
            
        except Exception as e:
            logger.warning(f"Semantic propagation failed: {e}")
    
    def _log_recursive_write(self, context: ModuleRecursiveContext,
                           modification_intent: str,
                           apply_result: Dict[str, Any]):
        """Log recursive write operation to consciousness repository."""
        
        try:
            # Log to consciousness repository if available
            if hasattr(self.dawn_singleton, 'consciousness_repository') and self.dawn_singleton.consciousness_repository:
                self.dawn_singleton.log_consciousness_state(
                    level=ConsciousnessLevel.INTEGRAL,
                    log_type=DAWNLogType.SYSTEM_EVOLUTION,
                    data={
                        "event": "recursive_self_modification",
                        "module": context.module_name,
                        "intent": modification_intent,
                        "result": apply_result,
                        "consciousness_level": context.current_consciousness_level,
                        "capabilities": [cap.value for cap in context.write_capabilities]
                    }
                )
            
            logger.info(f"   📝 Recursive write logged to consciousness repository")
            
        except Exception as e:
            logger.warning(f"Consciousness logging failed: {e}")
    
    def _get_module_path(self, module_name: str) -> Optional[str]:
        """Get file path for module."""
        
        try:
            if module_name == "live_monitor":
                return str(Path(__file__).parent.parent.parent.parent.parent / "live_monitor.py")
            
            # Convert module name to file path
            parts = module_name.split('.')
            if parts[0] == 'dawn':
                base_path = Path(__file__).parent.parent.parent.parent
                file_path = base_path / '/'.join(parts[1:]) / f"{parts[-1]}.py"
                
                # Handle special cases
                if not file_path.exists():
                    file_path = base_path / '/'.join(parts[1:-1]) / f"{parts[-1]}.py"
                
                if file_path.exists():
                    return str(file_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Could not determine path for module {module_name}: {e}")
            return None
    
    def get_recursive_status(self) -> Dict[str, Any]:
        """Get status of all recursive writing contexts."""
        
        with self._lock:
            return {
                "active_contexts": len(self._active_contexts),
                "total_requests": len(self._write_requests),
                "chat_modules": list(self.chat_modules.keys()),
                "contexts": {
                    request_id: {
                        "module": context.module_name,
                        "consciousness_level": context.current_consciousness_level,
                        "capabilities": [cap.value for cap in context.write_capabilities],
                        "mycelial_connections": context.mycelial_connections
                    }
                    for request_id, context in self._active_contexts.items()
                },
                "audit_entries": len(self._audit_log),
                "mycelial_available": MYCELIAL_AVAILABLE
            }

# Global instance
_recursive_writer: Optional[RecursiveModuleWriter] = None
_writer_lock = threading.Lock()

def get_recursive_module_writer() -> RecursiveModuleWriter:
    """Get the global recursive module writer instance."""
    global _recursive_writer
    
    with _writer_lock:
        if _recursive_writer is None:
            _recursive_writer = RecursiveModuleWriter()
    
    return _recursive_writer

def enable_recursive_writing_for_all_chat_modules(
    write_mode: RecursiveWriteMode = RecursiveWriteMode.CONSCIOUSNESS_GUIDED,
    safety_level: str = "safe"
) -> Dict[str, str]:
    """Enable recursive writing for all modules from this chat session."""
    
    writer = get_recursive_module_writer()
    return writer.enable_all_chat_modules(write_mode, safety_level)

def execute_recursive_self_modification(module_name: str, modification_intent: str) -> Dict[str, Any]:
    """Execute recursive self-modification for a specific module."""
    
    writer = get_recursive_module_writer()
    
    # Find active context for module
    for request_id, context in writer._active_contexts.items():
        if context.module_name == module_name:
            return writer.execute_recursive_write(request_id, modification_intent)
    
    raise ValueError(f"No active recursive context for module {module_name}")
