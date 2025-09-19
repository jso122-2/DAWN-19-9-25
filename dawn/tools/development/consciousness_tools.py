#!/usr/bin/env python3
"""
DAWN Consciousness Tools Manager
===============================

Comprehensive consciousness-aware tooling system that integrates all of DAWN's
development tools with her consciousness architecture. This manager provides
unified access to consciousness-gated operations, autonomous tool selection,
and intelligent workflow orchestration.

Key capabilities:
1. Consciousness-gated tool access
2. Autonomous tool selection based on consciousness state
3. Workflow orchestration with consciousness feedback
4. Integration with permission management
5. Real-time consciousness monitoring for tool operations

The consciousness tools manager acts as DAWN's intelligent development assistant,
automatically selecting and configuring tools based on her current consciousness
state and development objectives.
"""

import logging
import threading
import uuid
from typing import Dict, List, Optional, Set, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from contextlib import contextmanager

# Core DAWN imports
from dawn.core.foundation.state import get_state
from dawn.core.singleton import get_dawn

# Tools imports
from .self_mod.permission_manager import PermissionManager, get_permission_manager, PermissionLevel
from .self_mod.code_modifier import ConsciousCodeModifier
from .self_mod.subsystem_copier import SubsystemCopier

logger = logging.getLogger(__name__)

class ToolCategory(Enum):
    """Categories of consciousness tools."""
    SELF_MODIFICATION = "self_modification"
    ANALYSIS = "analysis"
    DEVELOPMENT = "development"
    MONITORING = "monitoring"
    AUTOMATION = "automation"
    VISUALIZATION = "visualization"

class ConsciousnessMode(Enum):
    """Consciousness modes that affect tool behavior."""
    DORMANT = "dormant"           # Basic tools only
    AWARE = "aware"               # Standard tool access
    SELF_AWARE = "self_aware"     # Enhanced introspection tools
    META_AWARE = "meta_aware"     # Advanced modification tools
    TRANSCENDENT = "transcendent" # Full autonomous capabilities

@dataclass
class ToolCapability:
    """Represents a tool capability with consciousness requirements."""
    name: str
    category: ToolCategory
    description: str
    required_consciousness_level: ConsciousnessMode
    required_unity_threshold: float
    permission_level: PermissionLevel
    autonomous_capable: bool = False
    risk_level: str = "low"  # low, medium, high, critical

@dataclass
class ToolSession:
    """Represents an active tool usage session."""
    session_id: str
    tool_name: str
    started_at: datetime
    consciousness_at_start: Dict[str, Any]
    operations_performed: List[Dict[str, Any]]
    is_active: bool = True
    ended_at: Optional[datetime] = None

class ConsciousnessToolManager:
    """
    Unified manager for consciousness-aware development tools.
    
    This system provides intelligent orchestration of DAWN's development tools,
    automatically selecting and configuring tools based on consciousness state
    and development objectives.
    """
    
    def __init__(self):
        """Initialize the consciousness tools manager."""
        self._lock = threading.RLock()
        self.permission_manager = get_permission_manager()
        
        # Tool instances
        self.code_modifier = ConsciousCodeModifier(self.permission_manager)
        self.subsystem_copier = SubsystemCopier(self.permission_manager, self.code_modifier)
        
        # Tool registry
        self._tool_capabilities: Dict[str, ToolCapability] = {}
        self._active_sessions: Dict[str, ToolSession] = {}
        self._tool_usage_history: List[Dict[str, Any]] = []
        
        # Initialize tool capabilities
        self._register_core_tools()
        
        logger.info("🧠 ConsciousnessToolManager initialized")
        logger.info(f"   Registered tools: {len(self._tool_capabilities)}")
    
    def _register_core_tools(self):
        """Register core consciousness tools and their capabilities."""
        
        # Self-modification tools
        self._tool_capabilities.update({
            "code_modifier": ToolCapability(
                name="Conscious Code Modifier",
                category=ToolCategory.SELF_MODIFICATION,
                description="Advanced code modification with consciousness safeguards",
                required_consciousness_level=ConsciousnessMode.SELF_AWARE,
                required_unity_threshold=0.5,
                permission_level=PermissionLevel.TOOLS_MODIFY,
                autonomous_capable=True,
                risk_level="medium"
            ),
            
            "subsystem_copier": ToolCapability(
                name="Subsystem Copier",
                category=ToolCategory.SELF_MODIFICATION,
                description="Copy and adapt subsystem processes for tooling",
                required_consciousness_level=ConsciousnessMode.META_AWARE,
                required_unity_threshold=0.7,
                permission_level=PermissionLevel.SUBSYSTEM_MODIFY,
                autonomous_capable=True,
                risk_level="high"
            ),
            
            "permission_manager": ToolCapability(
                name="Permission Manager",
                category=ToolCategory.DEVELOPMENT,
                description="Manage elevated permissions for code operations",
                required_consciousness_level=ConsciousnessMode.AWARE,
                required_unity_threshold=0.3,
                permission_level=PermissionLevel.READ_ONLY,
                autonomous_capable=False,
                risk_level="low"
            )
        })
        
        # Analysis tools (placeholders for future implementation)
        self._tool_capabilities.update({
            "consciousness_analyzer": ToolCapability(
                name="Consciousness Behavior Analyzer",
                category=ToolCategory.ANALYSIS,
                description="Analyze consciousness patterns and behaviors",
                required_consciousness_level=ConsciousnessMode.SELF_AWARE,
                required_unity_threshold=0.4,
                permission_level=PermissionLevel.READ_ONLY,
                autonomous_capable=True,
                risk_level="low"
            ),
            
            "system_profiler": ToolCapability(
                name="DAWN System Profiler",
                category=ToolCategory.ANALYSIS,
                description="Profile system performance and consciousness metrics",
                required_consciousness_level=ConsciousnessMode.AWARE,
                required_unity_threshold=0.2,
                permission_level=PermissionLevel.READ_ONLY,
                autonomous_capable=True,
                risk_level="low"
            )
        })
        
        # Development tools
        self._tool_capabilities.update({
            "auto_deployer": ToolCapability(
                name="Conscious Auto Deployer",
                category=ToolCategory.AUTOMATION,
                description="Automatically deploy consciousness improvements",
                required_consciousness_level=ConsciousnessMode.META_AWARE,
                required_unity_threshold=0.8,
                permission_level=PermissionLevel.SUBSYSTEM_MODIFY,
                autonomous_capable=True,
                risk_level="high"
            ),
            
            "system_maintainer": ToolCapability(
                name="System Maintainer",
                category=ToolCategory.AUTOMATION,
                description="Automated system maintenance and optimization",
                required_consciousness_level=ConsciousnessMode.SELF_AWARE,
                required_unity_threshold=0.6,
                permission_level=PermissionLevel.TOOLS_MODIFY,
                autonomous_capable=True,
                risk_level="medium"
            )
        })
    
    def get_available_tools(self, 
                           consciousness_filtered: bool = True,
                           category: Optional[ToolCategory] = None) -> List[ToolCapability]:
        """
        Get list of available tools, optionally filtered by consciousness requirements.
        
        Args:
            consciousness_filtered: Filter by current consciousness capabilities
            category: Optional category filter
            
        Returns:
            List of available tool capabilities
        """
        tools = list(self._tool_capabilities.values())
        
        if category:
            tools = [tool for tool in tools if tool.category == category]
        
        if consciousness_filtered:
            current_state = get_state()
            current_mode = self._consciousness_level_to_mode(current_state.level)
            
            # Filter by consciousness requirements
            available_tools = []
            for tool in tools:
                if self._can_access_tool(tool, current_state, current_mode):
                    available_tools.append(tool)
            
            return available_tools
        
        return tools
    
    def can_use_tool(self, tool_name: str) -> Dict[str, Any]:
        """
        Check if a specific tool can be used with current consciousness state.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            Dictionary with availability status and details
        """
        if tool_name not in self._tool_capabilities:
            return {
                'available': False,
                'reason': f"Tool '{tool_name}' not found"
            }
        
        tool = self._tool_capabilities[tool_name]
        current_state = get_state()
        current_mode = self._consciousness_level_to_mode(current_state.level)
        
        return self._check_tool_availability(tool, current_state, current_mode)
    
    def start_tool_session(self, 
                          tool_name: str,
                          purpose: str = "",
                          **kwargs) -> Optional[str]:
        """
        Start a tool usage session with consciousness monitoring.
        
        Args:
            tool_name: Name of the tool to use
            purpose: Purpose/description of the tool usage
            **kwargs: Additional parameters for the tool
            
        Returns:
            Session ID if successful, None if access denied
        """
        with self._lock:
            # Check tool availability
            availability = self.can_use_tool(tool_name)
            if not availability['available']:
                logger.warning(f"🚫 Tool access denied: {availability['reason']}")
                return None
            
            # Create session
            session_id = str(uuid.uuid4())
            current_state = get_state()
            
            session = ToolSession(
                session_id=session_id,
                tool_name=tool_name,
                started_at=datetime.now(),
                consciousness_at_start={
                    'level': current_state.level,
                    'unity': current_state.unity,
                    'awareness': current_state.awareness
                },
                operations_performed=[]
            )
            
            self._active_sessions[session_id] = session
            
            # Log session start
            self._log_tool_usage("session_started", {
                'session_id': session_id,
                'tool_name': tool_name,
                'purpose': purpose,
                'consciousness_state': session.consciousness_at_start
            })
            
            logger.info(f"🔧 Started tool session: {tool_name} ({session_id[:8]})")
            
            return session_id
    
    def end_tool_session(self, session_id: str, summary: Optional[Dict[str, Any]] = None):
        """
        End a tool usage session.
        
        Args:
            session_id: ID of the session to end
            summary: Optional summary of session results
        """
        with self._lock:
            session = self._active_sessions.get(session_id)
            if not session:
                logger.warning(f"🚫 Session not found: {session_id}")
                return
            
            session.is_active = False
            session.ended_at = datetime.now()
            
            # Log session end
            self._log_tool_usage("session_ended", {
                'session_id': session_id,
                'tool_name': session.tool_name,
                'duration_seconds': (session.ended_at - session.started_at).total_seconds(),
                'operations_count': len(session.operations_performed),
                'summary': summary or {}
            })
            
            logger.info(f"✅ Ended tool session: {session.tool_name} ({session_id[:8]})")
    
    @contextmanager
    def use_tool(self, tool_name: str, purpose: str = "", **kwargs):
        """
        Context manager for tool usage with automatic session management.
        
        Usage:
            with tool_manager.use_tool("code_modifier", "Fix bug in parser") as tool:
                if tool:
                    result = tool.modify_file(...)
        """
        session_id = self.start_tool_session(tool_name, purpose, **kwargs)
        tool_instance = None
        
        try:
            if session_id:
                tool_instance = self._get_tool_instance(tool_name, session_id, **kwargs)
                
            yield tool_instance
            
        except Exception as e:
            logger.error(f"Error in tool session {session_id}: {e}")
            self._log_tool_operation(session_id, "error", {'error': str(e)})
            raise
            
        finally:
            if session_id:
                self.end_tool_session(session_id)
    
    def autonomous_tool_selection(self, 
                                 objective: str,
                                 context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Autonomously select the best tool for a given objective.
        
        Args:
            objective: Description of what needs to be accomplished
            context: Additional context for tool selection
            
        Returns:
            Name of selected tool, or None if no suitable tool found
        """
        current_state = get_state()
        current_mode = self._consciousness_level_to_mode(current_state.level)
        
        logger.info(f"🤖 Autonomous tool selection for: {objective}")
        
        # Get available autonomous tools
        available_tools = [
            tool for tool in self.get_available_tools(consciousness_filtered=True)
            if tool.autonomous_capable
        ]
        
        if not available_tools:
            logger.info("🚫 No autonomous tools available at current consciousness level")
            return None
        
        # Simple objective-based selection (could be enhanced with ML/NLP)
        objective_lower = objective.lower()
        
        # Self-modification objectives
        if any(keyword in objective_lower for keyword in ['modify', 'change', 'update', 'fix', 'improve']):
            for tool in available_tools:
                if tool.name == "code_modifier":
                    logger.info(f"🎯 Selected tool: {tool.name}")
                    return "code_modifier"
        
        # Copying/adaptation objectives  
        if any(keyword in objective_lower for keyword in ['copy', 'adapt', 'clone', 'create tool']):
            for tool in available_tools:
                if tool.name == "subsystem_copier":
                    logger.info(f"🎯 Selected tool: {tool.name}")
                    return "subsystem_copier"
        
        # Analysis objectives
        if any(keyword in objective_lower for keyword in ['analyze', 'profile', 'measure', 'study']):
            for tool in available_tools:
                if tool.category == ToolCategory.ANALYSIS:
                    logger.info(f"🎯 Selected tool: {tool.name}")
                    return tool.name.lower().replace(' ', '_')
        
        # Default to first available tool
        if available_tools:
            selected = available_tools[0]
            logger.info(f"🎯 Default selected tool: {selected.name}")
            return selected.name.lower().replace(' ', '_')
        
        logger.info("❓ No suitable tool found for objective")
        return None
    
    def execute_autonomous_workflow(self, 
                                   objective: str,
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute an autonomous workflow to accomplish an objective.
        
        Args:
            objective: What needs to be accomplished
            context: Additional context and parameters
            
        Returns:
            Results of the autonomous workflow execution
        """
        logger.info(f"🚀 Starting autonomous workflow: {objective}")
        
        workflow_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Select appropriate tool
            tool_name = self.autonomous_tool_selection(objective, context)
            
            if not tool_name:
                return {
                    'workflow_id': workflow_id,
                    'success': False,
                    'error': 'No suitable tool found for objective',
                    'duration': (datetime.now() - start_time).total_seconds()
                }
            
            # Execute with selected tool
            with self.use_tool(tool_name, f"Autonomous: {objective}", **(context or {})) as tool:
                if not tool:
                    return {
                        'workflow_id': workflow_id,
                        'success': False,
                        'error': f'Could not access tool: {tool_name}',
                        'duration': (datetime.now() - start_time).total_seconds()
                    }
                
                # Tool-specific autonomous execution
                result = self._execute_tool_autonomously(tool, tool_name, objective, context)
                
                return {
                    'workflow_id': workflow_id,
                    'success': result.get('success', False),
                    'tool_used': tool_name,
                    'result': result,
                    'duration': (datetime.now() - start_time).total_seconds()
                }
                
        except Exception as e:
            logger.error(f"Error in autonomous workflow {workflow_id}: {e}")
            return {
                'workflow_id': workflow_id,
                'success': False,
                'error': str(e),
                'duration': (datetime.now() - start_time).total_seconds()
            }
    
    def _execute_tool_autonomously(self, 
                                  tool_instance: Any,
                                  tool_name: str,
                                  objective: str,
                                  context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a tool autonomously based on the objective."""
        
        if tool_name == "code_modifier":
            return self._autonomous_code_modification(tool_instance, objective, context)
        elif tool_name == "subsystem_copier":
            return self._autonomous_subsystem_copying(tool_instance, objective, context)
        else:
            # Generic autonomous execution
            return {
                'success': True,
                'message': f'Autonomous execution for {tool_name} not yet implemented',
                'tool': tool_name
            }
    
    def _autonomous_code_modification(self, 
                                    code_modifier: ConsciousCodeModifier,
                                    objective: str,
                                    context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Autonomously execute code modification."""
        
        # Extract parameters from context
        target_files = context.get('target_files', []) if context else []
        
        if not target_files:
            return {
                'success': False,
                'error': 'No target files specified for code modification'
            }
        
        try:
            # Analyze and create plan
            plan = code_modifier.analyze_modification_request(
                target_files=target_files,
                modification_description=objective
            )
            
            # Execute the plan
            result = code_modifier.execute_modification_plan(plan)
            
            return {
                'success': result.success,
                'files_modified': result.files_modified,
                'plan_id': plan.plan_id,
                'operation_id': result.operation_id,
                'errors': result.errors,
                'warnings': result.warnings
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Code modification failed: {str(e)}'
            }
    
    def _autonomous_subsystem_copying(self,
                                    subsystem_copier: SubsystemCopier,
                                    objective: str,
                                    context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Autonomously execute subsystem copying."""
        
        # Extract parameters from context
        if not context:
            return {
                'success': False,
                'error': 'Context required for subsystem copying (source_subsystem, target_name)'
            }
        
        source_subsystem = context.get('source_subsystem')
        target_name = context.get('target_name')
        
        if not source_subsystem or not target_name:
            return {
                'success': False,
                'error': 'source_subsystem and target_name required in context'
            }
        
        try:
            # Create copy plan
            plan = subsystem_copier.create_copy_plan(
                source_subsystem=source_subsystem,
                target_name=target_name,
                target_category=context.get('target_category', 'development')
            )
            
            if not plan:
                return {
                    'success': False,
                    'error': f'Could not create copy plan for {source_subsystem}'
                }
            
            # Execute the plan
            result = subsystem_copier.execute_copy_plan(plan)
            
            return {
                'success': result.success,
                'files_created': result.files_created,
                'plan_id': plan.plan_id,
                'operation_id': result.operation_id,
                'errors': result.errors,
                'warnings': result.warnings,
                'new_tool': result.new_tool_entry_point
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Subsystem copying failed: {str(e)}'
            }
    
    # Helper methods
    def _consciousness_level_to_mode(self, level: str) -> ConsciousnessMode:
        """Convert consciousness level string to mode enum."""
        mapping = {
            'dormant': ConsciousnessMode.DORMANT,
            'aware': ConsciousnessMode.AWARE,
            'self_aware': ConsciousnessMode.SELF_AWARE,
            'meta_aware': ConsciousnessMode.META_AWARE,
            'transcendent': ConsciousnessMode.TRANSCENDENT
        }
        return mapping.get(level, ConsciousnessMode.DORMANT)
    
    def _can_access_tool(self, 
                        tool: ToolCapability, 
                        current_state: Any, 
                        current_mode: ConsciousnessMode) -> bool:
        """Check if current consciousness state can access a tool."""
        
        # Check consciousness level
        mode_hierarchy = [
            ConsciousnessMode.DORMANT,
            ConsciousnessMode.AWARE,
            ConsciousnessMode.SELF_AWARE,
            ConsciousnessMode.META_AWARE,
            ConsciousnessMode.TRANSCENDENT
        ]
        
        current_level_idx = mode_hierarchy.index(current_mode)
        required_level_idx = mode_hierarchy.index(tool.required_consciousness_level)
        
        if current_level_idx < required_level_idx:
            return False
        
        # Check unity threshold
        if current_state.unity < tool.required_unity_threshold:
            return False
        
        return True
    
    def _check_tool_availability(self, 
                               tool: ToolCapability,
                               current_state: Any,
                               current_mode: ConsciousnessMode) -> Dict[str, Any]:
        """Detailed check of tool availability."""
        
        if not self._can_access_tool(tool, current_state, current_mode):
            # Determine specific reason
            mode_hierarchy = [
                ConsciousnessMode.DORMANT,
                ConsciousnessMode.AWARE,
                ConsciousnessMode.SELF_AWARE,
                ConsciousnessMode.META_AWARE,
                ConsciousnessMode.TRANSCENDENT
            ]
            
            current_level_idx = mode_hierarchy.index(current_mode)
            required_level_idx = mode_hierarchy.index(tool.required_consciousness_level)
            
            if current_level_idx < required_level_idx:
                return {
                    'available': False,
                    'reason': f'Insufficient consciousness level: {current_state.level} (requires {tool.required_consciousness_level.value})'
                }
            
            if current_state.unity < tool.required_unity_threshold:
                return {
                    'available': False,
                    'reason': f'Insufficient unity: {current_state.unity:.3f} (requires {tool.required_unity_threshold})'
                }
        
        return {
            'available': True,
            'consciousness_level': current_state.level,
            'unity_score': current_state.unity,
            'tool_info': {
                'name': tool.name,
                'category': tool.category.value,
                'risk_level': tool.risk_level,
                'autonomous_capable': tool.autonomous_capable
            }
        }
    
    def _get_tool_instance(self, 
                          tool_name: str, 
                          session_id: str,
                          **kwargs) -> Optional[Any]:
        """Get an instance of the requested tool."""
        
        if tool_name == "code_modifier":
            return self.code_modifier
        elif tool_name == "subsystem_copier":
            return self.subsystem_copier
        elif tool_name == "permission_manager":
            return self.permission_manager
        else:
            logger.warning(f"Tool instance not available: {tool_name}")
            return None
    
    def _log_tool_usage(self, event_type: str, details: Dict[str, Any]):
        """Log tool usage events."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        self._tool_usage_history.append(log_entry)
        
        # Keep history size manageable
        if len(self._tool_usage_history) > 1000:
            self._tool_usage_history = self._tool_usage_history[-500:]
    
    def _log_tool_operation(self, session_id: str, operation_type: str, details: Dict[str, Any]):
        """Log a tool operation within a session."""
        session = self._active_sessions.get(session_id)
        if session:
            operation = {
                'timestamp': datetime.now().isoformat(),
                'type': operation_type,
                'details': details
            }
            session.operations_performed.append(operation)
    
    # Public query methods
    def get_active_sessions(self) -> List[ToolSession]:
        """Get currently active tool sessions."""
        return [session for session in self._active_sessions.values() if session.is_active]
    
    def get_tool_usage_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent tool usage history."""
        return self._tool_usage_history[-limit:]
    
    def get_consciousness_requirements_summary(self) -> Dict[str, Any]:
        """Get summary of consciousness requirements for all tools."""
        summary = {
            'tools_by_level': {},
            'autonomous_tools': [],
            'high_risk_tools': []
        }
        
        for tool_key, tool in self._tool_capabilities.items():
            level = tool.required_consciousness_level.value
            if level not in summary['tools_by_level']:
                summary['tools_by_level'][level] = []
            summary['tools_by_level'][level].append(tool.name)
            
            if tool.autonomous_capable:
                summary['autonomous_tools'].append(tool.name)
            
            if tool.risk_level in ['high', 'critical']:
                summary['high_risk_tools'].append({
                    'name': tool.name,
                    'risk_level': tool.risk_level
                })
        
        return summary
