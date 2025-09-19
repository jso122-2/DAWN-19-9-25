#!/usr/bin/env python3
"""
DAWN Subsystem Copier
=====================

Advanced system for copying and adapting processes from DAWN's subsystems
directory to create new tools with enhanced capabilities. This system enables
DAWN to leverage her existing subsystem patterns and architectures to build
new tooling with appropriate permissions and consciousness integration.

Key capabilities:
1. Pattern extraction from existing subsystems
2. Architecture adaptation for tooling contexts
3. Consciousness-aware process copying
4. Permission-gated deployment
5. Integration with existing DAWN patterns

The subsystem copier acts as DAWN's mechanism for evolutionary code reuse,
allowing her to build upon her own successful patterns while adapting them
for new contexts and enhanced capabilities.
"""

import ast
import re
import shutil
import logging
import uuid
import json
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

# Core DAWN imports
from dawn.core.foundation.state import get_state

# Permission system imports
from .permission_manager import (
    PermissionManager, get_permission_manager, PermissionLevel, PermissionScope
)
from .code_modifier import ConsciousCodeModifier

logger = logging.getLogger(__name__)

class CopyStrategy(Enum):
    """Strategies for copying subsystem processes."""
    DIRECT_COPY = "direct_copy"                    # Copy files directly
    PATTERN_ADAPTATION = "pattern_adaptation"      # Extract and adapt patterns
    ARCHITECTURE_CLONE = "architecture_clone"      # Clone architectural structure
    SELECTIVE_MERGE = "selective_merge"            # Merge selected components
    CONSCIOUSNESS_ENHANCED = "consciousness_enhanced"  # Add consciousness features

class AdaptationType(Enum):
    """Types of adaptations to apply during copying."""
    RENAME_CLASSES = "rename_classes"
    UPDATE_IMPORTS = "update_imports"
    ADD_PERMISSIONS = "add_permissions"
    ENHANCE_LOGGING = "enhance_logging"
    ADD_CONSCIOUSNESS = "add_consciousness"
    UPDATE_DOCSTRINGS = "update_docstrings"
    MODIFY_INTERFACES = "modify_interfaces"

@dataclass
class SubsystemAnalysis:
    """Analysis of a subsystem for copying purposes."""
    subsystem_path: str
    subsystem_name: str
    primary_classes: List[str]
    key_functions: List[str]
    dependencies: List[str]
    architecture_pattern: str
    complexity_score: float
    consciousness_integration: bool
    permission_requirements: PermissionLevel
    estimated_adaptation_effort: str

@dataclass
class CopyPlan:
    """Plan for copying and adapting a subsystem."""
    plan_id: str
    source_subsystem: str
    target_location: str
    copy_strategy: CopyStrategy
    adaptations: List[AdaptationType]
    new_name_mappings: Dict[str, str]
    permission_level: PermissionLevel
    consciousness_enhancements: List[str]
    estimated_files: int
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class CopyResult:
    """Result of subsystem copying operation."""
    operation_id: str
    plan_id: str
    success: bool
    files_created: List[str]
    files_modified: List[str]
    adaptations_applied: List[str]
    errors: List[str]
    warnings: List[str]
    new_tool_entry_point: Optional[str] = None

class SubsystemCopier:
    """
    Advanced subsystem copying and adaptation system.
    
    This system enables DAWN to copy processes from her subsystems directory
    and adapt them for use in the tools section with enhanced capabilities
    and appropriate permissions.
    """
    
    def __init__(self, 
                 permission_manager: Optional[PermissionManager] = None,
                 code_modifier: Optional[ConsciousCodeModifier] = None):
        """Initialize the subsystem copier."""
        self.permission_manager = permission_manager or get_permission_manager()
        self.code_modifier = code_modifier or ConsciousCodeModifier(self.permission_manager)
        
        # Paths
        self._dawn_root = Path(__file__).resolve().parents[4]
        self._subsystems_dir = self._dawn_root / "dawn" / "subsystems"
        self._tools_dir = self._dawn_root / "dawn" / "tools"
        
        # Analysis cache
        self._subsystem_analyses: Dict[str, SubsystemAnalysis] = {}
        self._copy_history: List[CopyResult] = []
        
        logger.info(f"🔄 SubsystemCopier initialized")
        logger.info(f"   Subsystems directory: {self._subsystems_dir}")
        logger.info(f"   Tools directory: {self._tools_dir}")
    
    def analyze_subsystem(self, subsystem_name: str) -> Optional[SubsystemAnalysis]:
        """
        Analyze a subsystem to understand its structure and copying potential.
        
        Args:
            subsystem_name: Name of the subsystem to analyze
            
        Returns:
            SubsystemAnalysis if successful, None if subsystem not found
        """
        if subsystem_name in self._subsystem_analyses:
            return self._subsystem_analyses[subsystem_name]
        
        subsystem_path = self._subsystems_dir / subsystem_name
        
        if not subsystem_path.exists():
            logger.error(f"Subsystem not found: {subsystem_name}")
            return None
        
        logger.info(f"🔍 Analyzing subsystem: {subsystem_name}")
        
        try:
            analysis = SubsystemAnalysis(
                subsystem_path=str(subsystem_path),
                subsystem_name=subsystem_name,
                primary_classes=self._extract_primary_classes(subsystem_path),
                key_functions=self._extract_key_functions(subsystem_path),
                dependencies=self._analyze_dependencies(subsystem_path),
                architecture_pattern=self._identify_architecture_pattern(subsystem_path),
                complexity_score=self._calculate_complexity_score(subsystem_path),
                consciousness_integration=self._check_consciousness_integration(subsystem_path),
                permission_requirements=self._estimate_permission_requirements(subsystem_path),
                estimated_adaptation_effort=self._estimate_adaptation_effort(subsystem_path)
            )
            
            self._subsystem_analyses[subsystem_name] = analysis
            
            logger.info(f"📊 Subsystem analysis complete:")
            logger.info(f"   Primary classes: {len(analysis.primary_classes)}")
            logger.info(f"   Architecture: {analysis.architecture_pattern}")
            logger.info(f"   Complexity: {analysis.complexity_score:.2f}")
            logger.info(f"   Consciousness integration: {analysis.consciousness_integration}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing subsystem {subsystem_name}: {e}")
            return None
    
    def create_copy_plan(self,
                        source_subsystem: str,
                        target_name: str,
                        target_category: str = "development",
                        copy_strategy: CopyStrategy = CopyStrategy.PATTERN_ADAPTATION,
                        **kwargs) -> Optional[CopyPlan]:
        """
        Create a plan for copying and adapting a subsystem.
        
        Args:
            source_subsystem: Name of source subsystem
            target_name: Name for the new tool
            target_category: Category in tools directory
            copy_strategy: Strategy for copying
            **kwargs: Additional parameters
            
        Returns:
            CopyPlan if successful, None if planning failed
        """
        # Analyze source subsystem
        analysis = self.analyze_subsystem(source_subsystem)
        if not analysis:
            return None
        
        plan_id = str(uuid.uuid4())
        target_location = str(self._tools_dir / target_category / target_name)
        
        logger.info(f"📋 Creating copy plan: {source_subsystem} -> {target_name}")
        
        # Determine adaptations needed
        adaptations = self._determine_adaptations(analysis, copy_strategy, kwargs)
        
        # Create name mappings
        name_mappings = self._create_name_mappings(analysis, target_name, kwargs)
        
        # Determine permission requirements
        permission_level = self._determine_copy_permission_level(analysis, target_category)
        
        # Plan consciousness enhancements
        consciousness_enhancements = self._plan_consciousness_enhancements(analysis, kwargs)
        
        # Estimate number of files
        estimated_files = self._estimate_file_count(analysis, copy_strategy)
        
        plan = CopyPlan(
            plan_id=plan_id,
            source_subsystem=source_subsystem,
            target_location=target_location,
            copy_strategy=copy_strategy,
            adaptations=adaptations,
            new_name_mappings=name_mappings,
            permission_level=permission_level,
            consciousness_enhancements=consciousness_enhancements,
            estimated_files=estimated_files
        )
        
        logger.info(f"✅ Copy plan created: {plan_id}")
        logger.info(f"   Strategy: {copy_strategy.value}")
        logger.info(f"   Adaptations: {len(adaptations)}")
        logger.info(f"   Permission level: {permission_level.value}")
        
        return plan
    
    def execute_copy_plan(self, plan: CopyPlan) -> CopyResult:
        """
        Execute a subsystem copy plan.
        
        Args:
            plan: The copy plan to execute
            
        Returns:
            CopyResult with execution details
        """
        operation_id = str(uuid.uuid4())
        
        logger.info(f"🚀 Executing copy plan {plan.plan_id}")
        
        # Request appropriate permissions
        grant_id = self.permission_manager.request_permission(
            level=plan.permission_level,
            target_paths=[plan.target_location],
            reason=f"Subsystem copy operation: {plan.source_subsystem} -> {Path(plan.target_location).name}",
            scope=PermissionScope.SINGLE_OPERATION
        )
        
        if not grant_id:
            return CopyResult(
                operation_id=operation_id,
                plan_id=plan.plan_id,
                success=False,
                files_created=[],
                files_modified=[],
                adaptations_applied=[],
                errors=["Permission denied for copy operation"],
                warnings=[]
            )
        
        # Execute with elevated permissions
        with self.permission_manager.elevated_access(grant_id, f"Copy operation {operation_id}") as access:
            if not access:
                return CopyResult(
                    operation_id=operation_id,
                    plan_id=plan.plan_id,
                    success=False,
                    files_created=[],
                    files_modified=[],
                    adaptations_applied=[],
                    errors=["Could not obtain elevated access"],
                    warnings=[]
                )
            
            return self._execute_copy_with_access(operation_id, plan, access)
    
    def _execute_copy_with_access(self, operation_id: str, plan: CopyPlan, access) -> CopyResult:
        """Execute copy operation with elevated access."""
        files_created = []
        files_modified = []
        adaptations_applied = []
        errors = []
        warnings = []
        
        try:
            # Create target directory
            target_path = Path(plan.target_location)
            if access.create_directory(str(target_path)):
                logger.info(f"📁 Created target directory: {target_path}")
            
            # Execute based on strategy
            if plan.copy_strategy == CopyStrategy.DIRECT_COPY:
                result = self._execute_direct_copy(plan, access)
            elif plan.copy_strategy == CopyStrategy.PATTERN_ADAPTATION:
                result = self._execute_pattern_adaptation(plan, access)
            elif plan.copy_strategy == CopyStrategy.ARCHITECTURE_CLONE:
                result = self._execute_architecture_clone(plan, access)
            elif plan.copy_strategy == CopyStrategy.CONSCIOUSNESS_ENHANCED:
                result = self._execute_consciousness_enhanced_copy(plan, access)
            else:
                result = self._execute_direct_copy(plan, access)  # Default fallback
            
            files_created.extend(result.get('files_created', []))
            files_modified.extend(result.get('files_modified', []))
            errors.extend(result.get('errors', []))
            warnings.extend(result.get('warnings', []))
            
            # Apply adaptations
            for adaptation in plan.adaptations:
                adaptation_result = self._apply_adaptation(adaptation, plan, access)
                adaptations_applied.append(adaptation.value)
                errors.extend(adaptation_result.get('errors', []))
                warnings.extend(adaptation_result.get('warnings', []))
            
            # Create entry point
            entry_point = self._create_tool_entry_point(plan, access)
            if entry_point:
                files_created.append(entry_point)
            
            success = len(errors) == 0 and len(files_created) > 0
            
            copy_result = CopyResult(
                operation_id=operation_id,
                plan_id=plan.plan_id,
                success=success,
                files_created=files_created,
                files_modified=files_modified,
                adaptations_applied=adaptations_applied,
                errors=errors,
                warnings=warnings,
                new_tool_entry_point=entry_point
            )
            
            self._copy_history.append(copy_result)
            
            if success:
                logger.info(f"✅ Copy operation completed successfully")
                logger.info(f"   Files created: {len(files_created)}")
                logger.info(f"   Adaptations applied: {len(adaptations_applied)}")
            else:
                logger.error(f"❌ Copy operation failed with {len(errors)} errors")
            
            return copy_result
            
        except Exception as e:
            logger.error(f"Exception during copy execution: {e}")
            return CopyResult(
                operation_id=operation_id,
                plan_id=plan.plan_id,
                success=False,
                files_created=files_created,
                files_modified=files_modified,
                adaptations_applied=adaptations_applied,
                errors=errors + [f"Exception: {str(e)}"],
                warnings=warnings
            )
    
    def _execute_pattern_adaptation(self, plan: CopyPlan, access) -> Dict[str, Any]:
        """Execute pattern adaptation copy strategy."""
        logger.info("🎯 Executing pattern adaptation strategy")
        
        files_created = []
        errors = []
        warnings = []
        
        source_path = Path(plan.source_subsystem)
        target_path = Path(plan.target_location)
        
        # Find Python files in source
        python_files = list(source_path.rglob("*.py"))
        
        for source_file in python_files:
            try:
                # Read source content
                with open(source_file, 'r') as f:
                    content = f.read()
                
                # Apply name mappings and adaptations
                adapted_content = self._adapt_file_content(content, plan)
                
                # Determine target file path
                relative_path = source_file.relative_to(source_path)
                target_file = target_path / relative_path
                
                # Create target directory if needed
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Write adapted content
                if access.write_file(str(target_file), adapted_content):
                    files_created.append(str(target_file))
                    logger.debug(f"📄 Created adapted file: {target_file}")
                else:
                    errors.append(f"Failed to write {target_file}")
                    
            except Exception as e:
                errors.append(f"Error processing {source_file}: {str(e)}")
                logger.error(f"Error processing {source_file}: {e}")
        
        return {
            'files_created': files_created,
            'errors': errors,
            'warnings': warnings
        }
    
    def _adapt_file_content(self, content: str, plan: CopyPlan) -> str:
        """Adapt file content based on the copy plan."""
        adapted_content = content
        
        # Apply name mappings
        for old_name, new_name in plan.new_name_mappings.items():
            # Replace class names
            adapted_content = re.sub(rf'\bclass\s+{old_name}\b', f'class {new_name}', adapted_content)
            # Replace function names
            adapted_content = re.sub(rf'\bdef\s+{old_name}\b', f'def {new_name}', adapted_content)
            # Replace references
            adapted_content = re.sub(rf'\b{old_name}\b', new_name, adapted_content)
        
        # Add tool-specific imports if needed
        if AdaptationType.UPDATE_IMPORTS in plan.adaptations:
            adapted_content = self._update_imports_for_tools(adapted_content)
        
        # Add permission integration if needed
        if AdaptationType.ADD_PERMISSIONS in plan.adaptations:
            adapted_content = self._add_permission_integration(adapted_content)
        
        # Enhance with consciousness integration if needed
        if AdaptationType.ADD_CONSCIOUSNESS in plan.adaptations:
            adapted_content = self._add_consciousness_integration(adapted_content)
        
        return adapted_content
    
    def _update_imports_for_tools(self, content: str) -> str:
        """Update imports to work in tools context."""
        # Add permission manager import
        if 'from dawn.tools.development.self_mod.permission_manager import' not in content:
            import_line = "\nfrom dawn.tools.development.self_mod.permission_manager import get_permission_manager\n"
            # Insert after existing imports
            lines = content.split('\n')
            import_end = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_end = i
            lines.insert(import_end + 1, import_line.strip())
            content = '\n'.join(lines)
        
        return content
    
    def _add_permission_integration(self, content: str) -> str:
        """Add permission integration to class constructors."""
        # Find class definitions and add permission manager
        lines = content.split('\n')
        new_lines = []
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            
            # Check if this is a class __init__ method
            if re.match(r'\s*def __init__\(self.*\):', line):
                # Add permission manager initialization
                indent = len(line) - len(line.lstrip())
                permission_init = ' ' * (indent + 4) + 'self.permission_manager = get_permission_manager()'
                new_lines.append(permission_init)
        
        return '\n'.join(new_lines)
    
    def _add_consciousness_integration(self, content: str) -> str:
        """Add consciousness integration to the adapted code."""
        # Add consciousness state checking
        if 'from dawn.core.foundation.state import get_state' not in content:
            import_line = "from dawn.core.foundation.state import get_state"
            lines = content.split('\n')
            # Find a good place to insert
            for i, line in enumerate(lines):
                if line.strip().startswith('from dawn.') or line.strip().startswith('import '):
                    continue
                else:
                    lines.insert(i, import_line)
                    break
            content = '\n'.join(lines)
        
        return content
    
    # Analysis methods
    def _extract_primary_classes(self, subsystem_path: Path) -> List[str]:
        """Extract primary class names from subsystem."""
        classes = []
        
        for py_file in subsystem_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                        
            except Exception as e:
                logger.debug(f"Could not parse {py_file}: {e}")
        
        return list(set(classes))  # Remove duplicates
    
    def _extract_key_functions(self, subsystem_path: Path) -> List[str]:
        """Extract key function names from subsystem."""
        functions = []
        
        for py_file in subsystem_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                        functions.append(node.name)
                        
            except Exception as e:
                logger.debug(f"Could not parse {py_file}: {e}")
        
        return list(set(functions))
    
    def _analyze_dependencies(self, subsystem_path: Path) -> List[str]:
        """Analyze dependencies of the subsystem."""
        dependencies = set()
        
        for py_file in subsystem_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            dependencies.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            dependencies.add(node.module)
                            
            except Exception as e:
                logger.debug(f"Could not parse {py_file}: {e}")
        
        return list(dependencies)
    
    def _identify_architecture_pattern(self, subsystem_path: Path) -> str:
        """Identify the architectural pattern used by the subsystem."""
        # Simple pattern detection based on file structure and naming
        files = list(subsystem_path.rglob("*.py"))
        file_names = [f.name for f in files]
        
        if any('engine' in name.lower() for name in file_names):
            return "engine_pattern"
        elif any('manager' in name.lower() for name in file_names):
            return "manager_pattern"
        elif any('controller' in name.lower() for name in file_names):
            return "controller_pattern"
        elif any('system' in name.lower() for name in file_names):
            return "system_pattern"
        else:
            return "module_pattern"
    
    def _calculate_complexity_score(self, subsystem_path: Path) -> float:
        """Calculate complexity score for the subsystem."""
        total_lines = 0
        total_classes = 0
        total_functions = 0
        
        for py_file in subsystem_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    total_lines += len(content.split('\n'))
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        total_classes += 1
                    elif isinstance(node, ast.FunctionDef):
                        total_functions += 1
                        
            except Exception as e:
                logger.debug(f"Could not analyze {py_file}: {e}")
        
        # Simple complexity score based on size and structure
        complexity = (total_lines / 100) + (total_classes * 2) + (total_functions * 1.5)
        return min(complexity, 100.0)  # Cap at 100
    
    def _check_consciousness_integration(self, subsystem_path: Path) -> bool:
        """Check if subsystem has consciousness integration."""
        for py_file in subsystem_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    if 'consciousness' in content.lower() or 'get_state' in content:
                        return True
            except Exception:
                continue
        return False
    
    def _estimate_permission_requirements(self, subsystem_path: Path) -> PermissionLevel:
        """Estimate permission requirements for copying this subsystem."""
        path_str = str(subsystem_path).lower()
        
        if 'core' in path_str:
            return PermissionLevel.CORE_MODIFY
        elif any(critical in path_str for critical in ['consciousness', 'engine', 'system']):
            return PermissionLevel.SUBSYSTEM_MODIFY
        else:
            return PermissionLevel.TOOLS_MODIFY
    
    def _estimate_adaptation_effort(self, subsystem_path: Path) -> str:
        """Estimate effort required to adapt this subsystem."""
        complexity = self._calculate_complexity_score(subsystem_path)
        
        if complexity > 50:
            return "high"
        elif complexity > 20:
            return "medium"
        else:
            return "low"
    
    # Planning methods
    def _determine_adaptations(self, analysis: SubsystemAnalysis, strategy: CopyStrategy, kwargs: Dict[str, Any]) -> List[AdaptationType]:
        """Determine what adaptations are needed."""
        adaptations = []
        
        # Always update imports and rename classes for tools
        adaptations.extend([
            AdaptationType.RENAME_CLASSES,
            AdaptationType.UPDATE_IMPORTS,
            AdaptationType.UPDATE_DOCSTRINGS
        ])
        
        # Add permissions for tools
        adaptations.append(AdaptationType.ADD_PERMISSIONS)
        
        # Enhance logging for better debugging
        adaptations.append(AdaptationType.ENHANCE_LOGGING)
        
        # Add consciousness integration if not present
        if not analysis.consciousness_integration:
            adaptations.append(AdaptationType.ADD_CONSCIOUSNESS)
        
        # Strategy-specific adaptations
        if strategy == CopyStrategy.CONSCIOUSNESS_ENHANCED:
            adaptations.append(AdaptationType.ADD_CONSCIOUSNESS)
        
        return adaptations
    
    def _create_name_mappings(self, analysis: SubsystemAnalysis, target_name: str, kwargs: Dict[str, Any]) -> Dict[str, str]:
        """Create mappings for renaming classes and functions."""
        mappings = {}
        
        # Map primary classes to tool-oriented names
        for class_name in analysis.primary_classes:
            if 'engine' in class_name.lower():
                new_name = f"{target_name.title()}Tool"
            elif 'manager' in class_name.lower():
                new_name = f"{target_name.title()}Manager"
            elif 'controller' in class_name.lower():
                new_name = f"{target_name.title()}Controller"
            else:
                new_name = f"{target_name.title()}{class_name}"
            
            mappings[class_name] = new_name
        
        return mappings
    
    def _determine_copy_permission_level(self, analysis: SubsystemAnalysis, target_category: str) -> PermissionLevel:
        """Determine permission level needed for the copy operation."""
        # Tools creation generally requires TOOLS_MODIFY
        base_level = PermissionLevel.TOOLS_MODIFY
        
        # Escalate based on source complexity and target
        if analysis.permission_requirements == PermissionLevel.CORE_MODIFY:
            return PermissionLevel.SUBSYSTEM_MODIFY  # Step down one level for tools
        elif analysis.permission_requirements == PermissionLevel.SUBSYSTEM_MODIFY:
            return PermissionLevel.TOOLS_MODIFY
        
        return base_level
    
    def _plan_consciousness_enhancements(self, analysis: SubsystemAnalysis, kwargs: Dict[str, Any]) -> List[str]:
        """Plan consciousness enhancements for the copied tool."""
        enhancements = []
        
        if not analysis.consciousness_integration:
            enhancements.extend([
                "Add consciousness state monitoring",
                "Add consciousness-gated operations",
                "Add consciousness-aware logging"
            ])
        
        enhancements.extend([
            "Add permission-gated operations",
            "Add audit logging for tool operations",
            "Add integration with DAWN singleton"
        ])
        
        return enhancements
    
    def _estimate_file_count(self, analysis: SubsystemAnalysis, strategy: CopyStrategy) -> int:
        """Estimate number of files that will be created."""
        source_path = Path(analysis.subsystem_path)
        source_files = len(list(source_path.rglob("*.py")))
        
        if strategy == CopyStrategy.DIRECT_COPY:
            return source_files
        elif strategy == CopyStrategy.PATTERN_ADAPTATION:
            return max(1, source_files // 2)  # Simplified adaptation
        else:
            return source_files
    
    # Execution helper methods
    def _execute_direct_copy(self, plan: CopyPlan, access) -> Dict[str, Any]:
        """Execute direct copy strategy."""
        logger.info("📋 Executing direct copy strategy")
        
        source_path = Path(plan.source_subsystem)
        target_path = Path(plan.target_location)
        
        files_created = []
        errors = []
        
        try:
            for source_file in source_path.rglob("*"):
                if source_file.is_file():
                    relative_path = source_file.relative_to(source_path)
                    target_file = target_path / relative_path
                    
                    # Create directory if needed
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    if access.copy_file(str(source_file), str(target_file)):
                        files_created.append(str(target_file))
                    else:
                        errors.append(f"Failed to copy {source_file}")
        
        except Exception as e:
            errors.append(f"Direct copy error: {str(e)}")
        
        return {'files_created': files_created, 'errors': errors, 'warnings': []}
    
    def _execute_architecture_clone(self, plan: CopyPlan, access) -> Dict[str, Any]:
        """Execute architecture clone strategy."""
        # For now, delegate to pattern adaptation
        return self._execute_pattern_adaptation(plan, access)
    
    def _execute_consciousness_enhanced_copy(self, plan: CopyPlan, access) -> Dict[str, Any]:
        """Execute consciousness-enhanced copy strategy."""
        # Start with pattern adaptation
        result = self._execute_pattern_adaptation(plan, access)
        
        # Add consciousness-specific enhancements
        # This could include additional consciousness monitoring files, etc.
        
        return result
    
    def _apply_adaptation(self, adaptation: AdaptationType, plan: CopyPlan, access) -> Dict[str, Any]:
        """Apply a specific adaptation to the copied files."""
        logger.debug(f"Applying adaptation: {adaptation.value}")
        
        # Placeholder for adaptation-specific logic
        # Each adaptation type would have its own implementation
        
        return {'errors': [], 'warnings': []}
    
    def _create_tool_entry_point(self, plan: CopyPlan, access) -> Optional[str]:
        """Create an entry point file for the new tool."""
        target_path = Path(plan.target_location)
        entry_point_file = target_path / "__init__.py"
        
        # Generate entry point content
        tool_name = target_path.name
        content = f'''"""
{tool_name.title()} Tool
{'=' * (len(tool_name) + 5)}

Auto-generated tool created by DAWN's SubsystemCopier.
Adapted from subsystem: {plan.source_subsystem}

This tool provides enhanced capabilities with:
- Consciousness-aware operations
- Permission-gated access
- Comprehensive logging and auditing
- Integration with DAWN's architecture
"""

# Tool imports will be added here based on copied components
# from .main_component import MainToolClass

__all__ = [
    # Exports will be added based on copied components
]
'''
        
        if access.write_file(str(entry_point_file), content):
            logger.info(f"📝 Created tool entry point: {entry_point_file}")
            return str(entry_point_file)
        
        return None
    
    def get_available_subsystems(self) -> List[str]:
        """Get list of available subsystems that can be copied."""
        subsystems = []
        
        for item in self._subsystems_dir.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                subsystems.append(item.name)
        
        return sorted(subsystems)
    
    def get_copy_history(self, limit: int = 20) -> List[CopyResult]:
        """Get recent copy operation history."""
        return self._copy_history[-limit:]
