#!/usr/bin/env python3
"""
DAWN Self-Modification Patch Builder
===================================

Creates and applies real code modifications in isolated sandbox environments.
Supports multiple patch patterns including constant replacement, threshold tweaking,
and strategy swapping with comprehensive validation and rollback capabilities.

The patch builder enables safe testing of advisor-recommended modifications
before applying them to the main codebase.
"""

import re
import shutil
import pathlib
import uuid
import logging
import ast
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dawn.subsystems.self_mod.advisor import ModProposal, PatchType

logger = logging.getLogger(__name__)

class PatchStatus(Enum):
    """Status of patch application."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    VALIDATION_ERROR = "validation_error"
    FILE_NOT_FOUND = "file_not_found"
    PARSE_ERROR = "parse_error"

@dataclass
class PatchResult:
    """Result of patch application operation."""
    run_id: str
    sandbox_dir: str
    target_rel: str
    applied: bool
    reason: str = ""
    
    # Enhanced metadata
    status: PatchStatus = PatchStatus.FAILED
    original_content: str = ""
    modified_content: str = ""
    changes_made: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    
    # Execution details
    patch_type: Optional[PatchType] = None
    execution_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    # File information
    original_size: int = 0
    modified_size: int = 0
    lines_changed: int = 0

@dataclass
class SandboxEnvironment:
    """Isolated sandbox environment for testing modifications."""
    sandbox_id: str
    root_dir: pathlib.Path
    created_at: datetime
    active_patches: List[PatchResult] = field(default_factory=list)
    
    def cleanup(self):
        """Clean up the sandbox environment."""
        if self.root_dir.exists():
            shutil.rmtree(self.root_dir)
            logger.info(f"🧹 Cleaned up sandbox: {self.sandbox_id}")

class CodePatchBuilder:
    """
    Advanced code patch builder for consciousness modifications.
    
    Creates isolated sandbox environments and applies targeted code changes
    based on advisor recommendations with comprehensive validation.
    """
    
    def __init__(self, sandbox_root: str = "sandbox_mods", simulation_mode: bool = False):
        """Initialize the patch builder."""
        self.builder_id = str(uuid.uuid4())[:8]
        self.sandbox_root = pathlib.Path(sandbox_root)
        self.simulation_mode = simulation_mode
        
        # Only create actual directories if not in simulation mode
        if not simulation_mode:
            self.sandbox_root.mkdir(parents=True, exist_ok=True)
        
        # Active sandboxes
        self.active_sandboxes: Dict[str, SandboxEnvironment] = {}
        
        # Simulation tracking
        self.simulation_results: List[Dict[str, Any]] = []
        
        # Patch patterns
        self.constant_patterns = {
            'DU_STEP': r'(DU_STEP\s*=\s*)([0-9.]+)',
            'DELTA_UNITY': r'(delta\s*=\s*)([0-9.]+)',
            'UNITY_BOOST': r'(unity_boost\s*=\s*)([0-9.]+)',
            'AWARENESS_STEP': r'(awareness_step\s*=\s*)([0-9.]+)',
            'MOMENTUM_FACTOR': r'(momentum_factor\s*=\s*)([0-9.]+)',
            'THRESHOLD_VALUE': r'(threshold\s*=\s*)([0-9.]+)',
        }
        
        self.threshold_patterns = {
            'TRANSCENDENT_UNITY': r'(u\s*>=\s*)([0-9.]+)(\s*and\s*a\s*>=\s*)([0-9.]+)',
            'META_AWARE_UNITY': r'(u\s*>=\s*)([0-9.]+)(\s*and\s*a\s*>=\s*)([0-9.]+)',
            'COHERENT_UNITY': r'(u\s*>=\s*)([0-9.]+)(\s*and\s*a\s*>=\s*)([0-9.]+)',
            'UNITY_THRESHOLD': r'(unity.*?>=\s*)([0-9.]+)',
            'AWARENESS_THRESHOLD': r'(awareness.*?>=\s*)([0-9.]+)',
        }
        
        logger.info(f"🔧 Code Patch Builder initialized: {self.builder_id}")
    
    def make_sandbox(self, proposal: ModProposal) -> PatchResult:
        """
        Create a sandbox and apply the proposed modification.
        
        Args:
            proposal: The modification proposal to apply
            
        Returns:
            PatchResult with application details
        """
        start_time = time.time()
        run_id = str(uuid.uuid4())[:8]
        
        mode_indicator = "🎭 SIMULATION" if self.simulation_mode else "🔧"
        logger.info(f"{mode_indicator} Creating {'simulated ' if self.simulation_mode else ''}sandbox for proposal: {proposal.name}")
        logger.info(f"   Run ID: {run_id}")
        logger.info(f"   Target: {proposal.target}")
        logger.info(f"   Patch Type: {proposal.patch_type.value}")
        logger.info(f"   Mode: {'SIMULATION' if self.simulation_mode else 'LIVE'}")
        
        try:
            # Create sandbox environment (real or simulated)
            sandbox_env = self._create_sandbox_environment(run_id)
            
            # Comprehensive file existence and accessibility check
            file_check_result = self._validate_target_file(proposal.target, run_id)
            if not file_check_result['valid']:
                return PatchResult(
                    run_id=run_id,
                    sandbox_dir=str(sandbox_env.root_dir),
                    target_rel=proposal.target,
                    applied=False,
                    status=PatchStatus.FILE_NOT_FOUND,
                    reason=file_check_result['reason'],
                    execution_time=time.time() - start_time
                )
            
            src_path = file_check_result['path']
            
            # Calculate relative path structure
            if src_path.is_absolute():
                # Try to make it relative to current working directory
                try:
                    rel_path = src_path.relative_to(pathlib.Path.cwd())
                except ValueError:
                    # If that fails, use the filename
                    rel_path = src_path.name
            else:
                rel_path = src_path
            
            dst_path = sandbox_env.root_dir / rel_path
            
            if not self.simulation_mode:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                logger.info(f"📁 Copied {src_path} → {dst_path}")
                # Read original content from copied file
                original_content = dst_path.read_text(encoding='utf-8')
            else:
                logger.info(f"🎭 Simulated copy {src_path} → {dst_path}")
                # Read original content directly from source
                original_content = src_path.read_text(encoding='utf-8')
            original_size = len(original_content)
            
            # Apply the patch
            patch_result = self._apply_patch(
                proposal, dst_path, original_content, run_id, sandbox_env
            )
            
            # Update execution time
            patch_result.execution_time = time.time() - start_time
            
            # Store sandbox reference
            sandbox_env.active_patches.append(patch_result)
            self.active_sandboxes[run_id] = sandbox_env
            
            logger.info(f"🔧 Patch application completed: {patch_result.status.value}")
            
            return patch_result
            
        except Exception as e:
            logger.error(f"💥 Error creating sandbox: {e}")
            return PatchResult(
                run_id=run_id,
                sandbox_dir="",
                target_rel=proposal.target,
                applied=False,
                status=PatchStatus.FAILED,
                reason=f"Sandbox creation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _create_sandbox_environment(self, run_id: str) -> SandboxEnvironment:
        """Create a new sandbox environment (real or simulated)."""
        sandbox_dir = self.sandbox_root / run_id
        
        if not self.simulation_mode:
            sandbox_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created sandbox environment: {sandbox_dir}")
        else:
            logger.info(f"🎭 Simulated sandbox environment: {sandbox_dir}")
        
        env = SandboxEnvironment(
            sandbox_id=run_id,
            root_dir=sandbox_dir,
            created_at=datetime.now()
        )
        
        return env
    
    def _validate_target_file(self, target_path: str, run_id: str) -> Dict[str, Any]:
        """
        Comprehensively validate target file existence and accessibility.
        
        Args:
            target_path: Path to the target file
            run_id: Current run ID for logging context
            
        Returns:
            Dict with 'valid' bool, 'reason' str, and 'path' pathlib.Path
        """
        logger.debug(f"🔍 Validating target file: {target_path}")
        
        # Convert to Path object
        try:
            src_path = pathlib.Path(target_path)
        except Exception as e:
            return {
                'valid': False,
                'reason': f"Invalid path format '{target_path}': {e}",
                'path': None
            }
        
        # Check if path exists
        if not src_path.exists():
            # Try alternative paths if the direct path doesn't exist
            alternative_paths = self._get_alternative_paths(target_path)
            
            for alt_path in alternative_paths:
                logger.debug(f"🔍 Trying alternative path: {alt_path}")
                if alt_path.exists():
                    logger.info(f"🔍 Found file at alternative path: {alt_path}")
                    return {
                        'valid': True,
                        'reason': f"File found at alternative path: {alt_path}",
                        'path': alt_path
                    }
            
            # No valid path found
            search_locations = [str(p) for p in [src_path] + alternative_paths]
            return {
                'valid': False,
                'reason': f"Source file not found: '{target_path}'. Searched locations: {search_locations}",
                'path': None
            }
        
        # Check if it's a file (not a directory)
        if not src_path.is_file():
            return {
                'valid': False,
                'reason': f"Path exists but is not a file: '{target_path}' (is {'directory' if src_path.is_dir() else 'other'})",
                'path': None
            }
        
        # Check if file is readable
        try:
            with open(src_path, 'r', encoding='utf-8') as f:
                # Try to read first few bytes to verify accessibility
                f.read(100)
        except PermissionError:
            return {
                'valid': False,
                'reason': f"File exists but is not readable due to permissions: '{target_path}'",
                'path': None
            }
        except UnicodeDecodeError:
            # Try binary mode for non-text files
            try:
                with open(src_path, 'rb') as f:
                    f.read(100)
                logger.warning(f"🔍 File appears to be binary: {target_path}")
            except Exception as e:
                return {
                    'valid': False,
                    'reason': f"File exists but cannot be read: '{target_path}': {e}",
                    'path': None
                }
        except Exception as e:
            return {
                'valid': False,
                'reason': f"File exists but cannot be accessed: '{target_path}': {e}",
                'path': None
            }
        
        # Check file size (avoid extremely large files)
        try:
            file_size = src_path.stat().st_size
            max_size = 10 * 1024 * 1024  # 10MB limit
            if file_size > max_size:
                return {
                    'valid': False,
                    'reason': f"File is too large for modification: '{target_path}' ({file_size} bytes > {max_size} bytes)",
                    'path': None
                }
        except Exception as e:
            logger.warning(f"🔍 Could not check file size for {target_path}: {e}")
        
        logger.debug(f"✅ File validation successful: {target_path}")
        return {
            'valid': True,
            'reason': "File exists and is accessible",
            'path': src_path
        }
    
    def _get_alternative_paths(self, target_path: str) -> List[pathlib.Path]:
        """
        Generate alternative paths to search for the target file.
        
        This helps handle cases where paths have changed or files have been moved.
        """
        alternatives = []
        original_path = pathlib.Path(target_path)
        
        # If it's already an absolute path, try making it relative
        if original_path.is_absolute():
            try:
                rel_path = original_path.relative_to(pathlib.Path.cwd())
                alternatives.append(rel_path)
            except ValueError:
                pass
        
        # Common path transformations for DAWN system
        if 'dawn_core' in target_path:
            # Try the new dawn/core/foundation structure
            new_path = target_path.replace('dawn_core', 'dawn/core/foundation')
            alternatives.append(pathlib.Path(new_path))
            
            # Try without the dawn_core prefix
            if target_path.startswith('dawn_core/'):
                no_prefix = target_path[10:]  # Remove 'dawn_core/'
                alternatives.append(pathlib.Path(f"dawn/{no_prefix}"))
                alternatives.append(pathlib.Path(f"dawn/core/{no_prefix}"))
        
        # If path contains dawn/, try with dawn_core/
        if 'dawn/' in target_path and 'dawn_core' not in target_path:
            legacy_path = target_path.replace('dawn/', 'dawn_core/')
            alternatives.append(pathlib.Path(legacy_path))
        
        # Try in common directories
        filename = original_path.name
        common_dirs = [
            pathlib.Path("dawn/core/foundation"),
            pathlib.Path("dawn_core"),
            pathlib.Path("dawn/subsystems"),
            pathlib.Path("."),
        ]
        
        for common_dir in common_dirs:
            if common_dir.exists():
                alternatives.append(common_dir / filename)
        
        # Remove duplicates and the original path
        unique_alternatives = []
        seen = {original_path}
        
        for alt in alternatives:
            if alt not in seen:
                unique_alternatives.append(alt)
                seen.add(alt)
        
        return unique_alternatives
    
    def _write_modified_content(self, target_path: pathlib.Path, content: str, 
                               proposal: ModProposal, run_id: str) -> Dict[str, Any]:
        """
        Write modified content to file (real or simulated).
        
        Returns:
            Dict with success status and validation results
        """
        if not self.simulation_mode:
            # Real write operation
            target_path.write_text(content, encoding='utf-8')
            logger.info(f"📝 Modified content written to: {target_path}")
        else:
            # Simulated write operation
            logger.info(f"🎭 Simulated write to: {target_path}")
            # Store simulation result
            simulation_result = {
                'run_id': run_id,
                'proposal_name': proposal.name,
                'target_file': str(target_path),
                'patch_type': proposal.patch_type.value,
                'original_value': proposal.current_value,
                'proposed_value': proposal.proposed_value,
                'content_length': len(content),
                'timestamp': datetime.now().isoformat(),
                'action': 'file_write_simulated'
            }
            self.simulation_results.append(simulation_result)
        
        # Validate syntax (always do this, even in simulation)
        validation_errors = self._validate_python_syntax(content)
        
        return {
            'success': True,
            'validation_errors': validation_errors,
            'content_length': len(content)
        }
    
    def _apply_patch(self, proposal: ModProposal, target_path: pathlib.Path, 
                     original_content: str, run_id: str, 
                     sandbox_env: SandboxEnvironment) -> PatchResult:
        """Apply the specific patch based on proposal type."""
        
        if proposal.patch_type == PatchType.CONSTANT:
            return self._apply_constant_patch(
                proposal, target_path, original_content, run_id, sandbox_env
            )
        elif proposal.patch_type == PatchType.THRESHOLD:
            return self._apply_threshold_patch(
                proposal, target_path, original_content, run_id, sandbox_env
            )
        elif proposal.patch_type == PatchType.STRATEGY:
            return self._apply_strategy_patch(
                proposal, target_path, original_content, run_id, sandbox_env
            )
        else:
            return PatchResult(
                run_id=run_id,
                sandbox_dir=str(sandbox_env.root_dir),
                target_rel=str(target_path.relative_to(sandbox_env.root_dir)),
                applied=False,
                status=PatchStatus.FAILED,
                reason=f"Unsupported patch type: {proposal.patch_type.value}",
                patch_type=proposal.patch_type,
                original_content=original_content
            )
    
    def _apply_constant_patch(self, proposal: ModProposal, target_path: pathlib.Path,
                             original_content: str, run_id: str,
                             sandbox_env: SandboxEnvironment) -> PatchResult:
        """Apply constant value replacement patch."""
        
        changes_made = []
        modified_content = original_content
        
        # Try specific search pattern if provided
        if proposal.search_pattern:
            if proposal.search_pattern in modified_content:
                if proposal.replacement_code:
                    modified_content = modified_content.replace(
                        proposal.search_pattern, proposal.replacement_code
                    )
                    changes_made.append(f"Replaced '{proposal.search_pattern}' with '{proposal.replacement_code}'")
                else:
                    # Generate replacement based on values
                    new_pattern = proposal.search_pattern.replace(
                        str(proposal.current_value), str(proposal.proposed_value)
                    )
                    modified_content = modified_content.replace(
                        proposal.search_pattern, new_pattern
                    )
                    changes_made.append(f"Updated value in '{proposal.search_pattern}': {proposal.current_value} → {proposal.proposed_value}")
            else:
                # Pattern not found, try generic patterns
                modified_content, pattern_changes = self._try_generic_constant_patterns(
                    modified_content, proposal
                )
                changes_made.extend(pattern_changes)
        else:
            # Try all generic constant patterns
            modified_content, pattern_changes = self._try_generic_constant_patterns(
                modified_content, proposal
            )
            changes_made.extend(pattern_changes)
        
        # Validate the changes
        if modified_content == original_content:
            return PatchResult(
                run_id=run_id,
                sandbox_dir=str(sandbox_env.root_dir),
                target_rel=str(target_path.relative_to(sandbox_env.root_dir)),
                applied=False,
                status=PatchStatus.FAILED,
                reason="No matching constant patterns found",
                patch_type=proposal.patch_type,
                original_content=original_content,
                modified_content=modified_content
            )
        
        # Write modified content (real or simulated)
        write_result = self._write_modified_content(target_path, modified_content, proposal, run_id)
        validation_errors = write_result['validation_errors']
        
        return PatchResult(
            run_id=run_id,
            sandbox_dir=str(sandbox_env.root_dir),
            target_rel=str(target_path.relative_to(sandbox_env.root_dir)),
            applied=True,
            status=PatchStatus.SUCCESS if not validation_errors else PatchStatus.VALIDATION_ERROR,
            reason="Constant patch applied successfully" if not validation_errors else "Syntax validation failed",
            patch_type=proposal.patch_type,
            original_content=original_content,
            modified_content=modified_content,
            changes_made=changes_made,
            validation_errors=validation_errors,
            original_size=len(original_content),
            modified_size=len(modified_content),
            lines_changed=len(changes_made)
        )
    
    def _apply_threshold_patch(self, proposal: ModProposal, target_path: pathlib.Path,
                              original_content: str, run_id: str,
                              sandbox_env: SandboxEnvironment) -> PatchResult:
        """Apply threshold value adjustment patch."""
        
        changes_made = []
        modified_content = original_content
        
        # Try specific search pattern if provided
        if proposal.search_pattern:
            pattern = proposal.search_pattern
            if proposal.replacement_code:
                if pattern in modified_content:
                    modified_content = modified_content.replace(pattern, proposal.replacement_code)
                    changes_made.append(f"Replaced threshold pattern: '{pattern}' → '{proposal.replacement_code}'")
            else:
                # Use regex replacement with current/proposed values
                old_value = str(proposal.current_value)
                new_value = str(proposal.proposed_value)
                
                # Try direct value replacement in the pattern
                new_pattern = pattern.replace(old_value, new_value)
                if pattern in modified_content:
                    modified_content = modified_content.replace(pattern, new_pattern)
                    changes_made.append(f"Updated threshold: {old_value} → {new_value} in pattern")
        else:
            # Try generic threshold patterns
            modified_content, pattern_changes = self._try_generic_threshold_patterns(
                modified_content, proposal
            )
            changes_made.extend(pattern_changes)
        
        if modified_content == original_content:
            return PatchResult(
                run_id=run_id,
                sandbox_dir=str(sandbox_env.root_dir),
                target_rel=str(target_path.relative_to(sandbox_env.root_dir)),
                applied=False,
                status=PatchStatus.FAILED,
                reason="No matching threshold patterns found",
                patch_type=proposal.patch_type,
                original_content=original_content,
                modified_content=modified_content
            )
        
        # Write and validate (real or simulated)
        write_result = self._write_modified_content(target_path, modified_content, proposal, run_id)
        validation_errors = write_result['validation_errors']
        
        return PatchResult(
            run_id=run_id,
            sandbox_dir=str(sandbox_env.root_dir),
            target_rel=str(target_path.relative_to(sandbox_env.root_dir)),
            applied=True,
            status=PatchStatus.SUCCESS if not validation_errors else PatchStatus.VALIDATION_ERROR,
            reason="Threshold patch applied successfully" if not validation_errors else "Syntax validation failed",
            patch_type=proposal.patch_type,
            original_content=original_content,
            modified_content=modified_content,
            changes_made=changes_made,
            validation_errors=validation_errors,
            original_size=len(original_content),
            modified_size=len(modified_content),
            lines_changed=len(changes_made)
        )
    
    def _apply_strategy_patch(self, proposal: ModProposal, target_path: pathlib.Path,
                             original_content: str, run_id: str,
                             sandbox_env: SandboxEnvironment) -> PatchResult:
        """Apply strategy/function injection patch."""
        
        changes_made = []
        modified_content = original_content
        
        # Strategy patches involve more complex code injection
        if proposal.replacement_code:
            # Find insertion point (end of imports, beginning of class, etc.)
            lines = modified_content.split('\n')
            
            # Find appropriate insertion point
            insert_line = self._find_strategy_insertion_point(lines, proposal)
            
            if insert_line >= 0:
                # Insert the new strategy code
                lines.insert(insert_line, proposal.replacement_code)
                modified_content = '\n'.join(lines)
                changes_made.append(f"Inserted strategy code at line {insert_line + 1}")
                
                # If there's a search pattern, also replace it
                if proposal.search_pattern and proposal.search_pattern in modified_content:
                    # This might be replacing a function call or algorithm
                    function_name = self._extract_function_name_from_code(proposal.replacement_code)
                    if function_name:
                        # Replace calls to old function with new function
                        modified_content = re.sub(
                            proposal.search_pattern,
                            function_name + "()",
                            modified_content
                        )
                        changes_made.append(f"Replaced function calls: {proposal.search_pattern} → {function_name}()")
            else:
                return PatchResult(
                    run_id=run_id,
                    sandbox_dir=str(sandbox_env.root_dir),
                    target_rel=str(target_path.relative_to(sandbox_env.root_dir)),
                    applied=False,
                    status=PatchStatus.FAILED,
                    reason="Could not find appropriate insertion point for strategy patch",
                    patch_type=proposal.patch_type,
                    original_content=original_content,
                    modified_content=modified_content
                )
        else:
            return PatchResult(
                run_id=run_id,
                sandbox_dir=str(sandbox_env.root_dir),
                target_rel=str(target_path.relative_to(sandbox_env.root_dir)),
                applied=False,
                status=PatchStatus.FAILED,
                reason="Strategy patch requires replacement_code",
                patch_type=proposal.patch_type,
                original_content=original_content,
                modified_content=modified_content
            )
        
        # Write and validate (real or simulated)
        write_result = self._write_modified_content(target_path, modified_content, proposal, run_id)
        validation_errors = write_result['validation_errors']
        
        return PatchResult(
            run_id=run_id,
            sandbox_dir=str(sandbox_env.root_dir),
            target_rel=str(target_path.relative_to(sandbox_env.root_dir)),
            applied=True,
            status=PatchStatus.SUCCESS if not validation_errors else PatchStatus.VALIDATION_ERROR,
            reason="Strategy patch applied successfully" if not validation_errors else "Syntax validation failed",
            patch_type=proposal.patch_type,
            original_content=original_content,
            modified_content=modified_content,
            changes_made=changes_made,
            validation_errors=validation_errors,
            original_size=len(original_content),
            modified_size=len(modified_content),
            lines_changed=len(changes_made)
        )
    
    def _try_generic_constant_patterns(self, content: str, 
                                     proposal: ModProposal) -> Tuple[str, List[str]]:
        """Try generic constant replacement patterns."""
        changes_made = []
        modified_content = content
        
        for pattern_name, pattern in self.constant_patterns.items():
            new_content = re.sub(
                pattern,
                rf"\g<1>{proposal.proposed_value}",
                modified_content,
                count=1
            )
            if new_content != modified_content:
                changes_made.append(f"Applied {pattern_name} pattern: {proposal.current_value} → {proposal.proposed_value}")
                modified_content = new_content
                break  # Only apply first matching pattern
        
        return modified_content, changes_made
    
    def _try_generic_threshold_patterns(self, content: str,
                                       proposal: ModProposal) -> Tuple[str, List[str]]:
        """Try generic threshold replacement patterns."""
        changes_made = []
        modified_content = content
        
        for pattern_name, pattern in self.threshold_patterns.items():
            # For threshold patterns, we need to be more careful about replacement
            matches = re.finditer(pattern, modified_content)
            for match in matches:
                # Replace the threshold value
                old_match = match.group(0)
                new_match = old_match.replace(
                    str(proposal.current_value), 
                    str(proposal.proposed_value)
                )
                if new_match != old_match:
                    modified_content = modified_content.replace(old_match, new_match, 1)
                    changes_made.append(f"Applied {pattern_name} threshold: {proposal.current_value} → {proposal.proposed_value}")
                    break
            
            if changes_made:
                break  # Only apply first matching pattern
        
        return modified_content, changes_made
    
    def _find_strategy_insertion_point(self, lines: List[str], proposal: ModProposal) -> int:
        """Find appropriate insertion point for strategy code."""
        # Look for end of imports
        import_end = -1
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')) or line.strip() == '':
                import_end = i
            else:
                break
        
        # If we have a specific function name, try to find it
        if proposal.function_name:
            for i, line in enumerate(lines):
                if f"def {proposal.function_name}" in line:
                    return i  # Insert before the function
        
        # Default: insert after imports
        return import_end + 1 if import_end >= 0 else 0
    
    def _extract_function_name_from_code(self, code: str) -> Optional[str]:
        """Extract function name from code string."""
        lines = code.split('\n')
        for line in lines:
            if line.strip().startswith('def '):
                # Extract function name
                match = re.search(r'def\s+(\w+)', line)
                if match:
                    return match.group(1)
        return None
    
    def _validate_python_syntax(self, content: str) -> List[str]:
        """Validate Python syntax of modified content."""
        errors = []
        try:
            ast.parse(content)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Parse error: {str(e)}")
        
        return errors
    
    def get_sandbox_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific sandbox."""
        if run_id not in self.active_sandboxes:
            return None
        
        sandbox = self.active_sandboxes[run_id]
        return {
            'sandbox_id': sandbox.sandbox_id,
            'root_dir': str(sandbox.root_dir),
            'created_at': sandbox.created_at.isoformat(),
            'active_patches': len(sandbox.active_patches),
            'patch_details': [
                {
                    'target': patch.target_rel,
                    'status': patch.status.value,
                    'applied': patch.applied,
                    'changes': len(patch.changes_made),
                    'lines_changed': patch.lines_changed
                }
                for patch in sandbox.active_patches
            ]
        }
    
    def cleanup_sandbox(self, run_id: str) -> bool:
        """Clean up a specific sandbox."""
        if run_id in self.active_sandboxes:
            sandbox = self.active_sandboxes[run_id]
            sandbox.cleanup()
            del self.active_sandboxes[run_id]
            return True
        return False
    
    def cleanup_all_sandboxes(self):
        """Clean up all active sandboxes."""
        for sandbox in list(self.active_sandboxes.values()):
            sandbox.cleanup()
        self.active_sandboxes.clear()
        logger.info("🧹 Cleaned up all sandboxes")
    
    def get_simulation_results(self) -> List[Dict[str, Any]]:
        """Get all simulation results."""
        return self.simulation_results.copy()
    
    def clear_simulation_results(self):
        """Clear simulation results."""
        self.simulation_results.clear()
    
    def generate_simulation_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive simulation report.
        
        Returns:
            Dict containing simulation summary and details
        """
        if not self.simulation_mode:
            return {'error': 'Not in simulation mode'}
        
        total_simulations = len(self.simulation_results)
        if total_simulations == 0:
            return {
                'mode': 'SIMULATION',
                'total_simulations': 0,
                'message': 'No simulations performed yet'
            }
        
        # Group by patch type
        by_patch_type = {}
        by_target_file = {}
        successful_simulations = 0
        
        for result in self.simulation_results:
            patch_type = result.get('patch_type', 'unknown')
            target_file = result.get('target_file', 'unknown')
            
            # Count by patch type
            by_patch_type[patch_type] = by_patch_type.get(patch_type, 0) + 1
            
            # Count by target file
            by_target_file[target_file] = by_target_file.get(target_file, 0) + 1
            
            # Count successes (assume all simulations are successful for now)
            successful_simulations += 1
        
        # Generate summary
        report = {
            'mode': 'SIMULATION',
            'builder_id': self.builder_id,
            'total_simulations': total_simulations,
            'successful_simulations': successful_simulations,
            'failed_simulations': total_simulations - successful_simulations,
            'success_rate': successful_simulations / total_simulations if total_simulations > 0 else 0,
            'summary': {
                'by_patch_type': by_patch_type,
                'by_target_file': by_target_file,
                'most_modified_file': max(by_target_file.items(), key=lambda x: x[1])[0] if by_target_file else None,
                'most_common_patch_type': max(by_patch_type.items(), key=lambda x: x[1])[0] if by_patch_type else None
            },
            'simulations': self.simulation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        return report

# Convenience function
def make_sandbox(proposal: ModProposal, simulation_mode: bool = False) -> PatchResult:
    """Create a sandbox and apply the proposed modification."""
    builder = CodePatchBuilder(simulation_mode=simulation_mode)
    return builder.make_sandbox(proposal)

def make_simulation_sandbox(proposal: ModProposal) -> PatchResult:
    """Create a simulated sandbox without actual file modifications."""
    return make_sandbox(proposal, simulation_mode=True)

def demo_patch_builder():
    """Demonstrate patch builder functionality."""
    print("🔧 " + "="*70)
    print("🔧 DAWN CODE PATCH BUILDER DEMONSTRATION")
    print("🔧 " + "="*70)
    print()
    
    # Import required modules
    from dawn.subsystems.self_mod.advisor import ModProposal, PatchType, ModificationPriority
    
    # Initialize patch builder
    builder = CodePatchBuilder()
    print(f"🔧 Patch Builder ID: {builder.builder_id}")
    print(f"📁 Sandbox Root: {builder.sandbox_root}")
    
    # Create test proposals
    test_proposals = [
        ModProposal(
            name="unity_step_increase",
            target="dawn_core/tick_orchestrator.py",
            patch_type=PatchType.CONSTANT,
            current_value=0.03,
            proposed_value=0.04,
            notes="Increase unity step for momentum boost",
            priority=ModificationPriority.HIGH,
            search_pattern="delta = 0.03",
            replacement_code="delta = 0.04"
        ),
        ModProposal(
            name="transcendent_threshold_lower", 
            target="dawn/core/foundation/state.py",
            patch_type=PatchType.THRESHOLD,
            current_value=0.90,
            proposed_value=0.88,
            notes="Lower transcendent threshold for easier progression",
            priority=ModificationPriority.NORMAL,
            search_pattern="u >= .90 and a >= .90",
            replacement_code="u >= .88 and a >= .88"
        )
    ]
    
    print(f"\n🧪 Testing Patch Application:")
    print("="*50)
    
    for i, proposal in enumerate(test_proposals, 1):
        print(f"\n--- Test {i}: {proposal.name} ---")
        print(f"📋 Target: {proposal.target}")
        print(f"🔧 Type: {proposal.patch_type.value}")
        print(f"📝 Change: {proposal.current_value} → {proposal.proposed_value}")
        
        # Apply patch
        result = builder.make_sandbox(proposal)
        
        print(f"🚀 Result: {result.status.value}")
        print(f"✅ Applied: {result.applied}")
        
        if result.applied:
            print(f"   📁 Sandbox: {result.sandbox_dir}")
            print(f"   🎯 Target: {result.target_rel}")
            print(f"   🔄 Changes: {len(result.changes_made)}")
            for change in result.changes_made:
                print(f"     • {change}")
            
            if result.validation_errors:
                print(f"   ⚠️  Validation Errors: {len(result.validation_errors)}")
                for error in result.validation_errors:
                    print(f"     • {error}")
            else:
                print(f"   ✅ Syntax validation passed")
            
            print(f"   📊 Size: {result.original_size} → {result.modified_size} bytes")
            print(f"   ⏱️  Time: {result.execution_time:.3f}s")
        else:
            print(f"   ❌ Reason: {result.reason}")
        
        print("-" * 30)
    
    # Show builder status
    print(f"\n📊 Builder Status:")
    print(f"   Active Sandboxes: {len(builder.active_sandboxes)}")
    
    for run_id, sandbox in builder.active_sandboxes.items():
        status = builder.get_sandbox_status(run_id)
        print(f"   📁 {run_id}: {status['active_patches']} patches applied")
    
    # Cleanup demonstration
    print(f"\n🧹 Cleaning up demonstration sandboxes...")
    builder.cleanup_all_sandboxes()
    
    print(f"\n🔧 Patch Builder demonstration complete!")
    print("🔧 " + "="*70)

if __name__ == "__main__":
    demo_patch_builder()
