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

from dawn_core.self_mod.advisor import ModProposal, PatchType

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
    
    def __init__(self, sandbox_root: str = "sandbox_mods"):
        """Initialize the patch builder."""
        self.builder_id = str(uuid.uuid4())[:8]
        self.sandbox_root = pathlib.Path(sandbox_root)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        
        # Active sandboxes
        self.active_sandboxes: Dict[str, SandboxEnvironment] = {}
        
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
        
        logger.info(f"🔧 Creating sandbox for proposal: {proposal.name}")
        logger.info(f"   Run ID: {run_id}")
        logger.info(f"   Target: {proposal.target}")
        logger.info(f"   Patch Type: {proposal.patch_type.value}")
        
        try:
            # Create sandbox environment
            sandbox_env = self._create_sandbox_environment(run_id)
            
            # Copy target file to sandbox
            src_path = pathlib.Path(proposal.target)
            if not src_path.exists():
                return PatchResult(
                    run_id=run_id,
                    sandbox_dir=str(sandbox_env.root_dir),
                    target_rel=proposal.target,
                    applied=False,
                    status=PatchStatus.FILE_NOT_FOUND,
                    reason=f"Source file not found: {proposal.target}",
                    execution_time=time.time() - start_time
                )
            
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
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            
            logger.info(f"📁 Copied {src_path} → {dst_path}")
            
            # Read original content
            original_content = dst_path.read_text(encoding='utf-8')
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
        """Create a new sandbox environment."""
        sandbox_dir = self.sandbox_root / run_id
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        env = SandboxEnvironment(
            sandbox_id=run_id,
            root_dir=sandbox_dir,
            created_at=datetime.now()
        )
        
        logger.info(f"📁 Created sandbox environment: {sandbox_dir}")
        return env
    
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
        
        # Write modified content
        target_path.write_text(modified_content, encoding='utf-8')
        
        # Validate syntax
        validation_errors = self._validate_python_syntax(modified_content)
        
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
        
        # Write and validate
        target_path.write_text(modified_content, encoding='utf-8')
        validation_errors = self._validate_python_syntax(modified_content)
        
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
        
        # Write and validate
        target_path.write_text(modified_content, encoding='utf-8')
        validation_errors = self._validate_python_syntax(modified_content)
        
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

# Convenience function
def make_sandbox(proposal: ModProposal) -> PatchResult:
    """Create a sandbox and apply the proposed modification."""
    builder = CodePatchBuilder()
    return builder.make_sandbox(proposal)

def demo_patch_builder():
    """Demonstrate patch builder functionality."""
    print("🔧 " + "="*70)
    print("🔧 DAWN CODE PATCH BUILDER DEMONSTRATION")
    print("🔧 " + "="*70)
    print()
    
    # Import required modules
    from dawn_core.self_mod.advisor import ModProposal, PatchType, ModificationPriority
    
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
            target="dawn_core/state.py",
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
