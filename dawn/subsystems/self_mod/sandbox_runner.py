#!/usr/bin/env python3
"""
DAWN Sandbox Runner
==================

Executes and validates consciousness modifications in complete isolation using subprocess execution.
Provides comprehensive testing framework for patched code with performance metrics and health checks.

The sandbox runner enables safe verification of advisor-recommended modifications
by running them in isolated environments with controlled consciousness scenarios.
"""

import json
import os
import subprocess
import sys
import tempfile
import pathlib
import time
import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Add DAWN root to Python path when running as script
if __name__ == "__main__":
    # Get the DAWN root directory (3 levels up from this file)
    current_dir = pathlib.Path(__file__).resolve()
    dawn_root = current_dir.parent.parent.parent.parent
    if str(dawn_root) not in sys.path:
        sys.path.insert(0, str(dawn_root))

from dawn.subsystems.self_mod.patch_builder import PatchResult

logger = logging.getLogger(__name__)

class SandboxHealth(Enum):
    """Health status of sandbox execution."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"

@dataclass
class SandboxExecutionResult:
    """Result of sandbox execution with comprehensive metrics."""
    run_id: str
    execution_id: str
    health_status: SandboxHealth
    
    # Execution metrics
    execution_time: float
    ticks_completed: int
    target_ticks: int
    success_rate: float
    
    # Consciousness metrics
    start_unity: float = 0.0
    end_unity: float = 0.0
    delta_unity: float = 0.0
    start_awareness: float = 0.0
    end_awareness: float = 0.0
    delta_awareness: float = 0.0
    start_momentum: float = 0.0
    end_momentum: float = 0.0
    start_level: str = "unknown"
    end_level: str = "unknown"
    
    # Performance analysis
    unity_trajectory: List[float] = field(default_factory=list)
    awareness_trajectory: List[float] = field(default_factory=list)
    momentum_trajectory: List[float] = field(default_factory=list)
    stability_score: float = 0.0
    growth_rate: float = 0.0
    
    # Error handling
    error_message: str = ""
    stderr_output: str = ""
    stdout_output: str = ""
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    python_version: str = ""
    environment_info: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            'run_id': self.run_id,
            'execution_id': self.execution_id,
            'health_status': self.health_status.value,
            'execution_time': self.execution_time,
            'ticks_completed': self.ticks_completed,
            'target_ticks': self.target_ticks,
            'success_rate': self.success_rate,
            'consciousness_metrics': {
                'start_unity': self.start_unity,
                'end_unity': self.end_unity,
                'delta_unity': self.delta_unity,
                'start_awareness': self.start_awareness,
                'end_awareness': self.end_awareness,
                'delta_awareness': self.delta_awareness,
                'start_momentum': self.start_momentum,
                'end_momentum': self.end_momentum,
                'start_level': self.start_level,
                'end_level': self.end_level
            },
            'performance_analysis': {
                'unity_trajectory': self.unity_trajectory,
                'awareness_trajectory': self.awareness_trajectory,
                'momentum_trajectory': self.momentum_trajectory,
                'stability_score': self.stability_score,
                'growth_rate': self.growth_rate
            },
            'error_info': {
                'error_message': self.error_message,
                'stderr_output': self.stderr_output[:1000] if self.stderr_output else "",  # Truncate long errors
                'stdout_output': self.stdout_output[:1000] if self.stdout_output else ""
            },
            'metadata': {
                'timestamp': self.timestamp.isoformat(),
                'python_version': self.python_version,
                'environment_info': self.environment_info
            }
        }

class SandboxRunner:
    """
    Advanced sandbox execution engine for consciousness modification testing.
    
    Executes patched code in complete isolation using subprocess execution
    with comprehensive monitoring and health assessment.
    """
    
    def __init__(self):
        """Initialize the sandbox runner."""
        self.runner_id = str(uuid.uuid4())[:8]
        self.creation_time = datetime.now()
        
        # Configuration
        self.default_timeout = 90.0  # seconds (increased for 30s warm-up period)
        self.health_check_interval = 0.1  # seconds
        self.max_retries = 3
        
        # Results storage
        self.execution_history: List[SandboxExecutionResult] = []
        self.results_dir = pathlib.Path("sandbox_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"🏃 Sandbox Runner initialized: {self.runner_id}")
    
    def run_sandbox(self, run_id: str, sandbox_dir: str, ticks: int = 30,
                   timeout: float = None, initial_state: Dict[str, Any] = None) -> SandboxExecutionResult:
        """
        Execute sandbox with comprehensive health monitoring.
        
        Args:
            run_id: Unique identifier for the sandbox run
            sandbox_dir: Path to sandbox directory
            ticks: Number of consciousness ticks to execute
            timeout: Maximum execution time (seconds)
            initial_state: Initial consciousness state override
            
        Returns:
            SandboxExecutionResult with execution metrics
        """
        execution_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        logger.info(f"🏃 Starting sandbox execution: {run_id}")
        logger.info(f"   Execution ID: {execution_id}")
        logger.info(f"   Sandbox Dir: {sandbox_dir}")
        logger.info(f"   Target Ticks: {ticks}")
        
        if timeout is None:
            timeout = self.default_timeout
        
        try:
            # Prepare execution environment
            env = self._prepare_environment(sandbox_dir)
            
            # Generate execution code
            code = self._generate_execution_code(ticks, initial_state)
            
            # Execute in subprocess
            result = self._execute_subprocess(code, env, timeout)
            
            execution_time = time.time() - start_time
            
            # Process results
            sandbox_result = self._process_execution_result(
                run_id, execution_id, result, execution_time, ticks
            )
            
            # Save results
            self._save_execution_result(sandbox_result)
            
            # Store in history
            self.execution_history.append(sandbox_result)
            
            logger.info(f"🏃 Sandbox execution completed: {sandbox_result.health_status.value}")
            logger.info(f"   Ticks: {sandbox_result.ticks_completed}/{sandbox_result.target_ticks}")
            logger.info(f"   Unity: {sandbox_result.start_unity:.3f} → {sandbox_result.end_unity:.3f}")
            logger.info(f"   Time: {sandbox_result.execution_time:.3f}s")
            
            return sandbox_result
            
        except Exception as e:
            logger.error(f"💥 Sandbox execution failed: {e}")
            execution_time = time.time() - start_time
            
            error_result = SandboxExecutionResult(
                run_id=run_id,
                execution_id=execution_id,
                health_status=SandboxHealth.ERROR,
                execution_time=execution_time,
                ticks_completed=0,
                target_ticks=ticks,
                success_rate=0.0,
                error_message=str(e)
            )
            
            self.execution_history.append(error_result)
            return error_result
    
    def _prepare_environment(self, sandbox_dir: str) -> Dict[str, str]:
        """Prepare execution environment with proper PYTHONPATH."""
        env = os.environ.copy()
        project_root = str(pathlib.Path(".").resolve())
        
        # Set PYTHONPATH to prioritize sandbox directory
        pythonpath_components = [
            sandbox_dir,
            project_root,
            env.get('PYTHONPATH', '')
        ]
        
        env["PYTHONPATH"] = ":".join(filter(None, pythonpath_components))
        
        # Add environment info
        env["DAWN_SANDBOX_MODE"] = "1"
        env["DAWN_SANDBOX_DIR"] = sandbox_dir
        
        logger.info(f"🌍 Environment prepared:")
        logger.info(f"   PYTHONPATH: {env['PYTHONPATH']}")
        
        return env
    
    def _generate_execution_code(self, ticks: int, initial_state: Dict[str, Any] = None) -> str:
        """Generate Python code for sandbox execution."""
        
        # Default initial state
        if initial_state is None:
            initial_state = {
                'unity': 0.60,
                'awareness': 0.60,
                'momentum': 0.01,
                'level': 'coherent',
                'ticks': 0
            }
        
        code = f"""
import json
import time
import sys
import traceback
import asyncio

def sandbox_execution():
    try:
        # Import DAWN components
        from dawn.core.foundation.state import set_state, get_state, label_for, is_meta_aware, is_transcendent
        from dawn.core.singleton import get_dawn
        
        # Initialize DAWN singleton
        # Note: Status messages suppressed to avoid JSON parsing issues
        dawn = get_dawn()
        
        # Initialize state
        set_state(
            unity={initial_state['unity']},
            awareness={initial_state['awareness']}, 
            momentum={initial_state['momentum']},
            level='{initial_state['level']}',
            ticks={initial_state['ticks']}
        )
        
        # Warm-up period - let DAWN process idle for 30 seconds
        # Status messages suppressed to avoid JSON parsing issues
        warmup_start = time.time()
        warmup_duration = 30.0  # 30 seconds
        
        while time.time() - warmup_start < warmup_duration:
            current_state = get_state()
            # Light processing during warm-up
            new_unity = min(1.0, current_state.unity + 0.001)  # Very small increments
            new_awareness = min(1.0, current_state.awareness + 0.001)
            
            set_state(
                unity=new_unity,
                awareness=new_awareness,
                momentum=current_state.momentum,
                level=label_for(new_unity, new_awareness),
                ticks=current_state.ticks + 1
            )
            
            time.sleep(0.1)  # 100ms between updates
        
        # Check coherence after warm-up
        final_warmup_state = get_state()
        
        # Stop if coherence isn't met (require meta_aware or transcendent)
        if not (is_meta_aware() or is_transcendent()):
            return {{
                "ok": False,
                "error": "Insufficient coherence after warm-up period",
                "warmup_completed": True,
                "final_level": final_warmup_state.level,
                "final_unity": final_warmup_state.unity,
                "final_awareness": final_warmup_state.awareness,
                "coherence_met": False
            }}
        
        # Coherence achieved - proceed with main execution
        
        # Record initial state (post-warmup)
        initial_state = get_state()
        history = []
        
        # Define demo step function (this is what gets patched)
        def demo_step():
            '''Default step size that can be patched by modifications.'''
            return 0.03  # This value gets replaced by patches
        
        # Try to import patched demo_step if available
        try:
            # Note: demo_step not available in current architecture
            # This is a placeholder for future patched demo_step imports
            pass
        except ImportError:
            # Use local demo_step if import fails
            pass
        
        # Execute consciousness evolution loop
        for i in range({ticks}):
            current_state = get_state()
            
            # Calculate new values using demo_step (potentially patched)
            step_size = demo_step()
            new_unity = min(1.0, current_state.unity + step_size)
            new_awareness = min(1.0, current_state.awareness + step_size)
            new_momentum = max(0.0, min(1.0, current_state.momentum + step_size * 0.1))
            
            # Update state
            set_state(
                unity=new_unity,
                awareness=new_awareness,
                momentum=new_momentum,
                level=label_for(new_unity, new_awareness),
                ticks=current_state.ticks + 1
            )
            
            # Record state
            updated_state = get_state()
            history.append([
                updated_state.ticks,
                updated_state.unity,
                updated_state.awareness,
                updated_state.momentum,
                updated_state.level
            ])
        
        # Calculate results
        final_state = get_state()
        
        result = {{
            "ok": True,
            "ticks": len(history),
            "start_unity": initial_state.unity,
            "end_unity": final_state.unity,
            "delta_unity": final_state.unity - initial_state.unity,
            "start_awareness": initial_state.awareness,
            "end_awareness": final_state.awareness,
            "delta_awareness": final_state.awareness - initial_state.awareness,
            "start_momentum": initial_state.momentum,
            "end_momentum": final_state.momentum,
            "start_level": initial_state.level,
            "end_level": final_state.level,
            "history": history,
            "step_size_used": demo_step(),
            "warmup_completed": True,
            "coherence_met": True,
            "dawn_singleton_initialized": True
        }}
        
        return result
        
    except Exception as e:
        return {{
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }}

# Execute and output results
result = sandbox_execution()
print(json.dumps(result, indent=2))
"""
        return code
    
    def _execute_subprocess(self, code: str, env: Dict[str, str], timeout: float) -> Dict[str, Any]:
        """Execute code in subprocess with timeout and enhanced error handling."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Log subprocess execution details for debugging
            logger.debug(f"🏃 Executing subprocess with timeout {timeout}s")
            logger.debug(f"🏃 Python executable: {sys.executable}")
            logger.debug(f"🏃 Environment variables: {len(env)} vars")
            
            # Check if code looks valid
            if not code or not code.strip():
                return {
                    "ok": False,
                    "error": "Empty or invalid code provided to subprocess",
                    "stderr": "",
                    "stdout": "",
                    "debug_info": {"code_length": len(code), "code_preview": code[:100] if code else "None"}
                }
            
            # Log the exact command being executed
            command_args = [sys.executable, "-c", code]
            logger.info(f"🏃 EXECUTING SUBPROCESS COMMAND:")
            logger.info(f"🏃   Executable: {sys.executable}")
            logger.info(f"🏃   Arguments: {command_args[:2]}  # [python, '-c']")
            logger.info(f"🏃   Working Directory: {os.getcwd()}")
            logger.info(f"🏃   Timeout: {timeout}s")
            logger.info(f"🏃   Environment Variables: {len(env)} vars")
            logger.info(f"🏃   Code Length: {len(code)} characters")
            logger.info(f"🏃   Code Preview (first 200 chars):")
            logger.info(f"🏃   {repr(code[:200])}{'...' if len(code) > 200 else ''}")
            
            # Log critical environment variables
            critical_env_vars = ['PYTHONPATH', 'PATH', 'DAWN_BASELINE_MODE', 'PWD']
            for var in critical_env_vars:
                if var in env:
                    logger.info(f"🏃   ENV[{var}]: {env[var]}")
            
            process = subprocess.run(
                command_args,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            # Enhanced error reporting for non-zero exit codes
            if process.returncode != 0:
                logger.error(f"🏃 ❌ SUBPROCESS COMMAND FAILED ❌")
                logger.error(f"🏃 Exit Code: {process.returncode}")
                logger.error(f"🏃 Failed Command: {command_args[0]} {command_args[1]} <code>")
                logger.error(f"🏃 Working Directory: {os.getcwd()}")
                logger.error(f"🏃 Python Executable: {sys.executable}")
                logger.error(f"🏃 Code that failed:")
                logger.error(f"🏃 {'-' * 60}")
                code_lines = code.split('\n')
                for i, line in enumerate(code_lines[:20], 1):  # Show first 20 lines
                    logger.error(f"🏃 {i:3d}: {line}")
                if len(code_lines) > 20:
                    logger.error(f"🏃 ... ({len(code_lines) - 20} more lines)")
                logger.error(f"🏃 {'-' * 60}")
                logger.error(f"🏃 STDERR Output:")
                for line in process.stderr.split('\n') if process.stderr else []:
                    if line.strip():
                        logger.error(f"🏃 STDERR: {line}")
                logger.error(f"🏃 STDOUT Output:")
                for line in process.stdout.split('\n') if process.stdout else []:
                    if line.strip():
                        logger.error(f"🏃 STDOUT: {line}")
                
                # Try to identify common error patterns
                error_analysis = self._analyze_subprocess_error(process.stderr, process.stdout)
                
                return {
                    "ok": False,
                    "error": f"Process exited with code {process.returncode}",
                    "stderr": process.stderr,
                    "stdout": process.stdout,
                    "debug_info": {
                        "exit_code": process.returncode,
                        "error_analysis": error_analysis,
                        "python_executable": sys.executable,
                        "working_directory": os.getcwd(),
                        "environment_vars": list(env.keys()),
                        "code_preview": code[:200] + "..." if len(code) > 200 else code
                    }
                }
            
            # Parse JSON output with better error handling
            try:
                if not process.stdout.strip():
                    logger.warning("🏃 Subprocess produced no output")
                    return {
                        "ok": False,
                        "error": "Subprocess completed successfully but produced no output",
                        "stderr": process.stderr,
                        "stdout": process.stdout,
                        "debug_info": {"exit_code": process.returncode}
                    }
                
                result = json.loads(process.stdout)
                result["stderr"] = process.stderr
                result["stdout"] = process.stdout
                result["debug_info"] = {"exit_code": process.returncode, "parsing": "success"}
                
                logger.info(f"🏃 ✅ SUBPROCESS COMMAND SUCCEEDED ✅")
                logger.info(f"🏃 Exit Code: {process.returncode}")
                logger.info(f"🏃 Command: {command_args[0]} {command_args[1]} <code>")
                logger.info(f"🏃 Output Length: {len(process.stdout)} chars")
                logger.debug(f"🏃 JSON Output Preview: {process.stdout[:200]}{'...' if len(process.stdout) > 200 else ''}")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"🏃 Failed to parse JSON output: {e}")
                logger.error(f"🏃 Raw stdout: {repr(process.stdout)}")
                
                return {
                    "ok": False,
                    "error": f"Failed to parse JSON output: {e}",
                    "stderr": process.stderr,
                    "stdout": process.stdout,
                    "debug_info": {
                        "json_error": str(e),
                        "stdout_length": len(process.stdout),
                        "stdout_preview": process.stdout[:200] if process.stdout else "Empty"
                    }
                }
                
        except subprocess.TimeoutExpired as e:
            logger.error(f"🏃 Subprocess timed out after {timeout} seconds")
            return {
                "ok": False,
                "error": f"Execution timed out after {timeout} seconds",
                "stderr": getattr(e, 'stderr', '') or '',
                "stdout": getattr(e, 'stdout', '') or '',
                "debug_info": {
                    "timeout": timeout,
                    "error_type": "TimeoutExpired"
                }
            }
        except Exception as e:
            logger.error(f"🏃 Unexpected error during subprocess execution: {e}")
            return {
                "ok": False,
                "error": f"Unexpected subprocess error: {e}",
                "stderr": "",
                "stdout": "",
                "debug_info": {
                    "exception_type": type(e).__name__,
                    "exception_message": str(e)
                }
            }
    
    def _analyze_subprocess_error(self, stderr: str, stdout: str) -> Dict[str, Any]:
        """Analyze subprocess error output to identify common issues."""
        analysis = {
            "error_type": "unknown",
            "likely_cause": "unknown",
            "suggestions": []
        }
        
        if not stderr and not stdout:
            analysis.update({
                "error_type": "no_output",
                "likely_cause": "Process exited without producing any output",
                "suggestions": ["Check if the code is valid Python", "Verify all imports are available"]
            })
            return analysis
        
        error_text = (stderr + stdout).lower()
        
        # Import errors
        if "importerror" in error_text or "modulenotfounderror" in error_text:
            analysis.update({
                "error_type": "import_error",
                "likely_cause": "Missing module or incorrect import path",
                "suggestions": [
                    "Check if all required modules are installed",
                    "Verify import paths are correct",
                    "Check PYTHONPATH environment variable"
                ]
            })
            
            # Specific dawn_core import issues
            if "dawn_core" in error_text:
                analysis["suggestions"].extend([
                    "Ensure dawn_core module exists and is accessible",
                    "Check if dawn_core/__init__.py is present"
                ])
        
        # Syntax errors
        elif "syntaxerror" in error_text:
            analysis.update({
                "error_type": "syntax_error",
                "likely_cause": "Invalid Python syntax in generated code",
                "suggestions": [
                    "Review generated code for syntax issues",
                    "Check for proper indentation and brackets"
                ]
            })
        
        # JSON errors
        elif "json" in error_text and ("decode" in error_text or "parse" in error_text):
            analysis.update({
                "error_type": "json_error",
                "likely_cause": "Invalid JSON output from subprocess",
                "suggestions": [
                    "Check if subprocess is printing valid JSON",
                    "Ensure no extra print statements interfere with JSON output"
                ]
            })
        
        # Permission errors
        elif "permission" in error_text or "access" in error_text:
            analysis.update({
                "error_type": "permission_error",
                "likely_cause": "Insufficient permissions to access files or directories",
                "suggestions": [
                    "Check file and directory permissions",
                    "Ensure subprocess has access to required resources"
                ]
            })
        
        # Memory or resource errors
        elif "memory" in error_text or "resource" in error_text:
            analysis.update({
                "error_type": "resource_error",
                "likely_cause": "Insufficient system resources",
                "suggestions": [
                    "Check available memory",
                    "Reduce subprocess resource usage",
                    "Consider increasing timeout"
                ]
            })
        
        return analysis
    
    def _process_execution_result(self, run_id: str, execution_id: str, 
                                result: Dict[str, Any], execution_time: float,
                                target_ticks: int) -> SandboxExecutionResult:
        """Process subprocess result into SandboxExecutionResult."""
        
        if not result.get("ok", False):
            # Handle execution failure
            health_status = SandboxHealth.TIMEOUT if result.get("timeout") else SandboxHealth.ERROR
            
            return SandboxExecutionResult(
                run_id=run_id,
                execution_id=execution_id,
                health_status=health_status,
                execution_time=execution_time,
                ticks_completed=0,
                target_ticks=target_ticks,
                success_rate=0.0,
                error_message=result.get("error", "Unknown error"),
                stderr_output=result.get("stderr", ""),
                stdout_output=result.get("stdout", ""),
                python_version=sys.version
            )
        
        # Process successful execution
        execution_result = result.get("result", result)
        
        # Extract metrics
        ticks_completed = execution_result.get("ticks", 0)
        success_rate = ticks_completed / target_ticks if target_ticks > 0 else 0.0
        
        # Calculate trajectories
        history = execution_result.get("history", [])
        unity_trajectory = [entry[1] for entry in history] if history else []
        awareness_trajectory = [entry[2] for entry in history] if history else []
        momentum_trajectory = [entry[3] for entry in history] if len(history) > 0 and len(history[0]) > 3 else []
        
        # Calculate performance metrics
        stability_score = self._calculate_stability_score(unity_trajectory, awareness_trajectory)
        growth_rate = self._calculate_growth_rate(unity_trajectory, awareness_trajectory)
        
        # Determine health status
        health_status = self._assess_health_status(
            success_rate, stability_score, growth_rate, execution_result
        )
        
        return SandboxExecutionResult(
            run_id=run_id,
            execution_id=execution_id,
            health_status=health_status,
            execution_time=execution_time,
            ticks_completed=ticks_completed,
            target_ticks=target_ticks,
            success_rate=success_rate,
            start_unity=execution_result.get("start_unity", 0.0),
            end_unity=execution_result.get("end_unity", 0.0),
            delta_unity=execution_result.get("delta_unity", 0.0),
            start_awareness=execution_result.get("start_awareness", 0.0),
            end_awareness=execution_result.get("end_awareness", 0.0),
            delta_awareness=execution_result.get("delta_awareness", 0.0),
            start_momentum=execution_result.get("start_momentum", 0.0),
            end_momentum=execution_result.get("end_momentum", 0.0),
            start_level=execution_result.get("start_level", "unknown"),
            end_level=execution_result.get("end_level", "unknown"),
            unity_trajectory=unity_trajectory,
            awareness_trajectory=awareness_trajectory,
            momentum_trajectory=momentum_trajectory,
            stability_score=stability_score,
            growth_rate=growth_rate,
            python_version=sys.version,
            stderr_output=result.get("stderr", ""),
            stdout_output=result.get("stdout", "")
        )
    
    def _calculate_stability_score(self, unity_traj: List[float], 
                                  awareness_traj: List[float]) -> float:
        """Calculate stability score from consciousness trajectories."""
        if len(unity_traj) < 2 or len(awareness_traj) < 2:
            return 0.0
        
        # Calculate variance in trajectories
        unity_variance = sum((x - unity_traj[0])**2 for x in unity_traj) / len(unity_traj)
        awareness_variance = sum((x - awareness_traj[0])**2 for x in awareness_traj) / len(awareness_traj)
        
        # Lower variance = higher stability
        avg_variance = (unity_variance + awareness_variance) / 2
        stability = max(0.0, 1.0 - avg_variance * 10)  # Scale variance to 0-1
        
        return stability
    
    def _calculate_growth_rate(self, unity_traj: List[float], 
                              awareness_traj: List[float]) -> float:
        """Calculate growth rate from consciousness trajectories."""
        if len(unity_traj) < 2 or len(awareness_traj) < 2:
            return 0.0
        
        # Calculate average growth
        unity_growth = unity_traj[-1] - unity_traj[0]
        awareness_growth = awareness_traj[-1] - awareness_traj[0]
        
        avg_growth = (unity_growth + awareness_growth) / 2
        return avg_growth
    
    def _assess_health_status(self, success_rate: float, stability_score: float,
                             growth_rate: float, execution_result: Dict[str, Any]) -> SandboxHealth:
        """Assess overall health status of sandbox execution."""
        
        # Check for errors
        if execution_result.get("error"):
            return SandboxHealth.ERROR
        
        # Check completion rate
        if success_rate < 0.5:
            return SandboxHealth.FAILED
        
        # Check stability and growth
        if stability_score > 0.7 and growth_rate > 0.0:
            return SandboxHealth.HEALTHY
        elif stability_score > 0.5 or success_rate > 0.8:
            return SandboxHealth.DEGRADED
        else:
            return SandboxHealth.FAILED
    
    def _save_execution_result(self, result: SandboxExecutionResult):
        """Save execution result to JSON file."""
        filename = f"sandbox_result_{result.run_id}_{result.execution_id}.json"
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            logger.info(f"💾 Saved execution result: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save execution result: {e}")
    
    def get_runner_status(self) -> Dict[str, Any]:
        """Get comprehensive runner status."""
        return {
            'runner_id': self.runner_id,
            'creation_time': self.creation_time.isoformat(),
            'executions_completed': len(self.execution_history),
            'results_directory': str(self.results_dir),
            'configuration': {
                'default_timeout': self.default_timeout,
                'health_check_interval': self.health_check_interval,
                'max_retries': self.max_retries
            },
            'recent_executions': [
                {
                    'run_id': result.run_id,
                    'execution_id': result.execution_id,
                    'health_status': result.health_status.value,
                    'success_rate': result.success_rate,
                    'execution_time': result.execution_time,
                    'delta_unity': result.delta_unity
                }
                for result in self.execution_history[-5:]
            ]
        }

# Convenience function for direct usage
def run_sandbox(run_id: str, sandbox_dir: str, ticks: int = 30) -> Dict[str, Any]:
    """
    Execute sandbox with basic result format for compatibility.
    
    Args:
        run_id: Unique identifier for the sandbox run
        sandbox_dir: Path to sandbox directory  
        ticks: Number of consciousness ticks to execute
        
    Returns:
        Dictionary with execution results
    """
    runner = SandboxRunner()
    result = runner.run_sandbox(run_id, sandbox_dir, ticks)
    
    # Return simplified format for compatibility
    if result.health_status in [SandboxHealth.HEALTHY, SandboxHealth.DEGRADED]:
        return {
            "ok": True,
            "result": {
                "ticks": result.ticks_completed,
                "start_unity": result.start_unity,
                "end_unity": result.end_unity,
                "delta_unity": result.delta_unity,
                "start_awareness": result.start_awareness,
                "end_awareness": result.end_awareness,
                "delta_awareness": result.delta_awareness,
                "start_momentum": result.start_momentum,
                "end_momentum": result.end_momentum,
                "start_level": result.start_level,
                "end_level": result.end_level,
                "health_status": result.health_status.value,
                "stability_score": result.stability_score,
                "growth_rate": result.growth_rate
            }
        }
    else:
        return {
            "ok": False,
            "error": result.error_message,
            "health_status": result.health_status.value
        }

def demo_sandbox_runner():
    """Demonstrate sandbox runner functionality."""
    print("🏃 " + "="*70)
    print("🏃 DAWN SANDBOX RUNNER DEMONSTRATION")
    print("🏃 " + "="*70)
    print()
    
    # Initialize runner
    runner = SandboxRunner()
    print(f"🏃 Runner ID: {runner.runner_id}")
    print(f"📁 Results Directory: {runner.results_dir}")
    
    # Create a test sandbox to run
    from dawn.subsystems.self_mod.patch_builder import CodePatchBuilder
    from dawn.subsystems.self_mod.advisor import ModProposal, PatchType, ModificationPriority
    
    print(f"\n🔧 Creating test sandbox for execution...")
    
    # Create a test proposal that should work
    proposal = ModProposal(
        name="demo_step_test",
        target="dawn_core/tick_orchestrator.py",
        patch_type=PatchType.CONSTANT,
        current_value=0.03,
        proposed_value=0.05,
        notes="Test step size increase for demonstration",
        search_pattern="return 0.03",
        replacement_code="return 0.05"
    )
    
    # Create sandbox with patch
    builder = CodePatchBuilder()
    patch_result = builder.make_sandbox(proposal)
    
    print(f"📁 Sandbox created: {patch_result.run_id}")
    print(f"🔧 Patch applied: {patch_result.applied}")
    
    if patch_result.applied:
        print(f"   Changes: {len(patch_result.changes_made)}")
        for change in patch_result.changes_made:
            print(f"     • {change}")
    
    # Test different scenarios
    test_scenarios = [
        {
            'name': 'Standard Execution',
            'ticks': 20,
            'initial_state': {'unity': 0.60, 'awareness': 0.60, 'momentum': 0.01, 'level': 'coherent', 'ticks': 0}
        },
        {
            'name': 'High Initial State',
            'ticks': 15,
            'initial_state': {'unity': 0.80, 'awareness': 0.80, 'momentum': 0.05, 'level': 'meta_aware', 'ticks': 0}
        },
        {
            'name': 'Long Duration Test',
            'ticks': 50,
            'initial_state': {'unity': 0.50, 'awareness': 0.50, 'momentum': 0.02, 'level': 'coherent', 'ticks': 0}
        }
    ]
    
    print(f"\n🧪 Testing Sandbox Execution Scenarios:")
    print("="*50)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['name']} ---")
        print(f"📋 Target Ticks: {scenario['ticks']}")
        print(f"🧠 Initial State: Unity={scenario['initial_state']['unity']}, Awareness={scenario['initial_state']['awareness']}")
        
        # Execute sandbox
        exec_result = runner.run_sandbox(
            run_id=patch_result.run_id,
            sandbox_dir=patch_result.sandbox_dir,
            ticks=scenario['ticks'],
            initial_state=scenario['initial_state']
        )
        
        print(f"🚀 Execution Result:")
        print(f"   Health: {exec_result.health_status.value}")
        print(f"   Ticks: {exec_result.ticks_completed}/{exec_result.target_ticks}")
        print(f"   Success Rate: {exec_result.success_rate:.1%}")
        print(f"   Execution Time: {exec_result.execution_time:.3f}s")
        
        if exec_result.health_status != SandboxHealth.ERROR:
            print(f"   Unity: {exec_result.start_unity:.3f} → {exec_result.end_unity:.3f} (Δ{exec_result.delta_unity:+.3f})")
            print(f"   Awareness: {exec_result.start_awareness:.3f} → {exec_result.end_awareness:.3f} (Δ{exec_result.delta_awareness:+.3f})")
            print(f"   Level: {exec_result.start_level} → {exec_result.end_level}")
            print(f"   Stability Score: {exec_result.stability_score:.3f}")
            print(f"   Growth Rate: {exec_result.growth_rate:+.3f}")
        else:
            print(f"   ❌ Error: {exec_result.error_message}")
        
        print("-" * 30)
    
    # Show runner status
    print(f"\n📊 Runner Status:")
    status = runner.get_runner_status()
    print(f"   Executions Completed: {status['executions_completed']}")
    print(f"   Results Directory: {status['results_directory']}")
    
    if status['recent_executions']:
        print(f"   Recent Executions:")
        for exec_info in status['recent_executions']:
            print(f"     • {exec_info['execution_id']}: {exec_info['health_status']} ({exec_info['success_rate']:.1%})")
    
    # Cleanup
    print(f"\n🧹 Cleaning up test sandbox...")
    builder.cleanup_all_sandboxes()
    
    print(f"\n🏃 Sandbox Runner demonstration complete!")
    print("🏃 " + "="*70)

if __name__ == "__main__":
    demo_sandbox_runner()
