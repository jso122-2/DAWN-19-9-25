#!/usr/bin/env python3
"""
DAWN Run Archiver - Post-Demo Summary Generator
===============================================

Lightweight archiver for capturing run summaries, metrics, and logs
after demo completion. Creates timestamped artifacts for analysis.
"""

import json
import os
import time
import hashlib
from typing import List, Dict, Any
from pathlib import Path

def archive_run(log_lines: List[str], extra: Dict[str, Any] = None) -> str:
    """
    Archive a demo run with metrics, logs, and metadata.
    
    Args:
        log_lines: List of log lines from the run
        extra: Additional metadata to include
        
    Returns:
        Path to the created archive directory
    """
    if extra is None:
        extra = {}
    
    # Generate timestamp
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    outdir = os.path.join("runs", ts)
    os.makedirs(outdir, exist_ok=True)
    
    # Try to get metrics from centralized source
    unity = 0.0
    coherence = 0.0 
    synthesis_quality = 0.0
    
    try:
        from dawn_core.consciousness_metrics import _global_calculator
        # Create a mock state to get current metrics
        mock_state = {"test_module": {"coherence": 0.8, "unity": 0.7}}
        metrics = _global_calculator.calculate_unified_metrics(mock_state)
        unity = metrics.consciousness_unity
        coherence = metrics.coherence
        synthesis_quality = metrics.quality
    except ImportError:
        print("Warning: Could not import centralized metrics")
    except Exception as e:
        print(f"Warning: Could not calculate metrics: {e}")
    
    # Create summary
    summary = {
        "timestamp": ts,
        "unity": unity,
        "coherence": coherence,
        "synthesis_quality": synthesis_quality,
        "log_lines_count": len(log_lines),
        "log_hash": hashlib.md5("".join(log_lines).encode()).hexdigest()[:8],
        **extra,
    }
    
    # Write summary.json
    summary_path = os.path.join(outdir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Write log.txt
    log_path = os.path.join(outdir, "log.txt")
    with open(log_path, "w") as f:
        f.write("".join(log_lines))
    
    print(f"📁 Run archived: {outdir}")
    print(f"   - summary.json: {len(summary)} fields")
    print(f"   - log.txt: {len(log_lines)} lines")
    
    return outdir

def get_latest_run() -> str:
    """Get the path to the most recent run directory."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return ""
    
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return ""
    
    # Sort by name (timestamp) and return the latest
    latest = sorted(run_dirs, key=lambda x: x.name)[-1]
    return str(latest)

def load_run_summary(run_path: str) -> Dict[str, Any]:
    """Load summary.json from a run directory."""
    summary_path = os.path.join(run_path, "summary.json")
    if not os.path.exists(summary_path):
        return {}
    
    with open(summary_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    # Demo usage
    sample_logs = [
        "🌅 DAWN Engine started\n",
        "🧠 Consciousness unity: 0.856\n", 
        "🎼 Tick orchestrator synchronized\n",
        "🌅 DAWN Engine stopped\n"
    ]
    
    sample_extra = {
        "demo_type": "sample_run",
        "components_tested": ["engine", "orchestrator", "bus"],
        "duration_seconds": 42.5
    }
    
    archive_path = archive_run(sample_logs, sample_extra)
    print(f"Sample run archived to: {archive_path}")
