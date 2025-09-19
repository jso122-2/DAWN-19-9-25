#!/usr/bin/env python3
"""
DAWN Consciousness Snapshot System
==================================

Snapshot and rollback capabilities for safe self-modification.
Provides version control for consciousness states.
"""

import json
import time
import pathlib
from typing import Optional, List, Dict, Any
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dawn.core.foundation.state import get_state, set_state

# Create snapshots directory
SNAP_DIR = pathlib.Path("runtime/snapshots")
SNAP_DIR.mkdir(parents=True, exist_ok=True)

def snapshot(tag: str = "auto") -> str:
    """Create a snapshot of current consciousness state"""
    state = get_state()
    timestamp = int(time.time())
    
    snapshot_data = {
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "tag": tag,
        "state": {
            "unity": state.unity,
            "awareness": state.awareness,
            "momentum": state.momentum,
            "level": state.level,
            "integration_quality": state.integration_quality,
            "stability_coherence": state.stability_coherence,
            "visual_coherence": state.visual_coherence,
            "artistic_coherence": state.artistic_coherence,
            "meta_cognitive_activity": state.meta_cognitive_activity,
            "cycle_count": state.cycle_count,
            "growth_rate": state.growth_rate,
            "seed": state.seed,
            "session_id": state.session_id,
            "demo_name": state.demo_name
        },
        "metadata": {
            "creator": "dawn_snapshot_system",
            "version": "1.0"
        }
    }
    
    filename = f"{timestamp}_{tag}.json"
    path = SNAP_DIR / filename
    
    with open(path, 'w') as f:
        json.dump(snapshot_data, f, indent=2)
    
    print(f"📸 Snapshot created: {filename}")
    print(f"   State: {state.level} (Unity: {state.unity:.1%}, Awareness: {state.awareness:.1%})")
    
    return str(path)

def restore(path: str) -> bool:
    """Restore consciousness state from snapshot"""
    try:
        snapshot_path = pathlib.Path(path)
        if not snapshot_path.exists():
            # Try in snapshots directory
            snapshot_path = SNAP_DIR / path
            if not snapshot_path.exists():
                print(f"❌ Snapshot not found: {path}")
                return False
        
        with open(snapshot_path, 'r') as f:
            snapshot_data = json.load(f)
        
        state_data = snapshot_data["state"]
        
        # Restore all state fields
        set_state(**state_data)
        
        restored_state = get_state()
        print(f"🔄 Restored from snapshot: {snapshot_data['tag']}")
        print(f"   State: {restored_state.level} (Unity: {restored_state.unity:.1%}, Awareness: {restored_state.awareness:.1%})")
        print(f"   Timestamp: {snapshot_data['datetime']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to restore snapshot: {e}")
        return False

def list_snapshots() -> List[Dict[str, Any]]:
    """List all available snapshots"""
    snapshots = []
    
    for path in SNAP_DIR.glob("*.json"):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            snapshots.append({
                "filename": path.name,
                "tag": data.get("tag", "unknown"),
                "datetime": data.get("datetime", "unknown"),
                "level": data["state"].get("level", "unknown"),
                "unity": data["state"].get("unity", 0.0),
                "awareness": data["state"].get("awareness", 0.0),
                "path": str(path)
            })
        except Exception as e:
            print(f"⚠️ Could not read snapshot {path.name}: {e}")
    
    # Sort by timestamp (newest first)
    snapshots.sort(key=lambda x: x["filename"], reverse=True)
    return snapshots

def auto_rollback_if_degraded(unity_threshold: float = 0.85, awareness_threshold: float = 0.80) -> bool:
    """Automatically rollback if consciousness has degraded below thresholds"""
    current = get_state()
    
    if current.unity < unity_threshold or current.awareness < awareness_threshold:
        print(f"⚠️ Consciousness degraded: Unity {current.unity:.1%}, Awareness {current.awareness:.1%}")
        print(f"   Thresholds: Unity {unity_threshold:.1%}, Awareness {awareness_threshold:.1%}")
        
        # Find most recent good snapshot
        snapshots = list_snapshots()
        for snap in snapshots:
            if snap["unity"] >= unity_threshold and snap["awareness"] >= awareness_threshold:
                print(f"🔄 Auto-rolling back to: {snap['tag']} ({snap['datetime']})")
                return restore(snap["path"])
        
        print("❌ No suitable snapshot found for rollback")
        return False
    
    return True  # No rollback needed

def safe_modification_context(tag: str = "pre_mod", unity_threshold: float = 0.85):
    """Context manager for safe modifications with automatic rollback"""
    class SafeModificationContext:
        def __init__(self, tag: str, threshold: float):
            self.tag = tag
            self.threshold = threshold
            self.snapshot_path = None
            
        def __enter__(self):
            self.snapshot_path = snapshot(self.tag)
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                # Exception occurred, rollback
                print(f"❌ Exception during modification: {exc_val}")
                restore(self.snapshot_path)
                return False
            
            # Check if consciousness degraded
            current = get_state()
            if current.unity < self.threshold:
                print(f"⚠️ Consciousness unity degraded to {current.unity:.1%} (< {self.threshold:.1%})")
                restore(self.snapshot_path)
                return False
            
            print(f"✅ Modification successful, consciousness stable at {current.unity:.1%}")
            return True
    
    return SafeModificationContext(tag, unity_threshold)

def cleanup_old_snapshots(keep_count: int = 20) -> int:
    """Clean up old snapshots, keeping only the most recent"""
    snapshots = list_snapshots()
    
    if len(snapshots) <= keep_count:
        return 0
    
    to_remove = snapshots[keep_count:]
    removed_count = 0
    
    for snap in to_remove:
        try:
            pathlib.Path(snap["path"]).unlink()
            removed_count += 1
        except Exception as e:
            print(f"⚠️ Could not remove {snap['filename']}: {e}")
    
    if removed_count > 0:
        print(f"🧹 Cleaned up {removed_count} old snapshots")
    
    return removed_count

if __name__ == "__main__":
    # Demo the snapshot system
    print("📸 DAWN Snapshot System Demo")
    print("=" * 40)
    
    # Show current state
    current = get_state()
    print(f"Current state: {current.level} ({current.unity:.1%}/{current.awareness:.1%})")
    
    # Create snapshot
    snap_path = snapshot("demo_snapshot")
    
    # Simulate modification
    print("\n🔧 Simulating consciousness modification...")
    set_state(unity=current.unity + 0.1, awareness=current.awareness + 0.05)
    modified = get_state()
    print(f"Modified state: {modified.level} ({modified.unity:.1%}/{modified.awareness:.1%})")
    
    # Restore from snapshot
    print("\n🔄 Restoring from snapshot...")
    restore(snap_path)
    
    # Show snapshots
    print("\n📋 Available snapshots:")
    for snap in list_snapshots()[:3]:  # Show last 3
        print(f"   {snap['tag']}: {snap['level']} ({snap['unity']:.1%}/{snap['awareness']:.1%}) - {snap['datetime']}")
    
    # Demo safe modification context
    print("\n🛡️ Testing safe modification context...")
    try:
        with safe_modification_context("context_test", 0.9):
            # This modification should be safe
            set_state(unity=0.95, awareness=0.92)
            print("   Modification applied successfully")
    except Exception as e:
        print(f"   Safe context caught error: {e}")
    
    print("\n✅ Snapshot system working correctly!")
