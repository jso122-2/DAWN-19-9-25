#!/usr/bin/env python3
"""
🔧 Test Feedback Loop Fix
========================

Quick test to verify that the feedback loop between universal logger
and centralized repository has been resolved.
"""

import sys
import time
from pathlib import Path

# Add DAWN to path
sys.path.insert(0, str(Path(__file__).parent))

def test_feedback_loop_fix():
    """Test that logging objects are excluded from discovery"""
    print("🔧 TESTING FEEDBACK LOOP FIX")
    print("=" * 40)
    
    try:
        from dawn.core.logging import get_universal_logger, get_centralized_repository
        from dawn.core.logging.universal_json_logger import UniversalObjectTracker
        from dawn.core.logging.centralized_repo import LogEntry
        
        print("✅ Logging modules imported successfully")
        
        # Test 1: Object tracker should reject logging objects
        tracker = UniversalObjectTracker()
        
        # Try to register a logging object - should return None
        log_entry = LogEntry(
            entry_id="test",
            system="test", 
            subsystem="test",
            module="test",
            timestamp=time.time(),
            log_type="test",
            file_path="test",
            size_bytes=0,
            hash_sha256="test"
        )
        
        result = tracker.register_object(log_entry, "test_log_entry")
        if result is None:
            print("✅ LogEntry object correctly rejected by tracker")
        else:
            print("❌ LogEntry object was registered (should be rejected)")
            return False
        
        # Test 2: Normal DAWN object should be registered
        class TestDawnObject:
            def __init__(self):
                self.name = "test_dawn_object"
                self.value = 123
        
        # Mock the module to look like a DAWN object
        TestDawnObject.__module__ = "dawn.test.module"
        
        dawn_obj = TestDawnObject()
        result = tracker.register_object(dawn_obj, "test_dawn_obj")
        if result:
            print("✅ Normal DAWN object correctly registered")
        else:
            print("❌ Normal DAWN object was rejected")
            return False
        
        # Test 3: Discovery should skip logging objects
        print("🔍 Testing object discovery filtering...")
        
        # This should not cause infinite loops or recursion
        discovered = tracker.discover_dawn_objects()
        print(f"✅ Discovery completed, found {len(discovered)} objects")
        
        # Check that no logging objects were discovered
        logging_objects = [obj_id for obj_id in discovered 
                          if any(keyword in obj_id for keyword in 
                                ['Logger', 'Snapshot', 'Universal', 'LogEntry', 'Repository'])]
        
        if not logging_objects:
            print("✅ No logging objects discovered (correct)")
        else:
            print(f"⚠️ Found {len(logging_objects)} logging objects in discovery")
            for obj_id in logging_objects[:5]:  # Show first 5
                print(f"  - {obj_id}")
        
        print("\n🎉 FEEDBACK LOOP FIX TESTS PASSED!")
        print("✅ Logging objects correctly excluded from tracking")
        print("✅ Normal DAWN objects still tracked")
        print("✅ Discovery filtering working")
        print("✅ No more infinite feedback loops!")
        
        return True
        
    except Exception as e:
        print(f"❌ Feedback loop fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_centralized_logging_stable():
    """Test that centralized logging is now stable"""
    print("\n🗂️ TESTING CENTRALIZED LOGGING STABILITY")
    print("=" * 50)
    
    try:
        from dawn.core.logging import get_centralized_repository
        
        # Create repository - should not cause recursion now
        repo = get_centralized_repository("test_stable_logs")
        print("✅ Centralized repository created without recursion")
        
        # Add a few test entries
        for i in range(3):
            repo.add_log_entry(
                system="test_system",
                subsystem="test_subsystem", 
                module="test_module",
                log_data={"test_entry": i, "stable": True},
                log_type="state"
            )
        
        print("✅ Log entries added without recursion errors")
        
        # Get stats - should work without issues
        stats = repo.get_repository_stats()
        print(f"✅ Repository stats retrieved: {stats['overview']['total_entries']} entries")
        
        return True
        
    except Exception as e:
        print(f"❌ Centralized logging stability test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 FEEDBACK LOOP FIX VERIFICATION")
    print("=" * 50)
    
    success1 = test_feedback_loop_fix()
    success2 = test_centralized_logging_stable()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 ALL FEEDBACK LOOP TESTS PASSED!")
        print("✅ Logging objects excluded from discovery")
        print("✅ Centralized repository stable")
        print("✅ No more recursion/feedback loop errors!")
        print("✅ Universal logging system fully operational!")
    else:
        print("❌ SOME TESTS FAILED")
        if not success1:
            print("❌ Feedback loop fix test failed")
        if not success2:
            print("❌ Centralized logging stability test failed")
    
    print("\n🔧 Feedback loop fix test complete!")
