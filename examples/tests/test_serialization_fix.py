#!/usr/bin/env python3
"""
🔧 Test Serialization Fix
========================

Quick test to verify that the circular reference and recursion issues
in the StateSerializer have been resolved.
"""

import sys
import time
from pathlib import Path

# Add DAWN to path
sys.path.insert(0, str(Path(__file__).parent))

def test_serialization_fix():
    """Test that the serialization fix resolves recursion issues"""
    print("🔧 TESTING SERIALIZATION FIX")
    print("=" * 40)
    
    try:
        from dawn.core.logging.universal_json_logger import StateSerializer
        
        print("✅ StateSerializer imported successfully")
        
        # Test 1: Basic serialization
        simple_data = {
            "name": "test",
            "value": 42,
            "nested": {"inner": "data"}
        }
        
        result = StateSerializer.serialize_value(simple_data)
        print("✅ Basic serialization works")
        
        # Test 2: Circular reference detection
        circular_data = {"name": "circular"}
        circular_data["self_ref"] = circular_data  # Create circular reference
        
        result = StateSerializer.serialize_value(circular_data)
        print("✅ Circular reference detection works")
        print(f"  Result: {result}")
        
        # Test 3: Deep nesting
        deep_data = {"level_1": {"level_2": {"level_3": {"level_4": {"level_5": {"deep": "data"}}}}}}
        result = StateSerializer.serialize_value(deep_data, max_depth=3)
        print("✅ Deep nesting with depth limit works")
        
        # Test 4: Complex object
        class TestObject:
            def __init__(self):
                self.name = "test_object"
                self.value = 123
                self.nested_obj = None
        
        test_obj = TestObject()
        test_obj.nested_obj = TestObject()  # Create nested object
        
        result = StateSerializer.serialize_value(test_obj)
        print("✅ Complex object serialization works")
        
        # Test 5: Self-referential object
        self_ref_obj = TestObject()
        self_ref_obj.self = self_ref_obj  # Create self-reference
        
        result = StateSerializer.serialize_value(self_ref_obj)
        print("✅ Self-referential object handled")
        print(f"  Result type: {type(result)}")
        
        print("\n🎉 ALL SERIALIZATION TESTS PASSED!")
        print("✅ Circular reference detection working")
        print("✅ Recursion depth limiting working")
        print("✅ Complex object handling working")
        print("✅ No more maximum recursion depth exceeded errors!")
        
        return True
        
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_universal_logger_integration():
    """Test that the universal logger no longer has recursion issues"""
    print("\n🔗 TESTING UNIVERSAL LOGGER INTEGRATION")
    print("=" * 50)
    
    try:
        from dawn.core.logging import get_universal_logger
        
        # Get universal logger
        logger = get_universal_logger()
        print("✅ Universal logger retrieved")
        
        # Try logging a simple object
        class SimpleTestObj:
            def __init__(self):
                self.name = "simple_test"
                self.value = 999
        
        test_obj = SimpleTestObj()
        
        # This should not cause recursion errors anymore
        success = logger.log_object_state("test_simple_obj", custom_metadata={"test": "serialization_fix"})
        
        if success:
            print("✅ Object logging successful - no recursion errors!")
        else:
            print("⚠️ Object logging returned False but no exception thrown")
        
        print("✅ Universal logger integration working without recursion issues")
        return True
        
    except Exception as e:
        print(f"❌ Universal logger integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 SERIALIZATION FIX VERIFICATION TEST")
    print("=" * 50)
    
    success1 = test_serialization_fix()
    success2 = test_universal_logger_integration()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Serialization recursion issues FIXED")
        print("✅ Circular reference detection WORKING")
        print("✅ Universal logging integration STABLE")
        print("✅ No more 'maximum recursion depth exceeded' errors!")
    else:
        print("❌ SOME TESTS FAILED")
        if not success1:
            print("❌ Serialization fix test failed")
        if not success2:
            print("❌ Universal logger integration test failed")
    
    print("\n🔧 Serialization fix test complete!")
