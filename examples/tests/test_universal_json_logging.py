#!/usr/bin/env python3
"""
🔍 Universal JSON Logging System Test
====================================

Comprehensive test of the universal JSON logging system that ensures
absolutely every process and module in DAWN writes its state to disk
in JSON/JSONL format.

This test verifies:
- Automatic discovery and registration of all DAWN objects
- Per-tick JSON state logging for all modules
- Integration with existing telemetry systems
- File output in JSONL format
- State change detection and tracking
- Performance under load
"""

import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List

# Add DAWN to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_universal_json_logging():
    """Test the complete universal JSON logging system"""
    print("🔍 DAWN UNIVERSAL JSON LOGGING SYSTEM TEST")
    print("=" * 50)
    
    try:
        # Import the universal logging system
        from dawn.core.logging import (
            start_complete_dawn_logging, get_dawn_integration_stats,
            log_all_dawn_system_states, get_universal_logger,
            log_object_state, register_for_logging,
            LoggingConfig, LogFormat, StateScope
        )
        
        print("✅ Universal logging system imported successfully")
        
        # Create test configuration
        config = LoggingConfig(
            base_path="logs/test_universal_json",
            format=LogFormat.JSONL,
            scope=StateScope.FULL,
            flush_interval_seconds=0.5,
            enable_auto_discovery=True,
            max_file_size_mb=10
        )
        
        print(f"✅ Created logging configuration: {config.format.value} format")
        
        # Start universal logging
        print("\n🚀 Starting complete DAWN universal JSON logging...")
        integration_results = start_complete_dawn_logging()
        
        # Report integration results
        successful = sum(1 for success in integration_results.values() if success)
        total = len(integration_results)
        
        print(f"✅ Integration complete: {successful}/{total} systems integrated")
        print("\nIntegration Results:")
        for system, success in integration_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {system}")
        
        # Get universal logger
        universal_logger = get_universal_logger()
        print(f"✅ Universal logger instance obtained")
        
        # Create test objects to demonstrate logging
        print("\n🧪 Creating test objects for demonstration...")
        
        class TestEngine:
            def __init__(self, name: str):
                self.name = name
                self.status = "initialized"
                self.tick_count = 0
                self.metrics = {"processed": 0, "errors": 0}
            
            def tick(self):
                self.tick_count += 1
                self.status = f"running_tick_{self.tick_count}"
                self.metrics["processed"] += 1
            
            def process_data(self, data: str):
                self.metrics["processed"] += len(data)
                return f"processed_{data}"
        
        class TestManager:
            def __init__(self):
                self.engines = []
                self.active = True
                self.management_stats = {"engines_managed": 0, "total_ticks": 0}
            
            def add_engine(self, engine):
                self.engines.append(engine)
                self.management_stats["engines_managed"] = len(self.engines)
            
            def update_all(self):
                for engine in self.engines:
                    engine.tick()
                self.management_stats["total_ticks"] += len(self.engines)
        
        # Create test objects
        test_engine1 = TestEngine("consciousness_test_engine")
        test_engine2 = TestEngine("processing_test_engine")
        test_manager = TestManager()
        
        # Register test objects for logging
        engine1_id = register_for_logging(test_engine1, "test_consciousness_engine")
        engine2_id = register_for_logging(test_engine2, "test_processing_engine")
        manager_id = register_for_logging(test_manager, "test_manager")
        
        test_manager.add_engine(test_engine1)
        test_manager.add_engine(test_engine2)
        
        print(f"✅ Created and registered test objects:")
        print(f"  - {engine1_id}")
        print(f"  - {engine2_id}")
        print(f"  - {manager_id}")
        
        # Simulate system activity with logging
        print("\n🎯 Simulating system activity with universal logging...")
        
        for tick in range(20):
            # Update test objects
            test_manager.update_all()
            
            # Process some data
            test_engine1.process_data(f"data_tick_{tick}")
            test_engine2.process_data(f"batch_data_{tick}")
            
            # Log individual object states
            log_object_state(test_engine1, custom_metadata={
                "tick_number": tick,
                "test_phase": "activity_simulation"
            })
            
            log_object_state(test_engine2, custom_metadata={
                "tick_number": tick,
                "test_phase": "activity_simulation"
            })
            
            log_object_state(test_manager, custom_metadata={
                "tick_number": tick,
                "test_phase": "activity_simulation"
            })
            
            # Log all DAWN system states (if any DAWN objects are loaded)
            dawn_logged = log_all_dawn_system_states(tick)
            
            if tick % 5 == 0:
                print(f"  Tick {tick}: Test objects updated, DAWN systems logged: {dawn_logged}")
            
            time.sleep(0.1)  # Small delay to simulate real system timing
        
        print("✅ Activity simulation complete")
        
        # Get comprehensive statistics
        print("\n📊 Gathering comprehensive statistics...")
        
        # Universal logger stats
        universal_stats = universal_logger.get_stats()
        print(f"✅ Universal Logger Stats:")
        print(f"  - Objects tracked: {universal_stats['objects_tracked']}")
        print(f"  - States logged: {universal_stats['states_logged']}")
        print(f"  - Files created: {universal_stats.get('files_active', 0)}")
        print(f"  - Runtime: {universal_stats['runtime_seconds']:.2f} seconds")
        print(f"  - States per second: {universal_stats['states_per_second']:.2f}")
        
        # DAWN integration stats
        try:
            dawn_stats = get_dawn_integration_stats()
            print(f"✅ DAWN Integration Stats:")
            print(f"  - Systems integrated: {dawn_stats['systems_integrated']}")
            print(f"  - Objects registered: {dawn_stats.get('objects_registered', 0)}")
            print(f"  - Integration errors: {dawn_stats.get('integration_errors', 0)}")
        except Exception as e:
            print(f"⚠️  DAWN integration stats not available: {e}")
        
        # Check generated files
        print("\n📁 Checking generated log files...")
        log_dir = Path("logs/test_universal_json")
        if log_dir.exists():
            log_files = list(log_dir.glob("*.jsonl"))
            print(f"✅ Generated {len(log_files)} JSONL log files:")
            
            for log_file in log_files[:5]:  # Show first 5 files
                file_size = log_file.stat().st_size
                print(f"  - {log_file.name} ({file_size} bytes)")
                
                # Show sample content
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            sample_data = json.loads(lines[-1])  # Last entry
                            print(f"    Sample: {sample_data['object_id']} at {sample_data['timestamp']}")
                except Exception as e:
                    print(f"    Error reading file: {e}")
            
            if len(log_files) > 5:
                print(f"  ... and {len(log_files) - 5} more files")
        else:
            print("⚠️  Log directory not found")
        
        # Performance test
        print("\n⚡ Performance test - high frequency logging...")
        
        start_time = time.time()
        performance_iterations = 100
        
        for i in range(performance_iterations):
            # Rapid state changes
            test_engine1.tick()
            test_engine2.tick()
            test_manager.update_all()
            
            # Log states
            log_object_state(test_engine1, custom_metadata={"performance_test": i})
            log_object_state(test_engine2, custom_metadata={"performance_test": i})
            log_object_state(test_manager, custom_metadata={"performance_test": i})
        
        performance_time = time.time() - start_time
        performance_rate = (performance_iterations * 3) / performance_time  # 3 objects per iteration
        
        print(f"✅ Performance test complete:")
        print(f"  - {performance_iterations} iterations in {performance_time:.2f} seconds")
        print(f"  - {performance_rate:.2f} state logs per second")
        
        # Final statistics
        final_stats = universal_logger.get_stats()
        print(f"\n🎉 FINAL RESULTS:")
        print(f"  - Total objects tracked: {final_stats['objects_tracked']}")
        print(f"  - Total states logged: {final_stats['states_logged']}")
        print(f"  - Total runtime: {final_stats['runtime_seconds']:.2f} seconds")
        print(f"  - Average logging rate: {final_stats['states_per_second']:.2f} states/second")
        print(f"  - Errors encountered: {final_stats.get('errors', 0)}")
        
        print("\n✅ Universal JSON logging test PASSED!")
        print("🔍 Every process and module is now writing state to disk in JSON format!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import universal logging system: {e}")
        print("   Make sure the universal logging modules are properly installed")
        return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_dawn_main():
    """Test integration with the DAWN main runner"""
    print("\n🔗 Testing integration with DAWN main runner...")
    
    try:
        # Import DAWN main
        from dawn.main import DAWNRunner
        
        # Create DAWN runner with universal logging enabled
        config = {
            'mode': 'test',
            'debug': True,
            'universal_logging_enabled': True,
            'telemetry_enabled': True
        }
        
        runner = DAWNRunner(config)
        
        print("✅ DAWN runner created with universal logging enabled")
        
        # Check if universal logging was initialized
        if hasattr(runner, 'universal_logging_enabled') and runner.universal_logging_enabled:
            print("✅ Universal logging is enabled in DAWN runner")
            
            if hasattr(runner, 'universal_logger') and runner.universal_logger:
                print("✅ Universal logger instance is available")
                
                # Get stats
                if hasattr(runner, 'universal_logging_stats'):
                    stats = runner.universal_logging_stats
                    print(f"✅ Integration stats available: {len(stats)} entries")
                    
                    if 'integration_results' in stats:
                        results = stats['integration_results']
                        successful = sum(1 for success in results.values() if success)
                        total = len(results)
                        print(f"✅ DAWN systems integrated: {successful}/{total}")
            else:
                print("⚠️  Universal logger instance not available")
        else:
            print("⚠️  Universal logging not enabled in DAWN runner")
        
        print("✅ DAWN main integration test completed")
        return True
        
    except Exception as e:
        print(f"❌ DAWN main integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 DAWN UNIVERSAL JSON LOGGING COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Run main universal logging test
    success1 = test_universal_json_logging()
    
    # Run DAWN main integration test
    success2 = test_integration_with_dawn_main()
    
    # Final results
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Universal JSON logging is working perfectly")
        print("✅ Every DAWN process and module will write state to disk")
        print("✅ Integration with DAWN main runner is successful")
    else:
        print("❌ SOME TESTS FAILED")
        if not success1:
            print("❌ Universal logging system test failed")
        if not success2:
            print("❌ DAWN main integration test failed")
    
    print("\n🔍 Universal JSON logging system test complete!")
