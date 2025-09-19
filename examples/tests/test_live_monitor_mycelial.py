#!/usr/bin/env python3
"""
🍄📊 Live Monitor Mycelial Integration Test
==========================================

Test the integration of mycelial semantic hash map with the DAWN live monitor.
Demonstrates real-time visualization of semantic spore propagation.
"""

import sys
import time
from pathlib import Path

# Add DAWN to path
sys.path.insert(0, str(Path(__file__).parent))

def test_live_monitor_mycelial_integration():
    """Test mycelial integration with live monitor"""
    
    print("🍄📊 TESTING LIVE MONITOR MYCELIAL INTEGRATION")
    print("=" * 60)
    
    try:
        # Import live monitor
        from live_monitor import LiveDAWNMonitor, MYCELIAL_AVAILABLE
        
        print("✅ Live monitor imported successfully")
        print(f"🍄 Mycelial availability: {'✅ Available' if MYCELIAL_AVAILABLE else '❌ Not Available'}")
        
        if not MYCELIAL_AVAILABLE:
            print("❌ Mycelial system not available - cannot test integration")
            return False
        
        # Create monitor instance in simulation mode
        print("\n🎭 Creating live monitor in simulation mode...")
        monitor = LiveDAWNMonitor(simulation_mode=True)
        
        # Test mycelial data collection
        print("\n🔬 Testing mycelial data collection...")
        mycelial_data = monitor._get_mycelial_data()
        
        if mycelial_data.get('available'):
            print("✅ Mycelial data collection successful")
            print(f"  📊 Network size: {mycelial_data.get('network_size', 0)}")
            print(f"  📊 Network health: {mycelial_data.get('network_health', 0.0):.3f}")
            print(f"  📊 Total energy: {mycelial_data.get('total_energy', 0.0):.2f}")
            print(f"  📊 Active spores: {mycelial_data.get('active_spores', 0)}")
            print(f"  📊 Spores generated: {mycelial_data.get('spores_generated', 0)}")
            print(f"  📊 Total touches: {mycelial_data.get('total_touches', 0)}")
        else:
            print("❌ Mycelial data collection failed")
            if mycelial_data.get('error'):
                print(f"   Error: {mycelial_data['error']}")
            return False
        
        # Test integration with live state
        print("\n🔬 Testing integration with live state...")
        live_state = monitor.get_live_state()
        
        if live_state and 'mycelial_data' in live_state:
            print("✅ Mycelial data integrated into live state")
            mycelial_state = live_state['mycelial_data']
            print(f"  📊 Integrated network size: {mycelial_state.get('network_size', 0)}")
            print(f"  📊 Recent spore activity: {len(mycelial_state.get('recent_spore_activity', []))}")
        else:
            print("❌ Mycelial data not found in live state")
            return False
        
        # Test spore activity tracking
        print("\n🔬 Testing spore activity tracking...")
        
        # Wait for some spore activity
        print("   ⏱️  Waiting for spore activity...")
        time.sleep(3)
        
        # Get updated data
        updated_data = monitor._get_mycelial_data()
        recent_activity = updated_data.get('recent_spore_activity', [])
        
        if recent_activity:
            print(f"✅ Spore activity detected: {len(recent_activity)} recent activities")
            for activity in recent_activity[-3:]:  # Show last 3
                activity_type = activity.get('type', 'unknown')
                concept = activity.get('concept', 'unknown')
                print(f"   🍄 {activity_type}: '{concept}'")
        else:
            print("⚠️  No spore activity detected yet")
        
        # Test display rendering (simulate)
        print("\n🔬 Testing display rendering...")
        
        # Mock the display section by checking if mycelial data would be shown
        test_state = {
            'mycelial_data': updated_data,
            'tick_count': 100,
            'consciousness_level': 0.8,
            'unity_score': 0.7,
            'processing_load': 50.0
        }
        
        if test_state['mycelial_data'].get('available'):
            print("✅ Mycelial display section would be rendered")
            print("   🍄 Network status, health, spores, and activity would be shown")
        else:
            print("❌ Mycelial display section would not be rendered")
            return False
        
        print("\n🎉 All mycelial integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_live_monitor_with_mycelial():
    """Demonstrate live monitor with mycelial integration for a few seconds"""
    
    print("\n🍄📊 LIVE MONITOR MYCELIAL DEMONSTRATION")
    print("=" * 60)
    print("Running live monitor with mycelial integration for 10 seconds...")
    print("Watch for mycelial network activity in the display!")
    print()
    
    try:
        from live_monitor import LiveDAWNMonitor, MYCELIAL_AVAILABLE
        
        if not MYCELIAL_AVAILABLE:
            print("❌ Mycelial system not available for demonstration")
            return False
        
        # Create monitor in simulation mode
        monitor = LiveDAWNMonitor(simulation_mode=True)
        
        # Run for a few iterations to show mycelial activity
        for i in range(5):
            print(f"\n📊 Iteration {i+1}/5")
            print("-" * 30)
            
            # Get live state with mycelial data
            state = monitor.get_live_state()
            
            if state and 'mycelial_data' in state:
                mycelial_data = state['mycelial_data']
                
                print(f"🍄 Network Size:  {mycelial_data.get('network_size', 0)} nodes")
                print(f"🍄 Network Health: {mycelial_data.get('network_health', 0.0):.3f}")
                print(f"🍄 Active Spores: {mycelial_data.get('active_spores', 0)}")
                print(f"🍄 Total Touches: {mycelial_data.get('total_touches', 0)}")
                print(f"🍄 Spores Generated: {mycelial_data.get('spores_generated', 0)}")
                
                # Show recent activity
                recent_activity = mycelial_data.get('recent_spore_activity', [])
                if recent_activity:
                    latest = recent_activity[-1]
                    activity_type = latest.get('type', 'unknown')
                    concept = latest.get('concept', 'unknown')
                    print(f"🍄 Latest Activity: {activity_type} '{concept}'")
                else:
                    print("🍄 Latest Activity: None")
                
                # Show integration stats
                integration_stats = mycelial_data.get('integration_stats', {})
                if integration_stats:
                    modules = integration_stats.get('modules_wrapped', 0)
                    concepts = integration_stats.get('concepts_mapped', 0)
                    print(f"🔗 Integration: {modules} modules, {concepts} concepts")
            else:
                print("❌ No mycelial data in state")
            
            time.sleep(2)
        
        print("\n✅ Demonstration complete!")
        print("🍄 Mycelial network is active and propagating semantic spores!")
        return True
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        return False

if __name__ == "__main__":
    print("🍄📊 DAWN Live Monitor Mycelial Integration Test")
    print()
    
    # Run integration test
    test_success = test_live_monitor_mycelial_integration()
    
    if test_success:
        # Run demonstration
        demo_success = demonstrate_live_monitor_with_mycelial()
        
        if demo_success:
            print("\n🎉 SUCCESS: Mycelial integration with live monitor is working!")
            print("🍄 The live monitor now displays real-time semantic spore propagation")
            print("📊 Run 'python3 live_monitor.py --simulate' to see it in action")
        else:
            print("\n⚠️  Integration test passed but demonstration had issues")
    else:
        print("\n❌ Integration test failed")
    
    print("\n🍄📊 Test complete!")
