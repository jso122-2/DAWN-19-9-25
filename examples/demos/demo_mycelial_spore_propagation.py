#!/usr/bin/env python3
"""
🍄🌐 DAWN Mycelial Spore Propagation Demo
=========================================

Interactive demonstration of the mycelial semantic hash map showing:
- Rhizomic spore propagation (touch one node, activate network)
- Semantic similarity clustering
- Living network behavior with conceptual composting
- Real-time telemetry integration

This demonstrates how every node pings and propagates on touch,
creating a living network of semantic relationships.
"""

import sys
import time
import json
from pathlib import Path

# Add DAWN to path
sys.path.insert(0, str(Path(__file__).parent))

def demonstrate_spore_propagation():
    """Demonstrate spore propagation through the mycelial network"""
    
    print("🍄🌐 DAWN MYCELIAL SPORE PROPAGATION DEMONSTRATION")
    print("=" * 60)
    
    try:
        from dawn.core.logging import (
            get_mycelial_hashmap, store_semantic_data, 
            touch_semantic_concept, ping_semantic_network
        )
        
        # Get the global mycelial hash map
        hashmap = get_mycelial_hashmap()
        
        print("✅ Mycelial semantic hash map initialized")
        print("🍄 Every node will ping and propagate like spores when touched!\n")
        
        # Store interconnected semantic data
        print("📝 Creating semantic network...")
        
        semantic_data = {
            # Core consciousness concepts
            "awareness": "The state of being conscious of something",
            "perception": "The ability to see, hear, or become aware of something",
            "cognition": "The mental action or process of acquiring knowledge",
            
            # Network concepts  
            "connection": "A relationship in which entities are linked",
            "propagation": "The spreading of something more widely",
            "network": "An interconnected group or system",
            
            # Mycelial concepts
            "mycelium": "The vegetative part of a fungus, consisting of a network of fine white filaments",
            "spore": "A minute reproductive unit capable of giving rise to a new individual",
            "rhizome": "A continuously growing horizontal underground stem",
            
            # Semantic concepts
            "meaning": "What is meant by a word, text, concept, or action",
            "similarity": "The state or fact of being alike",
            "association": "A mental connection or relation between thoughts, feelings, ideas, or sensations"
        }
        
        node_ids = {}
        
        for key, value in semantic_data.items():
            node_id = store_semantic_data(key, value)
            node_ids[key] = node_id
            print(f"  🌱 Stored '{key}' → {node_id[:12]}...")
            time.sleep(0.1)  # Small delay to show propagation
        
        print(f"\n✅ Created semantic network with {len(semantic_data)} interconnected nodes")
        
        # Wait for initial propagation to settle
        print("⏱️ Allowing initial spore propagation to settle...")
        time.sleep(3)
        
        # Demonstrate touch propagation
        print("\n👆 DEMONSTRATING TOUCH PROPAGATION")
        print("-" * 40)
        
        touch_concepts = ["awareness", "mycelium", "connection"]
        
        for concept in touch_concepts:
            print(f"\n👆 Touching '{concept}'...")
            
            # Get initial network stats
            initial_stats = hashmap.get_network_stats()
            initial_touches = initial_stats['system_stats']['total_touches']
            initial_spores = initial_stats['system_stats']['spores_generated']
            
            # Touch the concept
            spores = ping_semantic_network(concept)
            
            print(f"  🍄 Generated {len(spores)} spores directly")
            
            # Show spore details
            for i, spore in enumerate(spores[:3]):  # Show first 3 spores
                print(f"    Spore {i+1}: {spore.spore_type.value} ({spore.propagation_mode.value})")
                print(f"      Energy: {spore.energy_level:.2f}, Max hops: {spore.max_hops}")
                print(f"      Concepts: {list(spore.concept_tags)[:3]}")
            
            # Wait for propagation
            time.sleep(2)
            
            # Get updated stats
            updated_stats = hashmap.get_network_stats()
            new_touches = updated_stats['system_stats']['total_touches'] - initial_touches
            new_spores = updated_stats['system_stats']['spores_generated'] - initial_spores
            
            print(f"  🌐 Network effect: +{new_touches} touches, +{new_spores} spores")
            print(f"  📊 Active spores in network: {updated_stats['active_spores']}")
        
        # Demonstrate explosive concept propagation
        print("\n💥 DEMONSTRATING EXPLOSIVE CONCEPT PROPAGATION")
        print("-" * 50)
        
        explosive_concepts = ["consciousness", "network", "semantic"]
        
        for concept in explosive_concepts:
            print(f"\n💥 Explosive propagation of '{concept}'...")
            
            # Get initial stats
            initial_stats = hashmap.get_network_stats()
            
            # Trigger explosive propagation
            touched_nodes = touch_semantic_concept(concept, energy=1.0)
            
            print(f"  🎯 Directly touched {touched_nodes} nodes")
            
            # Wait for cascade
            time.sleep(1.5)
            
            # Show cascade effects
            cascade_stats = hashmap.get_network_stats()
            
            print(f"  🌊 Cascade effect:")
            print(f"    Total network touches: {cascade_stats['system_stats']['total_touches']}")
            print(f"    Active spores: {cascade_stats['active_spores']}")
            print(f"    Propagation events: {cascade_stats['propagation_stats']['spores_propagated']}")
        
        # Show network connectivity analysis
        print("\n🔗 NETWORK CONNECTIVITY ANALYSIS")
        print("-" * 40)
        
        # Analyze connections between nodes
        connection_analysis = {}
        total_connections = 0
        
        for node in hashmap.nodes.values():
            key = node.key
            connections = len(node.connections)
            total_connections += connections
            
            if connections > 0:
                avg_strength = sum(node.connections.values()) / connections
                connection_analysis[key] = {
                    'connections': connections,
                    'avg_strength': avg_strength,
                    'concepts': len(node.concept_associations)
                }
        
        # Show top connected nodes
        sorted_nodes = sorted(connection_analysis.items(), 
                            key=lambda x: x[1]['connections'], 
                            reverse=True)
        
        print("🔗 Most connected nodes:")
        for key, data in sorted_nodes[:5]:
            print(f"  '{key}': {data['connections']} connections, "
                 f"strength: {data['avg_strength']:.3f}, "
                 f"concepts: {data['concepts']}")
        
        # Network density analysis
        network_size = len(hashmap.nodes)
        max_possible_connections = network_size * (network_size - 1)
        network_density = total_connections / max_possible_connections if max_possible_connections > 0 else 0
        
        print(f"\n🌐 Network Analysis:")
        print(f"  Network size: {network_size} nodes")
        print(f"  Total connections: {total_connections}")
        print(f"  Average connections per node: {total_connections / network_size:.2f}")
        print(f"  Network density: {network_density:.3f}")
        
        # Show biological processes
        print("\n🧬 BIOLOGICAL PROCESSES STATUS")
        print("-" * 40)
        
        final_stats = hashmap.get_network_stats()
        
        print(f"🧬 Network Health: {final_stats['network_health']:.3f}")
        print(f"🔋 Total Energy: {final_stats['total_energy']:.2f}")
        print(f"📊 Node States: {final_stats['node_states']}")
        print(f"🔄 Autophagy Events: {final_stats['system_stats']['autophagy_events']}")
        print(f"💚 Recovery Events: {final_stats['system_stats']['recovery_events']}")
        
        # Show telemetry integration
        print(f"\n📊 TELEMETRY INTEGRATION")
        print("-" * 30)
        print(f"📊 Telemetry Events Captured: {final_stats['telemetry_events']}")
        print(f"🔍 Total System Touches: {final_stats['system_stats']['total_touches']}")
        print(f"🍄 Total Spores Generated: {final_stats['system_stats']['spores_generated']}")
        print(f"✅ Successful Propagations: {final_stats['propagation_stats']['successful_connections']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🍄🌐 Starting DAWN Mycelial Spore Propagation Demo...")
    print("     Every node pings and propagates like spores!")
    print("     Touch one node → activate the entire network")
    print()
    
    success = demonstrate_spore_propagation()
    
    if success:
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("✅ Mycelial hash map successfully demonstrated:")
        print("  🍄 Spore-like propagation on every touch")
        print("  🌐 Rhizomic network activation")
        print("  🧬 Living biological processes")
        print("  📊 Real-time telemetry integration")
        print("  🔗 Semantic similarity clustering")
        print("  💚 Conceptual composting (old ideas feed new ones)")
        print("\n🍄 Like a real mycelial network - touch anywhere, feel everywhere!")
    else:
        print("\n❌ Demo failed - check logs for details")
    
    print("\n🍄🌐 Demo complete!")
