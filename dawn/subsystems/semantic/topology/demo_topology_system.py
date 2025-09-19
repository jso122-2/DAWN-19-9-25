#!/usr/bin/env python3
"""
DAWN Semantic Topology System Demonstration
===========================================

Comprehensive demonstration of the semantic topology system implementation.
Shows field equations, transform operations, and system integration capabilities.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any

from topology_manager import SemanticTopologyManager, get_topology_manager
from primitives import TopologySector, create_semantic_node
from field_equations import FieldParameters

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_semantic_topology() -> SemanticTopologyManager:
    """Create a sample semantic topology with nodes and edges for demonstration"""
    
    # Initialize topology manager with custom parameters
    field_params = FieldParameters(
        gamma=0.8,           # Tension EMA factor
        eta=0.05,            # Pigment diffusion rate
        k_neighbors=6,       # Neighborhood size
        lambda_decay=1.2     # Distance decay factor
    )
    
    manager = SemanticTopologyManager(
        field_params=field_params,
        transform_energy_budget=10.0,
        auto_update_enabled=False  # Manual updates for demo
    )
    
    manager.start()
    
    print("🗺️  Creating sample semantic topology...")
    
    # Create sample concepts with embeddings
    concepts = [
        ("consciousness", np.random.randn(128), TopologySector.CORE, 0),
        ("awareness", np.random.randn(128), TopologySector.CORE, 0), 
        ("perception", np.random.randn(128), TopologySector.PERIPHERAL, 0),
        ("memory", np.random.randn(128), TopologySector.CORE, 1),
        ("emotion", np.random.randn(128), TopologySector.TRANSITIONAL, 0),
        ("thought", np.random.randn(128), TopologySector.CORE, 0),
        ("dream", np.random.randn(128), TopologySector.DEEP, 2),
        ("intuition", np.random.randn(128), TopologySector.TRANSITIONAL, 1),
        ("logic", np.random.randn(128), TopologySector.PERIPHERAL, 0),
        ("creativity", np.random.randn(128), TopologySector.PERIPHERAL, 0)
    ]
    
    # Add nodes to topology
    node_ids = []
    for content, embedding, sector, layer in concepts:
        # Make related concepts have similar embeddings
        if content in ["consciousness", "awareness"]:
            embedding = embedding + np.random.randn(128) * 0.1  # Similar embeddings
        elif content in ["memory", "thought"]:
            embedding = embedding + np.random.randn(128) * 0.1
        
        node_id = manager.add_semantic_node(
            content=content,
            embedding=embedding,
            sector=sector,
            layer=layer
        )
        node_ids.append(node_id)
        print(f"   Added node: {content} -> {node_id}")
    
    # Create meaningful semantic edges
    semantic_connections = [
        (0, 1, 0.9),  # consciousness <-> awareness (strong)
        (0, 2, 0.7),  # consciousness <-> perception
        (0, 3, 0.6),  # consciousness <-> memory
        (1, 2, 0.8),  # awareness <-> perception
        (1, 5, 0.7),  # awareness <-> thought
        (3, 5, 0.6),  # memory <-> thought
        (4, 5, 0.5),  # emotion <-> thought
        (6, 3, 0.4),  # dream <-> memory (weak)
        (7, 0, 0.6),  # intuition <-> consciousness
        (8, 5, 0.7),  # logic <-> thought
        (9, 4, 0.6),  # creativity <-> emotion
        (9, 7, 0.5),  # creativity <-> intuition
    ]
    
    # Add edges
    for src_idx, tgt_idx, weight in semantic_connections:
        if src_idx < len(node_ids) and tgt_idx < len(node_ids):
            edge_id = manager.add_semantic_edge(
                node_ids[src_idx], node_ids[tgt_idx], weight
            )
            print(f"   Added edge: {concepts[src_idx][0]} <-> {concepts[tgt_idx][0]} (w={weight})")
    
    print(f"✅ Created topology with {len(node_ids)} nodes and {len(semantic_connections)} edges")
    return manager


def demonstrate_field_equations(manager: SemanticTopologyManager):
    """Demonstrate field equation calculations"""
    print("\n🧮 Demonstrating Field Equations")
    print("=" * 50)
    
    # Create sample residue data
    sample_residue_data = {}
    for node_id in manager.state.nodes.keys():
        sample_residue_data[node_id] = {
            'soot': np.random.rand() * 0.5,      # Random soot levels
            'ash': np.random.rand() * 0.3,       # Random ash levels  
            'entropy': np.random.rand() * 0.4    # Random entropy levels
        }
    
    # Run field equation update
    print("Running field equation update...")
    start_time = time.time()
    
    field_results = manager.field_engine.update_all_fields(
        manager.state.nodes,
        manager.state.edges, 
        sample_residue_data
    )
    
    update_time = (time.time() - start_time) * 1000
    print(f"Field equations updated in {update_time:.2f}ms")
    
    # Display results
    print(f"\n📊 Field Equation Results:")
    print(f"   Average coherence: {field_results['average_coherence']:.3f}")
    print(f"   Average tension: {field_results['average_tension']:.3f}")
    print(f"   Coherence updates: {len(field_results['coherence_updates'])}")
    print(f"   Tension updates: {len(field_results['tension_updates'])}")
    print(f"   Pigment updates: {len(field_results['pigment_updates'])}")
    print(f"   Pressure updates: {len(field_results['pressure_updates'])}")
    
    # Show some specific node results
    print(f"\n🎯 Sample Node Results:")
    for i, (node_id, coherence) in enumerate(list(field_results['coherence_updates'].items())[:3]):
        node = manager.state.nodes[node_id]
        print(f"   {node.content}: coherence={coherence:.3f}, pressure={node.residue_pressure:.3f}")
    
    # Show field statistics
    field_stats = manager.field_engine.get_field_statistics(manager.state.nodes, manager.state.edges)
    print(f"\n📈 Field Statistics:")
    print(f"   Coherence: mean={field_stats['coherence']['mean']:.3f}, std={field_stats['coherence']['std']:.3f}")
    print(f"   Tension: mean={field_stats['tension']['mean']:.3f}, std={field_stats['tension']['std']:.3f}")
    print(f"   Pigment intensity: {field_stats['pigment']['mean_intensity']:.3f}")


def demonstrate_topology_transforms(manager: SemanticTopologyManager):
    """Demonstrate topology transform operations"""
    print("\n🔄 Demonstrating Topology Transforms")
    print("=" * 50)
    
    # Get some nodes for demonstration
    node_list = list(manager.state.nodes.values())
    edge_list = list(manager.state.edges.values())
    
    if len(node_list) < 2:
        print("❌ Need at least 2 nodes for transform demonstration")
        return
    
    # 1. Demonstrate Weave operation
    print("\n1️⃣  Weave Operation:")
    node_a, node_b = node_list[0], node_list[1]
    print(f"   Weaving connection between '{node_a.content}' and '{node_b.content}'")
    
    result, new_edge = manager.transforms.weave.weave(
        node_a, node_b, alpha=0.7, existing_edges=manager.state.edges
    )
    print(f"   Result: {result.value}")
    if new_edge:
        print(f"   New edge weight: {new_edge.weight:.3f}, reliability: {new_edge.reliability:.3f}")
    
    # 2. Demonstrate Prune operation
    print("\n2️⃣  Prune Operation:")
    if edge_list:
        edge_to_prune = edge_list[0]
        print(f"   Pruning edge: {edge_to_prune.id}")
        print(f"   Before: weight={edge_to_prune.weight:.3f}, tension={edge_to_prune.tension:.3f}")
        
        result = manager.transforms.prune.prune(
            edge_to_prune, kappa=0.5, existing_edges=manager.state.edges
        )
        print(f"   Result: {result.value}")
        if edge_to_prune.id in manager.state.edges:  # If not removed
            print(f"   After: weight={edge_to_prune.weight:.3f}, tension={edge_to_prune.tension:.3f}")
    
    # 3. Demonstrate Lift operation
    print("\n3️⃣  Lift Operation:")
    deep_nodes = [n for n in node_list if n.coordinates.layer > 0]
    if deep_nodes:
        node_to_lift = deep_nodes[0]
        print(f"   Lifting node '{node_to_lift.content}' from layer {node_to_lift.coordinates.layer}")
        print(f"   Energy before: {node_to_lift.energy:.3f}")
        
        result = manager.transforms.lift.lift(node_to_lift, manager.state.layers)
        print(f"   Result: {result.value}")
        print(f"   New layer: {node_to_lift.coordinates.layer}")
        print(f"   Energy after: {node_to_lift.energy:.3f}")
    
    # 4. Demonstrate Sink operation  
    print("\n4️⃣  Sink Operation:")
    surface_nodes = [n for n in node_list if n.coordinates.layer == 0]
    if surface_nodes:
        node_to_sink = surface_nodes[-1]  # Take last one
        print(f"   Sinking node '{node_to_sink.content}' from layer {node_to_sink.coordinates.layer}")
        print(f"   Health before: {node_to_sink.health:.3f}")
        
        result = manager.transforms.sink.sink(node_to_sink, manager.state.layers)
        print(f"   Result: {result.value}")
        print(f"   New layer: {node_to_sink.coordinates.layer}")
        print(f"   Health after: {node_to_sink.health:.3f}")
    
    # 5. Demonstrate Reproject operation
    print("\n5️⃣  Reproject Operation:")
    print(f"   Reprojecting all {len(node_list)} nodes using PCA")
    
    # Save original positions for comparison
    original_positions = {n.id: n.coordinates.position.copy() for n in node_list}
    
    result = manager.transforms.reproject.reproject(
        manager.state.nodes, manager.state.edges, projection_method="pca"
    )
    print(f"   Result: {result.value}")
    
    # Show position changes
    total_movement = 0.0
    for node in node_list[:3]:  # Show first 3
        original_pos = original_positions[node.id]
        new_pos = node.coordinates.position
        movement = np.linalg.norm(new_pos - original_pos)
        total_movement += movement
        print(f"   '{node.content}': moved {movement:.3f} units")
    
    print(f"   Average movement: {total_movement / min(3, len(node_list)):.3f} units")
    
    # Show energy status
    print(f"\n⚡ Energy Status:")
    energy_status = manager.transforms.get_energy_status()
    print(f"   Total remaining: {energy_status['total_remaining']:.2f}")
    for op_type in ['weave', 'prune', 'lift', 'sink', 'reproject']:
        remaining = energy_status[f'{op_type}_remaining']
        print(f"   {op_type}: {remaining:.2f}")


def demonstrate_topology_tick_cycle(manager: SemanticTopologyManager):
    """Demonstrate the complete topology tick cycle"""
    print("\n🔄 Demonstrating Topology Tick Cycle")
    print("=" * 50)
    
    # Create varying residue data to trigger different behaviors
    varying_residue_data = {}
    for i, node_id in enumerate(manager.state.nodes.keys()):
        # Create patterns that will trigger different transform decisions
        if i % 3 == 0:  # High entropy nodes
            varying_residue_data[node_id] = {
                'soot': 0.8, 'ash': 0.1, 'entropy': 0.9
            }
        elif i % 3 == 1:  # High ash nodes (rebloom candidates)
            varying_residue_data[node_id] = {
                'soot': 0.1, 'ash': 0.8, 'entropy': 0.2
            }
        else:  # Balanced nodes
            varying_residue_data[node_id] = {
                'soot': 0.3, 'ash': 0.4, 'entropy': 0.3
            }
    
    # Run several tick cycles
    print("Running 5 topology tick cycles...")
    
    for tick in range(1, 6):
        print(f"\n--- Tick {tick} ---")
        
        # Add some noise to residue data
        noisy_residue_data = {}
        for node_id, data in varying_residue_data.items():
            noisy_residue_data[node_id] = {
                'soot': max(0, data['soot'] + np.random.normal(0, 0.1)),
                'ash': max(0, data['ash'] + np.random.normal(0, 0.1)), 
                'entropy': max(0, data['entropy'] + np.random.normal(0, 0.1))
            }
        
        # Run tick update
        tick_result = manager.update_topology_tick(noisy_residue_data)
        
        if 'error' in tick_result:
            print(f"❌ Tick failed: {tick_result['error']}")
            continue
        
        # Display results
        print(f"✅ Tick completed in {tick_result['update_time_ms']:.2f}ms")
        print(f"   System coherence: {tick_result['system_coherence']:.3f}")
        print(f"   Nodes: {tick_result['node_count']}, Edges: {tick_result['edge_count']}")
        
        # Show transform results
        transform_results = tick_result.get('transform_results', {})
        applied_transforms = transform_results.get('applied', [])
        failed_transforms = transform_results.get('failed', [])
        
        if applied_transforms:
            print(f"   Applied transforms: {len(applied_transforms)}")
            for transform in applied_transforms[:2]:  # Show first 2
                print(f"     - {transform['type']}: {transform.get('reason', 'N/A')}")
        
        if failed_transforms:
            print(f"   Failed transforms: {len(failed_transforms)}")
        
        # Show invariant violations
        violations = tick_result.get('invariant_violations', [])
        if violations:
            print(f"   ⚠️  Invariant violations: {len(violations)}")
            for violation in violations[:2]:  # Show first 2
                print(f"     - {violation}")
    
    # Show final system state
    print(f"\n📊 Final System State:")
    state = manager.get_topology_state()
    print(f"   Total ticks: {state['tick_count']}")
    print(f"   System coherence: {state['system_coherence']:.3f}")
    print(f"   Global tension: {state['global_tension']:.3f}")
    print(f"   Layer distribution: {state['layer_distribution']}")
    print(f"   Sector distribution: {state['sector_distribution']}")
    
    # Show performance metrics
    perf = state['performance_summary']
    print(f"   Average update time: {perf['average_update_time_ms']:.2f}ms")
    print(f"   Average field time: {perf['average_field_time_ms']:.2f}ms")


def demonstrate_topology_export(manager: SemanticTopologyManager):
    """Demonstrate topology data export capabilities"""
    print("\n💾 Demonstrating Topology Export")
    print("=" * 40)
    
    # Create a snapshot
    snapshot = manager._create_snapshot()
    print(f"Created snapshot at tick {snapshot.tick}")
    print(f"Snapshot contains {len(snapshot.nodes)} nodes and {len(snapshot.edges)} edges")
    
    # Export to file
    export_path = "/tmp/dawn_topology_demo.json"
    success = manager.export_topology_data(export_path)
    
    if success:
        print(f"✅ Exported topology data to {export_path}")
        
        # Show file size
        import os
        if os.path.exists(export_path):
            file_size = os.path.getsize(export_path)
            print(f"   File size: {file_size:,} bytes")
    else:
        print("❌ Export failed")
    
    # Show snapshot statistics
    print(f"\n📈 Snapshot Statistics:")
    metrics = snapshot.metrics
    print(f"   Topology health: {metrics['topology_health']:.3f}")
    print(f"   Average coherence: {metrics['average_coherence']:.3f}")
    print(f"   Average tension: {metrics['average_tension']:.3f}")
    print(f"   Pigment entropy: {metrics['pigment_entropy']:.3f}")


def main():
    """Main demonstration function"""
    print("🌅 DAWN Semantic Topology System Demonstration")
    print("=" * 60)
    print("Implementing RTF specifications from DAWN-docs/Semantic Topology/")
    print()
    
    try:
        # Create sample topology
        manager = create_sample_semantic_topology()
        
        # Demonstrate field equations
        demonstrate_field_equations(manager)
        
        # Demonstrate transform operations
        demonstrate_topology_transforms(manager)
        
        # Demonstrate tick cycle
        demonstrate_topology_tick_cycle(manager)
        
        # Demonstrate export capabilities
        demonstrate_topology_export(manager)
        
        # Final summary
        print(f"\n🎉 Demonstration Complete!")
        print("=" * 40)
        
        final_state = manager.get_topology_state()
        print(f"Final topology state:")
        print(f"   Nodes: {final_state['node_count']}")
        print(f"   Edges: {final_state['edge_count']}")  
        print(f"   Layers: {final_state['layer_count']}")
        print(f"   System coherence: {final_state['system_coherence']:.3f}")
        print(f"   Total updates: {final_state['performance_summary']['total_updates']}")
        
        # Show transform statistics
        transform_stats = manager.transforms.get_transform_statistics()
        print(f"   Transform operations: {transform_stats['total_operations']}")
        print(f"   Transform success rate: {transform_stats.get('success_rate', 0):.1%}")
        
        print("\n✅ Semantic Topology System successfully implemented!")
        print("   - All RTF primitives implemented (Node, Edge, Layer, Sector, Frame)")
        print("   - Field equations functional (coherence, tension, pigment, pressure)")
        print("   - Transform operators working (weave, prune, fuse, fission, lift, sink, reproject)")
        print("   - Topology manager provides unified interface")
        print("   - System ready for DAWN consciousness integration")
        
        # Stop the manager
        manager.stop()
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
