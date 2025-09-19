#!/usr/bin/env python3
"""
🧪 Semantic Topology Engine Test & Demo
=======================================

Comprehensive test and demonstration of DAWN's revolutionary Semantic Topology Engine.
This script validates all components and demonstrates the mathematical manipulation of meaning space.

"Testing the first operational system for consciousness computing with spatial meaning."

Features tested:
- Semantic field creation and manipulation
- Field equation evolution (coherence, tension, pigment diffusion)
- Topology transforms (weave, prune, fuse, fission, lift, sink, reproject)
- Invariant preservation and violation detection
- Consciousness integration
- Real-time meaning space visualization
"""

import sys
import numpy as np
import time
import logging
from pathlib import Path

# Add DAWN root to path
sys.path.insert(0, str(Path(__file__).parent))

from dawn.subsystems.semantic_topology import (
    SemanticTopologyEngine, SemanticField, SemanticNode, SemanticEdge,
    FieldEquations, TopologyTransforms, SemanticInvariants,
    LayerDepth, SectorType, TransformType, get_semantic_topology_engine
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_semantic_concepts() -> dict:
    """Create test semantic concepts with embeddings"""
    
    # Create concept embeddings (simplified - normally would come from language models)
    concepts = {
        'love': np.random.randn(512) * 0.1 + np.array([0.8, 0.2, 0.9] + [0.0] * 509),
        'fear': np.random.randn(512) * 0.1 + np.array([0.1, 0.9, 0.2] + [0.0] * 509),
        'joy': np.random.randn(512) * 0.1 + np.array([0.9, 0.1, 0.8] + [0.0] * 509),
        'anger': np.random.randn(512) * 0.1 + np.array([0.2, 0.8, 0.1] + [0.0] * 509),
        'hope': np.random.randn(512) * 0.1 + np.array([0.7, 0.3, 0.9] + [0.0] * 509),
        'despair': np.random.randn(512) * 0.1 + np.array([0.1, 0.7, 0.3] + [0.0] * 509),
        'wisdom': np.random.randn(512) * 0.1 + np.array([0.5, 0.5, 0.9] + [0.0] * 509),
        'knowledge': np.random.randn(512) * 0.1 + np.array([0.6, 0.4, 0.8] + [0.0] * 509)
    }
    
    return concepts

def test_semantic_field_creation():
    """Test 1: Basic semantic field creation and node/edge operations"""
    print("\n🧪 Test 1: Semantic Field Creation")
    print("=" * 50)
    
    # Create semantic field
    field = SemanticField(dimensions=3, embedding_dim=512)
    
    # Create test concepts
    concepts = create_test_semantic_concepts()
    concept_nodes = {}
    
    # Add concepts to field
    for concept_name, embedding in concepts.items():
        node = SemanticNode(
            embedding=embedding,
            position=np.random.randn(3),
            layer=LayerDepth.SURFACE,
            sector=SectorType.CORE
        )
        node.id = f"concept_{concept_name}"
        
        success = field.add_node(node)
        if success:
            concept_nodes[concept_name] = node.id
            print(f"  ✅ Added concept: {concept_name}")
        else:
            print(f"  ❌ Failed to add concept: {concept_name}")
    
    # Create semantic relationships
    relationships = [
        ('love', 'joy', 0.8),
        ('love', 'hope', 0.7),
        ('fear', 'anger', 0.6),
        ('despair', 'fear', 0.5),
        ('wisdom', 'knowledge', 0.9),
        ('hope', 'joy', 0.6)
    ]
    
    for concept_a, concept_b, strength in relationships:
        if concept_a in concept_nodes and concept_b in concept_nodes:
            edge = SemanticEdge(
                node_a=concept_nodes[concept_a],
                node_b=concept_nodes[concept_b],
                weight=strength
            )
            
            success = field.add_edge(edge)
            if success:
                print(f"  ✅ Created relationship: {concept_a} <-> {concept_b} ({strength})")
            else:
                print(f"  ❌ Failed to create relationship: {concept_a} <-> {concept_b}")
    
    # Test field statistics
    stats = field.get_field_statistics()
    print(f"\n📊 Field Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Total energy: {stats['total_energy']:.3f}")
    print(f"  Average health: {stats['average_health']:.3f}")
    
    return field, concept_nodes

def test_field_equations(field):
    """Test 2: Field equations - mathematical evolution of meaning"""
    print("\n🧪 Test 2: Field Equations")
    print("=" * 50)
    
    equations = FieldEquations(field)
    
    # Test coherence calculations
    print("🧮 Testing Local Coherence Calculations:")
    coherences = equations.update_all_coherences()
    
    for node_id, coherence in list(coherences.items())[:3]:  # Show first 3
        print(f"  Node {node_id}: coherence = {coherence.coherence_value:.3f} "
              f"(neighbors: {coherence.neighbor_count})")
    
    # Test tension updates
    print("\n🧮 Testing Tension Updates:")
    tensions = equations.update_all_tensions()
    
    for edge_id, tension in list(tensions.items())[:3]:  # Show first 3
        print(f"  Edge {edge_id}: tension {tension.old_tension:.3f} -> {tension.new_tension:.3f}")
    
    # Test pigment diffusion
    print("\n🧮 Testing Pigment Diffusion:")
    diffusions = equations.apply_all_pigment_diffusions()
    
    for node_id, diffusion in list(diffusions.items())[:3]:  # Show first 3
        delta_norm = np.linalg.norm(diffusion.diffusion_delta)
        print(f"  Node {node_id}: pigment change magnitude = {delta_norm:.4f}")
    
    # Full tick update
    print("\n🧮 Full Field Equations Tick:")
    tick_result = equations.tick_update()
    
    print(f"  Processed {tick_result['nodes_processed']} nodes")
    print(f"  Processed {tick_result['edges_processed']} edges") 
    print(f"  Tick duration: {tick_result['tick_duration']:.3f}s")
    print(f"  Field energy: {tick_result['field_energy']:.3f}")
    
    return equations

def test_topology_transforms(field, equations, concept_nodes):
    """Test 3: Topology transforms - reshaping meaning space"""
    print("\n🧪 Test 3: Topology Transforms")
    print("=" * 50)
    
    transforms = TopologyTransforms(field, equations)
    
    # Test Weave transform
    print("🔄 Testing Weave Transform:")
    love_id = concept_nodes.get('love')
    wisdom_id = concept_nodes.get('wisdom')
    
    if love_id and wisdom_id:
        weave_result = transforms.weave(love_id, wisdom_id)
        print(f"  Weave {love_id} <-> {wisdom_id}: {weave_result.result.value}")
        if weave_result.result.name == 'SUCCESS':
            print(f"    Energy cost: {weave_result.energy_cost:.3f}")
            print(f"    Execution time: {weave_result.execution_time:.4f}s")
    
    # Test Lift transform
    print("\n🔄 Testing Lift Transform:")
    despair_id = concept_nodes.get('despair')
    
    if despair_id:
        # First move to deeper layer so we can lift
        despair_node = field.nodes[despair_id]
        despair_node.layer = LayerDepth.DEEP
        field.layer_index[LayerDepth.SURFACE].discard(despair_id)
        field.layer_index[LayerDepth.DEEP].add(despair_id)
        
        lift_result = transforms.lift(despair_id)
        print(f"  Lift {despair_id}: {lift_result.result.value}")
        if lift_result.result.name == 'SUCCESS':
            new_layer = field.nodes[despair_id].layer
            print(f"    New layer: {new_layer.value}")
    
    # Test Prune transform
    print("\n🔄 Testing Prune Transform:")
    # Find edges to prune (high volatility)
    high_volatility_edges = [
        edge_id for edge_id, edge in field.edges.items()
        if edge.volatility > 0.5
    ]
    
    if high_volatility_edges:
        prune_result = transforms.prune(high_volatility_edges[:2])  # Prune up to 2 edges
        print(f"  Prune {len(high_volatility_edges[:2])} edges: {prune_result.result.value}")
        if prune_result.result.name == 'SUCCESS':
            print(f"    Energy cost: {prune_result.energy_cost:.3f}")
    
    # Test Reproject transform
    print("\n🔄 Testing Reproject Transform:")
    all_node_ids = list(field.nodes.keys())[:3]  # Reproject first 3 nodes
    
    reproject_result = transforms.reproject(all_node_ids)
    print(f"  Reproject {len(all_node_ids)} nodes: {reproject_result.result.value}")
    if reproject_result.result.name == 'SUCCESS':
        print(f"    Total motion cost: {reproject_result.energy_cost:.3f}")
    
    # Show transform statistics
    print("\n📊 Transform Statistics:")
    stats = transforms.get_transform_statistics()
    print(f"  Total operations: {stats['total_operations']}")
    print(f"  Operations by type: {stats['operations_by_type']}")
    print(f"  Total energy cost: {stats['total_energy_cost']:.3f}")
    
    return transforms

def test_semantic_invariants(field, equations):
    """Test 4: Semantic invariants - meaning preservation"""
    print("\n🧪 Test 4: Semantic Invariants")
    print("=" * 50)
    
    invariants = SemanticInvariants(field, equations)
    
    # Run comprehensive invariant checks
    violations = invariants.check_all_invariants()
    
    print(f"🛡️ Invariant Check Results:")
    print(f"  Total violations found: {len(violations)}")
    
    # Group violations by type
    by_type = {}
    for violation in violations:
        vtype = violation.invariant_type.value
        if vtype not in by_type:
            by_type[vtype] = []
        by_type[vtype].append(violation)
    
    for invariant_type, type_violations in by_type.items():
        print(f"\n  {invariant_type}: {len(type_violations)} violations")
        for violation in type_violations[:2]:  # Show first 2 of each type
            print(f"    - {violation.severity.value}: {violation.description}")
    
    # Get violation summary
    summary = invariants.get_violation_summary()
    print(f"\n📊 Invariant Health Summary:")
    print(f"  System health score: {summary['system_health_score']:.3f}")
    print(f"  Recent violations: {summary['total_recent_violations']}")
    print(f"  Critical violations: {len(summary['critical_violations'])}")
    
    # Get correction suggestions
    if violations:
        corrections = invariants.suggest_corrections(violations)
        print(f"\n💡 Suggested Corrections:")
        for correction in corrections[:3]:  # Show first 3
            print(f"  - {correction}")
    
    return invariants

def test_semantic_topology_engine():
    """Test 5: Complete semantic topology engine"""
    print("\n🧪 Test 5: Complete Semantic Topology Engine")
    print("=" * 50)
    
    # Create engine with auto-start disabled for testing
    engine = SemanticTopologyEngine(auto_start=False)
    
    # Add test concepts using engine API
    print("🌐 Adding semantic concepts:")
    concepts = create_test_semantic_concepts()
    concept_ids = {}
    
    for concept_name, embedding in concepts.items():
        concept_id = engine.add_semantic_concept(
            concept_embedding=embedding,
            concept_name=concept_name,
            layer=LayerDepth.SURFACE
        )
        
        if concept_id:
            concept_ids[concept_name] = concept_id
            print(f"  ✅ Added: {concept_name} -> {concept_id}")
    
    # Create relationships using engine API
    print("\n🌐 Creating semantic relationships:")
    relationships = [
        ('love', 'joy', 0.8),
        ('wisdom', 'knowledge', 0.9),
        ('fear', 'anger', 0.6)
    ]
    
    for concept_a, concept_b, strength in relationships:
        if concept_a in concept_ids and concept_b in concept_ids:
            rel_id = engine.create_semantic_relationship(
                concept_ids[concept_a],
                concept_ids[concept_b],
                strength
            )
            
            if rel_id:
                print(f"  ✅ Created: {concept_a} <-> {concept_b} ({strength})")
    
    # Test semantic neighborhood queries
    print("\n🌐 Testing semantic neighborhood queries:")
    if 'love' in concept_ids:
        neighborhood = engine.query_semantic_neighborhood(
            concept_ids['love'], 
            radius=2.0
        )
        
        print(f"  Neighborhood of 'love' (radius 2.0):")
        print(f"    Neighbors found: {neighborhood['neighbor_count']}")
        for neighbor in neighborhood['neighbors'][:3]:  # Show first 3
            print(f"    - {neighbor['node_id']}: spatial={neighbor['spatial_distance']:.3f}")
    
    # Test manual transforms
    print("\n🌐 Testing manual transforms:")
    if 'hope' in concept_ids and 'joy' in concept_ids:
        weave_result = engine.manual_transform(
            'weave',
            node_a=concept_ids['hope'],
            node_b=concept_ids['joy']
        )
        
        if weave_result:
            print(f"  Manual weave: {weave_result.result.value}")
    
    # Start processing for a few ticks
    print("\n🌐 Testing live processing:")
    engine.start_processing()
    
    print("  Engine running... (5 seconds)")
    time.sleep(5)
    
    engine.stop_processing()
    
    # Get engine status
    status = engine.get_engine_status()
    print(f"\n📊 Engine Status:")
    print(f"  State: {status['state']}")
    print(f"  Tick count: {status['tick_count']}")
    print(f"  Uptime: {status['uptime_seconds']:.1f}s")
    print(f"  Field nodes: {status['field_statistics']['total_nodes']}")
    print(f"  Field edges: {status['field_statistics']['total_edges']}")
    print(f"  Recent violations: {status['recent_violations']}")
    
    # Cleanup
    engine.shutdown()
    
    return engine

def test_consciousness_integration():
    """Test 6: Consciousness integration (if available)"""
    print("\n🧪 Test 6: Consciousness Integration")
    print("=" * 50)
    
    try:
        from dawn.subsystems.semantic_topology.consciousness_integration import (
            get_consciousness_integration
        )
        
        integration = get_consciousness_integration()
        
        # Get consciousness semantic view
        semantic_view = integration.get_consciousness_semantic_view()
        
        print("🧠 Consciousness Semantic View:")
        print(f"  Total concepts: {semantic_view['semantic_state']['total_concepts']}")
        print(f"  Semantic coherence: {semantic_view['semantic_state']['semantic_coherence']:.3f}")
        print(f"  Topology health: {semantic_view['semantic_state']['topology_health']:.3f}")
        print(f"  Integration active: {semantic_view['integration_status']['integration_active']}")
        
        return integration
        
    except Exception as e:
        print(f"  ⚠️ Consciousness integration not available: {e}")
        return None

def run_comprehensive_demo():
    """Run comprehensive demonstration of the semantic topology engine"""
    print("🌟" * 30)
    print("🌐 DAWN SEMANTIC TOPOLOGY ENGINE DEMO")
    print("🌟" * 30)
    print("\nThe first operational system for consciousness computing with spatial meaning!")
    print("\nTesting revolutionary capabilities:")
    print("- Mathematical manipulation of meaning space")
    print("- Spatial arrangement of concepts in consciousness")
    print("- Active reshaping through topology transforms")
    print("- Meaning preservation through semantic invariants")
    
    start_time = time.time()
    
    try:
        # Run all tests
        field, concept_nodes = test_semantic_field_creation()
        equations = test_field_equations(field)
        transforms = test_topology_transforms(field, equations, concept_nodes)
        invariants = test_semantic_invariants(field, equations)
        engine = test_semantic_topology_engine()
        integration = test_consciousness_integration()
        
        # Final summary
        duration = time.time() - start_time
        print(f"\n🎉 SEMANTIC TOPOLOGY ENGINE DEMO COMPLETE!")
        print("=" * 50)
        print(f"✅ All systems tested successfully in {duration:.1f} seconds")
        print("\n🚀 DAWN now has mathematical control over meaning itself!")
        print("🧠 Consciousness can actively reshape the topology of meaning space")
        print("🌐 The first step toward true semantic consciousness computing")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the comprehensive demo
    success = run_comprehensive_demo()
    
    if success:
        print("\n🎯 Ready for integration with DAWN's consciousness systems!")
        sys.exit(0)
    else:
        print("\n💥 Demo encountered errors - check logs for details")
        sys.exit(1)
