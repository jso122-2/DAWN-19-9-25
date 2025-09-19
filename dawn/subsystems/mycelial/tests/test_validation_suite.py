"""
🔬 Comprehensive Validation Suite
=================================

Validates that the mycelial layer implementation follows the biological
principles and mathematical formulas specified in the DAWN documentation.
"""

import unittest
import numpy as np
import time
import logging
from typing import Dict, List, Any

# Import mycelial components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from integrated_system import IntegratedMycelialSystem
from core import MycelialNode, MycelialEdge, NodeState

# Suppress debug logs during testing
logging.getLogger().setLevel(logging.WARNING)

class ValidationSuite:
    """
    Comprehensive validation suite that tests the mycelial layer against
    the biological principles and mathematical specifications from DAWN docs.
    """
    
    def __init__(self):
        self.test_results = {}
        self.validation_errors = []
        
    def run_full_validation(self) -> Dict[str, Any]:
        """Run the complete validation suite"""
        print("🔬 Starting DAWN Mycelial Layer Validation Suite")
        print("=" * 60)
        
        # Core behavior validation
        self.validate_biological_principles()
        self.validate_mathematical_formulas()
        self.validate_energy_conservation()
        self.validate_nutrient_economy()
        self.validate_growth_gates()
        self.validate_autophagy_mechanisms()
        self.validate_metabolite_cycling()
        self.validate_cluster_dynamics()
        self.validate_system_integration()
        self.validate_performance_characteristics()
        
        # Generate report
        return self.generate_validation_report()
    
    def validate_biological_principles(self):
        """Validate core biological principles"""
        print("\n🧬 Validating Biological Principles...")
        
        system = IntegratedMycelialSystem(max_nodes=100)
        
        # Test 1: Dendritic growth with selective pruning
        print("  Testing dendritic growth...")
        system.add_node("neuron1", pressure=0.8, drift_alignment=0.7)
        system.add_node("neuron2", pressure=0.75, drift_alignment=0.72)
        system.add_node("neuron3", pressure=0.1, drift_alignment=0.05)
        
        # Run some ticks to allow growth
        for _ in range(10):
            system.tick_update()
        
        # Should have connections between similar nodes, not dissimilar ones
        neuron1 = system.mycelial_layer.nodes["neuron1"]
        connected_to_similar = "neuron2" in neuron1.connections
        connected_to_dissimilar = "neuron3" in neuron1.connections
        
        self.test_results['dendritic_growth'] = {
            'passed': connected_to_similar and not connected_to_dissimilar,
            'details': f"Connected to similar: {connected_to_similar}, to dissimilar: {connected_to_dissimilar}"
        }
        
        # Test 2: Mitochondrial resource sharing
        print("  Testing mitochondrial resource sharing...")
        
        # Set up energy imbalance
        neuron1.energy = 0.9  # High energy (blooming)
        if "neuron2" in neuron1.connections:
            neuron2 = system.mycelial_layer.nodes["neuron2"]
            neuron2.energy = 0.2  # Low energy (starving)
            
            initial_diff = neuron1.energy - neuron2.energy
            
            # Run energy flow update
            if system.energy_flow_manager:
                system.energy_flow_manager.tick_update()
                
                final_diff = neuron1.energy - neuron2.energy
                energy_shared = final_diff < initial_diff
                
                self.test_results['mitochondrial_sharing'] = {
                    'passed': energy_shared,
                    'details': f"Initial diff: {initial_diff:.3f}, final diff: {final_diff:.3f}"
                }
            else:
                self.test_results['mitochondrial_sharing'] = {
                    'passed': False,
                    'details': "Energy flow manager not enabled"
                }
        else:
            self.test_results['mitochondrial_sharing'] = {
                'passed': False,
                'details': "No connections formed for energy sharing test"
            }
        
        # Test 3: Living substrate (responds to real-time pressures)
        print("  Testing living substrate responsiveness...")
        
        initial_pressure = neuron1.pressure
        
        # Apply external pressure
        system.tick_update(external_pressures={'global_pressure': 0.3})
        
        pressure_increased = neuron1.pressure > initial_pressure
        
        self.test_results['living_substrate'] = {
            'passed': pressure_increased,
            'details': f"Pressure changed from {initial_pressure:.3f} to {neuron1.pressure:.3f}"
        }
        
        print(f"    ✓ Dendritic growth: {'PASS' if self.test_results['dendritic_growth']['passed'] else 'FAIL'}")
        print(f"    ✓ Mitochondrial sharing: {'PASS' if self.test_results['mitochondrial_sharing']['passed'] else 'FAIL'}")
        print(f"    ✓ Living substrate: {'PASS' if self.test_results['living_substrate']['passed'] else 'FAIL'}")
    
    def validate_mathematical_formulas(self):
        """Validate mathematical formulas from DAWN documentation"""
        print("\n📐 Validating Mathematical Formulas...")
        
        system = IntegratedMycelialSystem(max_nodes=50)
        
        # Add test nodes
        system.add_node("test1", pressure=0.6, drift_alignment=0.7, recency=0.8, entropy=0.3)
        system.add_node("test2", pressure=0.4, drift_alignment=0.5, recency=0.6, entropy=0.2)
        
        node1 = system.mycelial_layer.nodes["test1"]
        node2 = system.mycelial_layer.nodes["test2"]
        
        # Test 1: Demand calculation formula
        print("  Testing demand calculation (D_i = wP*P_i + wΔ*drift_align_i + wR*recency_i - wσ*σ_i)...")
        
        if system.nutrient_economy:
            # Manual calculation
            weights = system.nutrient_economy.budget
            expected_demand = (
                weights.pressure_weight * node1.pressure +
                weights.drift_weight * node1.drift_alignment +
                weights.recency_weight * node1.recency +
                weights.entropy_weight * node1.entropy
            )
            
            # System calculation
            computed_demand = system.nutrient_economy._compute_node_demand(node1)
            
            formula_correct = abs(expected_demand - computed_demand) < 0.01
            
            self.test_results['demand_formula'] = {
                'passed': formula_correct,
                'details': f"Expected: {expected_demand:.3f}, Computed: {computed_demand:.3f}"
            }
        else:
            self.test_results['demand_formula'] = {'passed': False, 'details': "Nutrient economy disabled"}
        
        # Test 2: Softmax allocation formula  
        print("  Testing softmax allocation (a_i = softmax(D)_i * B_t)...")
        
        if system.nutrient_economy:
            system.nutrient_economy.compute_all_demands()
            allocations = system.nutrient_economy.allocate_nutrients()
            
            # Sum of allocations should approximately equal budget allocated
            total_allocated = sum(allocations.values())
            budget_used = system.nutrient_economy.budget.allocated_this_tick
            
            allocation_correct = abs(total_allocated - budget_used) < 0.01
            
            self.test_results['allocation_formula'] = {
                'passed': allocation_correct,
                'details': f"Allocated: {total_allocated:.3f}, Budget used: {budget_used:.3f}"
            }
        else:
            self.test_results['allocation_formula'] = {'passed': False, 'details': "Nutrient economy disabled"}
        
        # Test 3: Energy update formula
        print("  Testing energy update (energy_i = clamp(energy_i + η*nutrients_i - basal_cost_i, 0, E_max))...")
        
        initial_energy = node1.energy
        nutrients_received = 0.2
        efficiency = 0.9
        basal_cost = node1.basal_cost
        
        expected_energy = max(0.0, min(
            initial_energy + efficiency * nutrients_received - basal_cost,
            node1.max_energy
        ))
        
        # Reset energy and apply formula
        node1.energy = initial_energy
        actual_change = node1.update_energy(nutrients_received, efficiency)
        actual_energy = node1.energy
        
        energy_formula_correct = abs(expected_energy - actual_energy) < 0.01
        
        self.test_results['energy_formula'] = {
            'passed': energy_formula_correct,
            'details': f"Expected: {expected_energy:.3f}, Actual: {actual_energy:.3f}"
        }
        
        print(f"    ✓ Demand formula: {'PASS' if self.test_results['demand_formula']['passed'] else 'FAIL'}")
        print(f"    ✓ Allocation formula: {'PASS' if self.test_results['allocation_formula']['passed'] else 'FAIL'}")
        print(f"    ✓ Energy formula: {'PASS' if self.test_results['energy_formula']['passed'] else 'FAIL'}")
    
    def validate_energy_conservation(self):
        """Validate energy conservation laws"""
        print("\n⚡ Validating Energy Conservation...")
        
        system = IntegratedMycelialSystem(max_nodes=20)
        
        # Add nodes with known energy
        total_initial_energy = 0.0
        for i in range(10):
            energy = 0.5 + i * 0.05  # Varied initial energies
            system.add_node(f"energy_test_{i}", energy=energy)
            total_initial_energy += energy
        
        # Force some connections
        for i in range(5):
            system.mycelial_layer.add_edge(f"energy_test_{i}", f"energy_test_{i+1}")
        
        # Record initial total energy
        initial_system_energy = sum(node.energy for node in system.mycelial_layer.nodes.values())
        
        # Run several ticks
        for _ in range(20):
            system.tick_update()
        
        # Check energy conservation
        final_system_energy = sum(node.energy for node in system.mycelial_layer.nodes.values())
        
        # Allow for some energy loss due to basal costs and inefficiencies
        energy_loss = initial_system_energy - final_system_energy
        acceptable_loss = initial_system_energy * 0.3  # 30% acceptable loss due to metabolism
        
        energy_conserved = 0 <= energy_loss <= acceptable_loss
        
        self.test_results['energy_conservation'] = {
            'passed': energy_conserved,
            'details': f"Initial: {initial_system_energy:.3f}, Final: {final_system_energy:.3f}, Loss: {energy_loss:.3f}"
        }
        
        print(f"    ✓ Energy conservation: {'PASS' if energy_conserved else 'FAIL'}")
    
    def validate_nutrient_economy(self):
        """Validate nutrient economy behavior"""
        print("\n🌱 Validating Nutrient Economy...")
        
        system = IntegratedMycelialSystem(max_nodes=30)
        
        # Test demand-based allocation
        system.add_node("high_demand", pressure=0.9, drift_alignment=0.8, recency=0.9, entropy=0.1)
        system.add_node("low_demand", pressure=0.1, drift_alignment=0.2, recency=0.1, entropy=0.8)
        
        if system.nutrient_economy:
            # Run allocation
            demands = system.nutrient_economy.compute_all_demands()
            allocations = system.nutrient_economy.allocate_nutrients()
            
            high_demand_node = demands.get("high_demand", 0)
            low_demand_node = demands.get("low_demand", 0)
            
            high_allocation = allocations.get("high_demand", 0)
            low_allocation = allocations.get("low_demand", 0)
            
            # High demand node should get more allocation
            demand_based_allocation = (high_demand_node > low_demand_node and 
                                     high_allocation > low_allocation)
            
            self.test_results['demand_based_allocation'] = {
                'passed': demand_based_allocation,
                'details': f"High demand: {high_demand_node:.3f}/{high_allocation:.3f}, Low: {low_demand_node:.3f}/{low_allocation:.3f}"
            }
            
            # Test budget conservation
            total_allocation = sum(allocations.values())
            budget_available = system.nutrient_economy.budget.budget_per_tick
            
            budget_respected = total_allocation <= budget_available * 1.01  # Small tolerance
            
            self.test_results['budget_conservation'] = {
                'passed': budget_respected,
                'details': f"Allocated: {total_allocation:.3f}, Budget: {budget_available:.3f}"
            }
        else:
            self.test_results['demand_based_allocation'] = {'passed': False, 'details': "Nutrient economy disabled"}
            self.test_results['budget_conservation'] = {'passed': False, 'details': "Nutrient economy disabled"}
        
        print(f"    ✓ Demand-based allocation: {'PASS' if self.test_results['demand_based_allocation']['passed'] else 'FAIL'}")
        print(f"    ✓ Budget conservation: {'PASS' if self.test_results['budget_conservation']['passed'] else 'FAIL'}")
    
    def validate_growth_gates(self):
        """Validate growth gate mechanisms"""
        print("\n🌱 Validating Growth Gates...")
        
        system = IntegratedMycelialSystem(max_nodes=20)
        
        if system.growth_gate:
            # Test energy threshold
            system.add_node("low_energy", energy=0.1)  # Below growth threshold
            system.add_node("high_energy", energy=0.8)  # Above growth threshold
            system.add_node("target", energy=0.5)
            
            low_energy_node = system.mycelial_layer.nodes["low_energy"]
            high_energy_node = system.mycelial_layer.nodes["high_energy"]
            target_node = system.mycelial_layer.nodes["target"]
            
            # Evaluate growth proposals
            low_decision, _ = system.growth_gate.evaluate_growth_proposal(low_energy_node, target_node, system.mycelial_layer)
            high_decision, _ = system.growth_gate.evaluate_growth_proposal(high_energy_node, target_node, system.mycelial_layer)
            
            energy_threshold_works = (low_decision.value == 'rejected_energy' and 
                                    high_decision.value != 'rejected_energy')
            
            self.test_results['energy_threshold'] = {
                'passed': energy_threshold_works,
                'details': f"Low energy: {low_decision.value}, High energy: {high_decision.value}"
            }
            
            # Test similarity threshold
            system.add_node("similar", pressure=0.7, drift_alignment=0.7, energy=0.8)
            system.add_node("dissimilar", pressure=0.1, drift_alignment=0.1, energy=0.8)
            system.add_node("reference", pressure=0.7, drift_alignment=0.7, energy=0.8)
            
            similar_node = system.mycelial_layer.nodes["similar"]
            dissimilar_node = system.mycelial_layer.nodes["dissimilar"]
            reference_node = system.mycelial_layer.nodes["reference"]
            
            similar_decision, _ = system.growth_gate.evaluate_growth_proposal(similar_node, reference_node, system.mycelial_layer)
            dissimilar_decision, _ = system.growth_gate.evaluate_growth_proposal(dissimilar_node, reference_node, system.mycelial_layer)
            
            similarity_threshold_works = (similar_decision.value == 'approved' and 
                                        dissimilar_decision.value == 'rejected_similarity')
            
            self.test_results['similarity_threshold'] = {
                'passed': similarity_threshold_works,
                'details': f"Similar: {similar_decision.value}, Dissimilar: {dissimilar_decision.value}"
            }
        else:
            self.test_results['energy_threshold'] = {'passed': False, 'details': "Growth gates disabled"}
            self.test_results['similarity_threshold'] = {'passed': False, 'details': "Growth gates disabled"}
        
        print(f"    ✓ Energy threshold: {'PASS' if self.test_results['energy_threshold']['passed'] else 'FAIL'}")
        print(f"    ✓ Similarity threshold: {'PASS' if self.test_results['similarity_threshold']['passed'] else 'FAIL'}")
    
    def validate_autophagy_mechanisms(self):
        """Validate autophagy and metabolite production"""
        print("\n🔄 Validating Autophagy Mechanisms...")
        
        system = IntegratedMycelialSystem(max_nodes=20)
        
        if system.autophagy_manager:
            # Create a starving node
            system.add_node("starving", energy=0.05)  # Very low energy
            starving_node = system.mycelial_layer.nodes["starving"]
            
            # Simulate starvation over time
            for _ in range(10):  # Should trigger autophagy
                starving_node.energy = 0.05
                starving_node._update_state()
                system.tick_update(delta_time=1.0)
            
            # Check if autophagy was triggered
            autophagy_triggered = starving_node.id in system.autophagy_manager.nodes_in_autophagy
            
            self.test_results['autophagy_trigger'] = {
                'passed': autophagy_triggered,
                'details': f"Node state: {getattr(starving_node, 'state', 'unknown')}, In autophagy: {autophagy_triggered}"
            }
            
            # Test metabolite production
            if system.metabolite_manager:
                initial_metabolites = len(system.metabolite_manager.active_metabolites)
                
                # Continue simulation to produce metabolites
                for _ in range(5):
                    system.tick_update(delta_time=1.0)
                
                final_metabolites = len(system.metabolite_manager.active_metabolites)
                metabolites_produced = final_metabolites > initial_metabolites
                
                self.test_results['metabolite_production'] = {
                    'passed': metabolites_produced,
                    'details': f"Initial: {initial_metabolites}, Final: {final_metabolites}"
                }
            else:
                self.test_results['metabolite_production'] = {'passed': False, 'details': "Metabolite manager disabled"}
        else:
            self.test_results['autophagy_trigger'] = {'passed': False, 'details': "Autophagy manager disabled"}
            self.test_results['metabolite_production'] = {'passed': False, 'details': "Autophagy manager disabled"}
        
        print(f"    ✓ Autophagy trigger: {'PASS' if self.test_results['autophagy_trigger']['passed'] else 'FAIL'}")
        print(f"    ✓ Metabolite production: {'PASS' if self.test_results['metabolite_production']['passed'] else 'FAIL'}")
    
    def validate_metabolite_cycling(self):
        """Validate metabolite absorption and recycling"""
        print("\n🧬 Validating Metabolite Cycling...")
        
        system = IntegratedMycelialSystem(max_nodes=20)
        
        if system.metabolite_manager:
            # Create nodes for metabolite testing
            system.add_node("producer", energy=0.05)
            system.add_node("consumer", energy=0.4)  # Has capacity to absorb
            
            # Create connection for metabolite distribution
            system.mycelial_layer.add_edge("producer", "consumer")
            
            producer = system.mycelial_layer.nodes["producer"]
            consumer = system.mycelial_layer.nodes["consumer"]
            
            # Manually create a metabolite
            from metabolites import MetaboliteType
            metabolite = system.metabolite_manager.produce_metabolite(
                source_node=producer,
                metabolite_type=MetaboliteType.ENERGY_RESIDUE,
                energy_content=0.1,
                semantic_content={'test': True}
            )
            
            initial_consumer_energy = consumer.energy
            
            # Process metabolite distribution and absorption
            system.metabolite_manager.distribute_metabolites()
            system.metabolite_manager.process_absorptions()
            
            # Check if consumer absorbed energy
            energy_absorbed = consumer.energy > initial_consumer_energy
            
            self.test_results['metabolite_absorption'] = {
                'passed': energy_absorbed,
                'details': f"Consumer energy: {initial_consumer_energy:.3f} -> {consumer.energy:.3f}"
            }
            
            # Test metabolite decay
            initial_potency = metabolite.potency
            for _ in range(10):
                metabolite.decay(delta_time=1.0)
            
            potency_decayed = metabolite.potency < initial_potency
            
            self.test_results['metabolite_decay'] = {
                'passed': potency_decayed,
                'details': f"Potency: {initial_potency:.3f} -> {metabolite.potency:.3f}"
            }
        else:
            self.test_results['metabolite_absorption'] = {'passed': False, 'details': "Metabolite manager disabled"}
            self.test_results['metabolite_decay'] = {'passed': False, 'details': "Metabolite manager disabled"}
        
        print(f"    ✓ Metabolite absorption: {'PASS' if self.test_results['metabolite_absorption']['passed'] else 'FAIL'}")
        print(f"    ✓ Metabolite decay: {'PASS' if self.test_results['metabolite_decay']['passed'] else 'FAIL'}")
    
    def validate_cluster_dynamics(self):
        """Validate cluster fusion and fission"""
        print("\n🔄 Validating Cluster Dynamics...")
        
        system = IntegratedMycelialSystem(max_nodes=50, enable_clustering=True)
        
        if system.cluster_manager:
            # Create nodes in clusters
            for i in range(20):
                system.add_node(f"cluster_test_{i}", energy=0.7, health=0.8)
            
            # Run clustering
            for _ in range(20):
                system.tick_update()
            
            # Check if clusters were formed
            clusters_formed = len(system.cluster_manager.clusters) > 0
            
            self.test_results['cluster_formation'] = {
                'passed': clusters_formed,
                'details': f"Clusters formed: {len(system.cluster_manager.clusters)}"
            }
            
            # Test cluster metrics
            if clusters_formed:
                cluster_stats = system.cluster_manager.get_cluster_statistics()
                
                efficiency = cluster_stats['clustering_efficiency']
                metrics_valid = 0.0 <= efficiency <= 1.0
                
                self.test_results['cluster_metrics'] = {
                    'passed': metrics_valid,
                    'details': f"Clustering efficiency: {efficiency:.3f}"
                }
            else:
                self.test_results['cluster_metrics'] = {'passed': False, 'details': "No clusters formed"}
        else:
            self.test_results['cluster_formation'] = {'passed': False, 'details': "Cluster manager disabled"}
            self.test_results['cluster_metrics'] = {'passed': False, 'details': "Cluster manager disabled"}
        
        print(f"    ✓ Cluster formation: {'PASS' if self.test_results['cluster_formation']['passed'] else 'FAIL'}")
        print(f"    ✓ Cluster metrics: {'PASS' if self.test_results['cluster_metrics']['passed'] else 'FAIL'}")
    
    def validate_system_integration(self):
        """Validate system integration and coordination"""
        print("\n🧠 Validating System Integration...")
        
        system = IntegratedMycelialSystem(max_nodes=30)
        
        # Test full system tick
        initial_state = system.get_system_status()
        
        # Run several ticks
        tick_results = []
        for i in range(10):
            result = system.tick_update(
                external_pressures={'global_pressure': 0.1 * i},
                consciousness_state={'consciousness_level': 0.5 + 0.05 * i}
            )
            tick_results.append(result)
        
        final_state = system.get_system_status()
        
        # Check that system responds to external inputs
        initial_nodes = initial_state['metrics']['total_nodes']
        final_nodes = final_state['metrics']['total_nodes']
        
        # System should be stable or growing
        system_stable = final_nodes >= initial_nodes
        
        self.test_results['system_stability'] = {
            'passed': system_stable,
            'details': f"Nodes: {initial_nodes} -> {final_nodes}"
        }
        
        # Check that all components are coordinated
        all_ticks_completed = all('tick_count' in result for result in tick_results)
        
        self.test_results['component_coordination'] = {
            'passed': all_ticks_completed,
            'details': f"Completed ticks: {len(tick_results)}/10"
        }
        
        print(f"    ✓ System stability: {'PASS' if self.test_results['system_stability']['passed'] else 'FAIL'}")
        print(f"    ✓ Component coordination: {'PASS' if self.test_results['component_coordination']['passed'] else 'FAIL'}")
    
    def validate_performance_characteristics(self):
        """Validate performance and scalability"""
        print("\n⚡ Validating Performance Characteristics...")
        
        # Test small system performance
        small_system = IntegratedMycelialSystem(max_nodes=20)
        
        for i in range(10):
            small_system.add_node(f"perf_small_{i}")
        
        start_time = time.time()
        for _ in range(20):
            small_system.tick_update()
        small_time = time.time() - start_time
        
        # Test larger system performance
        large_system = IntegratedMycelialSystem(max_nodes=100)
        
        for i in range(50):
            large_system.add_node(f"perf_large_{i}")
        
        start_time = time.time()
        for _ in range(20):
            large_system.tick_update()
        large_time = time.time() - start_time
        
        # Performance should scale reasonably (not exponentially)
        scalability_factor = large_time / max(small_time, 0.001)  # Avoid division by zero
        reasonable_scaling = scalability_factor < 10  # Less than 10x slowdown for 5x nodes
        
        self.test_results['performance_scaling'] = {
            'passed': reasonable_scaling,
            'details': f"Small: {small_time:.3f}s, Large: {large_time:.3f}s, Factor: {scalability_factor:.2f}x"
        }
        
        # Test memory efficiency
        import sys
        large_system_size = sys.getsizeof(large_system)
        memory_reasonable = large_system_size < 50 * 1024 * 1024  # Less than 50MB
        
        self.test_results['memory_efficiency'] = {
            'passed': memory_reasonable,
            'details': f"System size: {large_system_size / 1024 / 1024:.2f} MB"
        }
        
        print(f"    ✓ Performance scaling: {'PASS' if self.test_results['performance_scaling']['passed'] else 'FAIL'}")
        print(f"    ✓ Memory efficiency: {'PASS' if self.test_results['memory_efficiency']['passed'] else 'FAIL'}")
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        print("\n" + "=" * 60)
        print("📊 VALIDATION REPORT")
        print("=" * 60)
        
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        total_tests = len(self.test_results)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed ({pass_rate:.1%})")
        
        # Group results by category
        categories = {
            'Biological Principles': ['dendritic_growth', 'mitochondrial_sharing', 'living_substrate'],
            'Mathematical Formulas': ['demand_formula', 'allocation_formula', 'energy_formula'],
            'Energy Systems': ['energy_conservation', 'demand_based_allocation', 'budget_conservation'],
            'Growth & Autophagy': ['energy_threshold', 'similarity_threshold', 'autophagy_trigger'],
            'Metabolite Systems': ['metabolite_production', 'metabolite_absorption', 'metabolite_decay'],
            'Cluster Dynamics': ['cluster_formation', 'cluster_metrics'],
            'System Integration': ['system_stability', 'component_coordination'],
            'Performance': ['performance_scaling', 'memory_efficiency']
        }
        
        category_results = {}
        for category, test_names in categories.items():
            category_tests = [self.test_results.get(name, {'passed': False}) for name in test_names if name in self.test_results]
            if category_tests:
                category_passed = sum(1 for test in category_tests if test['passed'])
                category_total = len(category_tests)
                category_rate = category_passed / category_total
                category_results[category] = {
                    'passed': category_passed,
                    'total': category_total,
                    'rate': category_rate
                }
                
                status = "✅ PASS" if category_rate >= 0.8 else "⚠️  PARTIAL" if category_rate >= 0.5 else "❌ FAIL"
                print(f"\n{category}: {category_passed}/{category_total} ({category_rate:.1%}) {status}")
                
                for test_name in test_names:
                    if test_name in self.test_results:
                        test_result = self.test_results[test_name]
                        status_icon = "✓" if test_result['passed'] else "✗"
                        print(f"  {status_icon} {test_name}: {test_result['details']}")
        
        # Generate recommendations
        recommendations = []
        
        if pass_rate < 0.8:
            recommendations.append("System validation is below 80%. Review failed tests and improve implementation.")
        
        if category_results.get('Mathematical Formulas', {}).get('rate', 0) < 1.0:
            recommendations.append("Mathematical formula implementation needs review.")
        
        if category_results.get('Energy Systems', {}).get('rate', 0) < 0.8:
            recommendations.append("Energy conservation and allocation systems need improvement.")
        
        if category_results.get('Performance', {}).get('rate', 0) < 0.8:
            recommendations.append("Performance optimization needed for scalability.")
        
        if not recommendations:
            recommendations.append("All systems operating within acceptable parameters!")
        
        print(f"\n🎯 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        overall_status = "EXCELLENT" if pass_rate >= 0.95 else "GOOD" if pass_rate >= 0.8 else "NEEDS_IMPROVEMENT" if pass_rate >= 0.6 else "CRITICAL"
        
        print(f"\n🏆 Overall System Status: {overall_status}")
        print("=" * 60)
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'pass_rate': pass_rate,
                'overall_status': overall_status
            },
            'categories': category_results,
            'detailed_results': self.test_results,
            'recommendations': recommendations,
            'validation_errors': self.validation_errors
        }

# Main execution
if __name__ == "__main__":
    validator = ValidationSuite()
    report = validator.run_full_validation()
    
    # Save report
    import json
    with open("mycelial_validation_report.json", "w") as f:
        # Convert numpy types to native Python for JSON serialization
        def convert_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        json.dump(convert_types(report), f, indent=2)
    
    print(f"\n📄 Detailed report saved to: mycelial_validation_report.json")
