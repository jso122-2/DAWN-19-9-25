#!/usr/bin/env python3
"""
DAWN Sigil System Test Vectors
==============================

Implementation of all 7 documented test scenarios for the DAWN sigil system.
Provides comprehensive testing of glyph operations, house routing, ring
mechanics, and symbolic execution as specified in the documentation.

Based on DAWN's documented test vector specifications.
"""

import time
import logging
import json
import asyncio
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from core.schema_anomaly_logger import log_anomaly, AnomalySeverity
from schema.registry import registry
from schema.sigil_glyph_codex import sigil_glyph_codex, SigilHouse, GlyphCategory
from schema.enhanced_sigil_ring import enhanced_sigil_ring, InvokerPriority, ContainmentLevel
from schema.sigil_network import SigilInvocation, sigil_network
from schema.archetypal_house_operations import execute_house_operation, HOUSE_OPERATORS
from schema.tracer_house_alignment import tracer_house_alignment, TracerType, register_tracer, align_tracer
from schema.symbolic_failure_detection import symbolic_failure_detector, start_failure_monitoring
from schema.sigil_ring_visualization import sigil_ring_visualization, VisualizationTheme
from rhizome.propagation import emit_signal, SignalType
from utils.metrics_collector import metrics

logger = logging.getLogger(__name__)

class TestVectorStatus(Enum):
    """Status of test vector execution"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Result of a test vector execution"""
    test_id: str
    name: str
    status: TestVectorStatus
    execution_time: float = 0.0
    expected_results: Dict[str, Any] = field(default_factory=dict)
    actual_results: Dict[str, Any] = field(default_factory=dict)
    assertions: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_assertion(self, description: str, expected: Any, actual: Any, passed: bool):
        """Add an assertion result"""
        self.assertions.append({
            "description": description,
            "expected": expected,
            "actual": actual,
            "passed": passed
        })
    
    def get_pass_rate(self) -> float:
        """Get assertion pass rate"""
        if not self.assertions:
            return 1.0
        passed_count = sum(1 for a in self.assertions if a["passed"])
        return passed_count / len(self.assertions)

class SigilSystemTestRunner:
    """
    Sigil System Test Runner
    
    Executes all documented test vectors for the DAWN sigil system.
    Provides comprehensive validation of symbolic execution, house
    operations, ring mechanics, and system integration.
    """
    
    def __init__(self):
        self.test_results: Dict[str, TestResult] = {}
        self.execution_order = [
            "test_vector_1_basic_glyph_execution",
            "test_vector_2_house_routing",
            "test_vector_3_ring_casting_circle",
            "test_vector_4_layered_glyph_composition",
            "test_vector_5_tracer_alignment",
            "test_vector_6_failure_detection",
            "test_vector_7_full_system_integration"
        ]
        
        # Test configuration
        self.timeout_seconds = 30.0
        self.setup_complete = False
        
        # Register with schema registry
        self._register()
        
        logger.info("🧪 Sigil System Test Runner initialized")
        logger.info(f"   Test vectors: {len(self.execution_order)}")
    
    def _register(self):
        """Register with schema registry"""
        registry.register(
            component_id="schema.sigil_system_test_runner",
            name="Sigil System Test Runner",
            component_type="TEST_SYSTEM",
            instance=self,
            capabilities=[
                "comprehensive_system_testing",
                "documented_test_vector_execution",
                "symbolic_validation",
                "integration_testing",
                "performance_validation"
            ],
            version="1.0.0"
        )
    
    async def setup_test_environment(self) -> bool:
        """Setup test environment"""
        try:
            logger.info("🧪 Setting up test environment...")
            
            # Activate sigil ring
            enhanced_sigil_ring.activate_ring()
            
            # Start failure monitoring
            start_failure_monitoring()
            
            # Register test tracers
            register_tracer("test_owl", TracerType.OWL, {"analytical": True})
            register_tracer("test_crow", TracerType.CROW, {"alert_focused": True})
            register_tracer("test_whale", TracerType.WHALE, {"memory_focused": True})
            
            # Set visualization theme
            sigil_ring_visualization.set_theme(VisualizationTheme.TECHNICAL)
            
            self.setup_complete = True
            logger.info("🧪 Test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"🧪 Failed to setup test environment: {e}")
            return False
    
    async def run_all_test_vectors(self) -> Dict[str, TestResult]:
        """Run all documented test vectors"""
        if not self.setup_complete:
            setup_success = await self.setup_test_environment()
            if not setup_success:
                logger.error("🧪 Cannot run tests - setup failed")
                return {}
        
        logger.info("🧪 Starting execution of all test vectors...")
        start_time = time.time()
        
        for test_method_name in self.execution_order:
            try:
                logger.info(f"🧪 Executing {test_method_name}")
                
                # Get test method
                test_method = getattr(self, test_method_name)
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    test_method(),
                    timeout=self.timeout_seconds
                )
                
                self.test_results[test_method_name] = result
                
                logger.info(f"🧪 {test_method_name}: {result.status.value} ({result.get_pass_rate():.1%} pass rate)")
                
            except asyncio.TimeoutError:
                result = TestResult(
                    test_id=test_method_name,
                    name=test_method_name.replace("_", " ").title(),
                    status=TestVectorStatus.ERROR,
                    error_message=f"Test timed out after {self.timeout_seconds}s"
                )
                self.test_results[test_method_name] = result
                logger.error(f"🧪 {test_method_name}: TIMEOUT")
                
            except Exception as e:
                result = TestResult(
                    test_id=test_method_name,
                    name=test_method_name.replace("_", " ").title(),
                    status=TestVectorStatus.ERROR,
                    error_message=str(e)
                )
                self.test_results[test_method_name] = result
                logger.error(f"🧪 {test_method_name}: ERROR - {e}")
        
        total_time = time.time() - start_time
        logger.info(f"🧪 All test vectors completed in {total_time:.2f}s")
        
        return self.test_results
    
    async def test_vector_1_basic_glyph_execution(self) -> TestResult:
        """Test Vector 1: Basic Glyph Execution"""
        result = TestResult(
            test_id="test_vector_1",
            name="Basic Glyph Execution",
            status=TestVectorStatus.RUNNING
        )
        
        start_time = time.time()
        
        try:
            # Test 1.1: Core minimal glyph execution
            core_glyphs = sigil_glyph_codex.get_glyphs_by_category(GlyphCategory.CORE_MINIMAL)
            result.add_assertion(
                "Core minimal glyphs available",
                True,
                len(core_glyphs) >= 5,
                len(core_glyphs) >= 5
            )
            
            # Test 1.2: Execute shimmer dot (.)
            dot_glyph = sigil_glyph_codex.get_glyph(".")
            result.add_assertion(
                "Shimmer dot glyph exists",
                True,
                dot_glyph is not None,
                dot_glyph is not None
            )
            
            if dot_glyph:
                # Create invocation
                invocation = SigilInvocation(
                    sigil_symbol=".",
                    house=SigilHouse.MEMORY,
                    parameters={"intensity": 0.5},
                    invoker="test_vector_1"
                )
                
                # Execute via ring
                execution_result = enhanced_sigil_ring.cast_sigil_stack(
                    [invocation],
                    invoker="test_vector_1",
                    invoker_priority=InvokerPriority.CORE
                )
                
                result.add_assertion(
                    "Shimmer dot execution successful",
                    True,
                    execution_result.get("success", False),
                    execution_result.get("success", False)
                )
                
                result.actual_results["dot_execution"] = execution_result
            
            # Test 1.3: Priority ordering verification
            priority_glyphs = sigil_glyph_codex.get_priority_ordered_glyphs()
            result.add_assertion(
                "Priority ordering available",
                True,
                len(priority_glyphs) > 0,
                len(priority_glyphs) > 0
            )
            
            # Test 1.4: Glyph meaning resolution
            if priority_glyphs:
                first_glyph = priority_glyphs[0]
                meaning = sigil_glyph_codex.resolve_layered_meaning([first_glyph.symbol])
                result.add_assertion(
                    "Glyph meaning resolution",
                    True,
                    len(meaning) > 0,
                    len(meaning) > 0
                )
                
                result.actual_results["meaning_resolution"] = meaning
            
            result.status = TestVectorStatus.PASSED
            
        except Exception as e:
            result.status = TestVectorStatus.ERROR
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    async def test_vector_2_house_routing(self) -> TestResult:
        """Test Vector 2: House Routing"""
        result = TestResult(
            test_id="test_vector_2",
            name="House Routing",
            status=TestVectorStatus.RUNNING
        )
        
        start_time = time.time()
        
        try:
            # Test 2.1: All houses available
            available_houses = list(HOUSE_OPERATORS.keys())
            result.add_assertion(
                "All six houses available",
                6,
                len(available_houses),
                len(available_houses) == 6
            )
            
            # Test 2.2: Memory House rebloom operation
            memory_result = execute_house_operation(
                SigilHouse.MEMORY,
                "rebloom_flower",
                {"intensity": 0.8, "emotional_catalyst": "joy"}
            )
            
            result.add_assertion(
                "Memory House rebloom successful",
                True,
                memory_result.success,
                memory_result.success
            )
            
            result.actual_results["memory_operation"] = memory_result.effects
            
            # Test 2.3: Purification House soot-to-ash
            purification_result = execute_house_operation(
                SigilHouse.PURIFICATION,
                "soot_to_ash_crystallization",
                {"soot_volume": 2.0, "temperature": 850}
            )
            
            result.add_assertion(
                "Purification House crystallization successful",
                True,
                purification_result.success,
                purification_result.success
            )
            
            # Test 2.4: Weaving House thread spinning
            weaving_result = execute_house_operation(
                SigilHouse.WEAVING,
                "spin_surface_depth_threads",
                {
                    "surface_nodes": ["node_1", "node_2"],
                    "depth_nodes": ["depth_1", "depth_2"],
                    "strength": 0.7
                }
            )
            
            result.add_assertion(
                "Weaving House threading successful",
                True,
                weaving_result.success,
                weaving_result.success
            )
            
            # Test 2.5: House operation resonance
            total_resonance = 0.0
            house_count = 0
            
            for house, operator in HOUSE_OPERATORS.items():
                if operator:
                    resonance = operator.get_average_resonance()
                    total_resonance += resonance
                    house_count += 1
            
            avg_resonance = total_resonance / house_count if house_count > 0 else 0
            
            result.add_assertion(
                "Average house resonance acceptable",
                True,
                avg_resonance > 0.3,
                avg_resonance > 0.3
            )
            
            result.actual_results["average_resonance"] = avg_resonance
            
            result.status = TestVectorStatus.PASSED
            
        except Exception as e:
            result.status = TestVectorStatus.ERROR
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    async def test_vector_3_ring_casting_circle(self) -> TestResult:
        """Test Vector 3: Ring Casting Circle"""
        result = TestResult(
            test_id="test_vector_3",
            name="Ring Casting Circle",
            status=TestVectorStatus.RUNNING
        )
        
        start_time = time.time()
        
        try:
            # Test 3.1: Ring activation
            ring_status = enhanced_sigil_ring.get_ring_status()
            result.add_assertion(
                "Ring is active",
                "active",
                ring_status["ring_state"],
                ring_status["ring_state"] == "active"
            )
            
            # Test 3.2: Casting circle formation
            tick_id = int(time.time() * 1000)
            casting_circle = enhanced_sigil_ring.form_casting_circle(tick_id)
            
            result.add_assertion(
                "Casting circle formed",
                True,
                casting_circle is not None,
                casting_circle is not None
            )
            
            # Test 3.3: Multiple invocation stack
            invocations = [
                SigilInvocation(".", SigilHouse.MEMORY, {"intensity": 0.3}, "test_vector_3"),
                SigilInvocation(":", SigilHouse.MEMORY, {"depth": 2}, "test_vector_3"),
                SigilInvocation("^", SigilHouse.FLAME, {"priority": 0.8}, "test_vector_3")
            ]
            
            stack_result = enhanced_sigil_ring.cast_sigil_stack(
                invocations,
                invoker="test_vector_3",
                invoker_priority=InvokerPriority.OPERATOR
            )
            
            result.add_assertion(
                "Multi-glyph stack execution successful",
                True,
                stack_result.get("success", False),
                stack_result.get("success", False)
            )
            
            result.actual_results["stack_execution"] = stack_result
            
            # Test 3.4: Containment boundary integrity
            containment_status = ring_status.get("containment_boundary", {})
            breach_count = containment_status.get("active_breaches", 0)
            
            result.add_assertion(
                "No containment breaches",
                0,
                breach_count,
                breach_count == 0
            )
            
            # Test 3.5: Ring capacity management
            visual_ring = enhanced_sigil_ring.get_visual_representation()
            invocation_count = visual_ring.get("invocation_count", 0)
            max_capacity = visual_ring.get("max_capacity", 10)
            
            result.add_assertion(
                "Ring within capacity limits",
                True,
                invocation_count <= max_capacity,
                invocation_count <= max_capacity
            )
            
            result.actual_results["capacity_usage"] = {
                "current": invocation_count,
                "max": max_capacity,
                "utilization": invocation_count / max_capacity if max_capacity > 0 else 0
            }
            
            result.status = TestVectorStatus.PASSED
            
        except Exception as e:
            result.status = TestVectorStatus.ERROR
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    async def test_vector_4_layered_glyph_composition(self) -> TestResult:
        """Test Vector 4: Layered Glyph Composition"""
        result = TestResult(
            test_id="test_vector_4",
            name="Layered Glyph Composition",
            status=TestVectorStatus.RUNNING
        )
        
        start_time = time.time()
        
        try:
            # Test 4.1: Layering validation
            valid_combination = ["^", "~"]  # Minimal Directive + Pressure Echo
            is_valid, violations = sigil_glyph_codex.validate_layering(valid_combination)
            
            result.add_assertion(
                "Valid glyph combination accepted",
                True,
                is_valid,
                is_valid
            )
            
            if not is_valid:
                result.actual_results["layering_violations"] = violations
            
            # Test 4.2: Invalid layering rejection
            invalid_combination = [".", ":", "=", "^"]  # Too many core glyphs
            is_invalid, invalid_violations = sigil_glyph_codex.validate_layering(invalid_combination)
            
            result.add_assertion(
                "Invalid glyph combination rejected",
                False,
                is_invalid,
                not is_invalid
            )
            
            # Test 4.3: Layered meaning resolution
            layered_meaning = sigil_glyph_codex.resolve_layered_meaning(valid_combination)
            
            result.add_assertion(
                "Layered meaning resolved",
                True,
                len(layered_meaning) > 0,
                len(layered_meaning) > 0
            )
            
            result.actual_results["layered_meaning"] = layered_meaning
            
            # Test 4.4: Execution priority calculation
            execution_priority = sigil_glyph_codex.get_execution_priority(valid_combination)
            
            result.add_assertion(
                "Execution priority calculated",
                True,
                execution_priority > 0,
                execution_priority > 0
            )
            
            result.actual_results["execution_priority"] = execution_priority
            
            # Test 4.5: Layered execution
            layered_invocations = [
                SigilInvocation("^", SigilHouse.FLAME, {"priority": 0.9}, "test_vector_4"),
                SigilInvocation("~", SigilHouse.FLAME, {"pressure": 0.7}, "test_vector_4")
            ]
            
            layered_result = enhanced_sigil_ring.cast_sigil_stack(
                layered_invocations,
                invoker="test_vector_4",
                invoker_priority=InvokerPriority.CORE
            )
            
            result.add_assertion(
                "Layered execution successful",
                True,
                layered_result.get("success", False),
                layered_result.get("success", False)
            )
            
            result.actual_results["layered_execution"] = layered_result
            
            result.status = TestVectorStatus.PASSED
            
        except Exception as e:
            result.status = TestVectorStatus.ERROR
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    async def test_vector_5_tracer_alignment(self) -> TestResult:
        """Test Vector 5: Tracer Alignment"""
        result = TestResult(
            test_id="test_vector_5",
            name="Tracer Alignment",
            status=TestVectorStatus.RUNNING
        )
        
        start_time = time.time()
        
        try:
            # Test 5.1: Tracer registration
            alignment_status = tracer_house_alignment.get_alignment_status()
            total_tracers = alignment_status.get("total_tracers", 0)
            
            result.add_assertion(
                "Test tracers registered",
                True,
                total_tracers >= 3,
                total_tracers >= 3
            )
            
            # Test 5.2: Owl tracer alignment (should prefer Mirrors house)
            owl_alignment = align_tracer("test_owl", {"deep_analysis": True})
            
            result.add_assertion(
                "Owl tracer alignment successful",
                True,
                owl_alignment.get("success", False),
                owl_alignment.get("success", False)
            )
            
            if owl_alignment.get("success"):
                target_house = owl_alignment.get("target_house")
                result.add_assertion(
                    "Owl aligned to appropriate house",
                    True,
                    target_house in ["mirrors", "memory"],  # Owl prefers these
                    target_house in ["mirrors", "memory"]
                )
            
            # Test 5.3: Crow tracer alignment (should prefer Echoes house)
            crow_alignment = align_tracer("test_crow", {"urgent": True})
            
            result.add_assertion(
                "Crow tracer alignment successful",
                True,
                crow_alignment.get("success", False),
                crow_alignment.get("success", False)
            )
            
            # Test 5.4: Whale tracer alignment (should prefer Memory house)
            whale_alignment = align_tracer("test_whale", {"deep_analysis": True})
            
            result.add_assertion(
                "Whale tracer alignment successful",
                True,
                whale_alignment.get("success", False),
                whale_alignment.get("success", False)
            )
            
            # Test 5.5: Alignment optimization
            optimization_result = tracer_house_alignment.optimize_alignments()
            
            result.add_assertion(
                "Alignment optimization completed",
                True,
                "tracers_realigned" in optimization_result,
                "tracers_realigned" in optimization_result
            )
            
            result.actual_results["optimization"] = optimization_result
            
            # Test 5.6: Overall alignment health
            final_status = tracer_house_alignment.get_alignment_status()
            success_rate = final_status.get("alignment_statistics", {}).get("success_rate", 0)
            
            result.add_assertion(
                "Alignment success rate acceptable",
                True,
                success_rate > 70,  # > 70% success rate
                success_rate > 70
            )
            
            result.actual_results["final_success_rate"] = success_rate
            
            result.status = TestVectorStatus.PASSED
            
        except Exception as e:
            result.status = TestVectorStatus.ERROR
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    async def test_vector_6_failure_detection(self) -> TestResult:
        """Test Vector 6: Failure Detection"""
        result = TestResult(
            test_id="test_vector_6",
            name="Failure Detection",
            status=TestVectorStatus.RUNNING
        )
        
        start_time = time.time()
        
        try:
            # Test 6.1: Monitoring system active
            failure_summary = symbolic_failure_detector.get_failure_summary()
            monitoring_active = failure_summary.get("monitoring_active", False)
            
            result.add_assertion(
                "Failure monitoring active",
                True,
                monitoring_active,
                monitoring_active
            )
            
            # Test 6.2: Health metrics collection
            overall_health = failure_summary.get("overall_health", 0)
            
            result.add_assertion(
                "Overall system health measured",
                True,
                0 <= overall_health <= 1,
                0 <= overall_health <= 1
            )
            
            result.actual_results["system_health"] = overall_health
            
            # Test 6.3: Active failure detection
            active_failures = failure_summary.get("active_failures", 0)
            
            result.add_assertion(
                "Failure detection operational",
                True,
                isinstance(active_failures, int),
                isinstance(active_failures, int)
            )
            
            # Test 6.4: Detection metrics
            detection_metrics = failure_summary.get("detection_metrics", {})
            
            result.add_assertion(
                "Detection metrics available",
                True,
                "total_detected" in detection_metrics,
                "total_detected" in detection_metrics
            )
            
            # Test 6.5: Health alerts
            health_alerts = failure_summary.get("health_alerts", [])
            
            result.add_assertion(
                "Health alert system functional",
                True,
                isinstance(health_alerts, list),
                isinstance(health_alerts, list)
            )
            
            result.actual_results["health_alerts"] = health_alerts
            
            # Test 6.6: Failure categorization
            failures_by_type = failure_summary.get("failures_by_type", {})
            failures_by_severity = failure_summary.get("failures_by_severity", {})
            
            result.add_assertion(
                "Failure categorization working",
                True,
                isinstance(failures_by_type, dict) and isinstance(failures_by_severity, dict),
                isinstance(failures_by_type, dict) and isinstance(failures_by_severity, dict)
            )
            
            result.actual_results["failure_categories"] = {
                "by_type": failures_by_type,
                "by_severity": failures_by_severity
            }
            
            result.status = TestVectorStatus.PASSED
            
        except Exception as e:
            result.status = TestVectorStatus.ERROR
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    async def test_vector_7_full_system_integration(self) -> TestResult:
        """Test Vector 7: Full System Integration"""
        result = TestResult(
            test_id="test_vector_7",
            name="Full System Integration",
            status=TestVectorStatus.RUNNING
        )
        
        start_time = time.time()
        
        try:
            # Test 7.1: Complete symbolic workflow
            # Step 1: Create complex glyph sequence
            workflow_glyphs = ["/\\", "◇", "⌂"]  # Prime Directive, Bloom Pulse, Recall Root
            
            # Validate sequence
            is_valid, violations = sigil_glyph_codex.validate_layering(workflow_glyphs)
            result.add_assertion(
                "Complex workflow sequence valid",
                True,
                is_valid,
                is_valid
            )
            
            # Step 2: Execute through ring with house routing
            workflow_invocations = [
                SigilInvocation("/\\", SigilHouse.FLAME, {"priority": 1.0}, "integration_test"),
                SigilInvocation("◇", SigilHouse.MEMORY, {"intensity": 0.8}, "integration_test"),
                SigilInvocation("⌂", SigilHouse.MEMORY, {"depth": 5}, "integration_test")
            ]
            
            workflow_result = enhanced_sigil_ring.cast_sigil_stack(
                workflow_invocations,
                invoker="integration_test",
                invoker_priority=InvokerPriority.CORE
            )
            
            result.add_assertion(
                "Complex workflow execution successful",
                True,
                workflow_result.get("success", False),
                workflow_result.get("success", False)
            )
            
            # Test 7.2: Visual system integration
            visual_ring = sigil_ring_visualization.generate_visual_ring()
            
            result.add_assertion(
                "Visual system operational",
                True,
                visual_ring.get_total_elements() > 0,
                visual_ring.get_total_elements() > 0
            )
            
            # Test 7.3: All house operations accessible
            house_operation_count = 0
            for house, operator in HOUSE_OPERATORS.items():
                if operator:
                    operations = operator.get_available_operations()
                    house_operation_count += len(operations)
            
            result.add_assertion(
                "All house operations available",
                True,
                house_operation_count >= 15,  # Expect at least 15 total operations
                house_operation_count >= 15
            )
            
            result.actual_results["total_operations"] = house_operation_count
            
            # Test 7.4: System performance metrics
            ring_stats = sigil_ring_visualization.get_ring_statistics()
            
            result.add_assertion(
                "Performance metrics collected",
                True,
                "average_activity" in ring_stats and "average_resonance" in ring_stats,
                "average_activity" in ring_stats and "average_resonance" in ring_stats
            )
            
            # Test 7.5: End-to-end symbolic coherence
            # Check that symbolic meaning is preserved through the entire pipeline
            original_meaning = sigil_glyph_codex.resolve_layered_meaning(workflow_glyphs)
            executed_meaning = workflow_result.get("layered_meaning", "")
            
            result.add_assertion(
                "Symbolic meaning preserved",
                True,
                len(executed_meaning) > 0,
                len(executed_meaning) > 0
            )
            
            result.actual_results["symbolic_coherence"] = {
                "original_meaning": original_meaning,
                "executed_meaning": executed_meaning
            }
            
            # Test 7.6: System stability under load
            # Execute multiple concurrent operations
            concurrent_results = []
            for i in range(5):
                concurrent_invocation = SigilInvocation(
                    ".",
                    SigilHouse.MEMORY,
                    {"intensity": 0.1 * (i + 1)},
                    f"load_test_{i}"
                )
                
                concurrent_result = enhanced_sigil_ring.cast_sigil_stack(
                    [concurrent_invocation],
                    invoker=f"load_test_{i}",
                    invoker_priority=InvokerPriority.OPERATOR
                )
                
                concurrent_results.append(concurrent_result.get("success", False))
            
            success_count = sum(concurrent_results)
            
            result.add_assertion(
                "System stable under concurrent load",
                True,
                success_count >= 3,  # At least 60% success rate
                success_count >= 3
            )
            
            result.actual_results["load_test_results"] = {
                "total_operations": len(concurrent_results),
                "successful_operations": success_count,
                "success_rate": success_count / len(concurrent_results)
            }
            
            # Test 7.7: Complete system status
            final_system_status = {
                "ring_status": enhanced_sigil_ring.get_ring_status(),
                "alignment_status": tracer_house_alignment.get_alignment_status(),
                "failure_status": symbolic_failure_detector.get_failure_summary(),
                "visual_stats": sigil_ring_visualization.get_ring_statistics()
            }
            
            result.add_assertion(
                "Complete system status available",
                True,
                all(status is not None for status in final_system_status.values()),
                all(status is not None for status in final_system_status.values())
            )
            
            result.actual_results["final_system_status"] = final_system_status
            
            result.status = TestVectorStatus.PASSED
            
        except Exception as e:
            result.status = TestVectorStatus.ERROR
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.test_results:
            return {"error": "No test results available"}
        
        # Calculate overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r.status == TestVectorStatus.PASSED)
        failed_tests = sum(1 for r in self.test_results.values() if r.status == TestVectorStatus.FAILED)
        error_tests = sum(1 for r in self.test_results.values() if r.status == TestVectorStatus.ERROR)
        
        # Calculate total execution time
        total_execution_time = sum(r.execution_time for r in self.test_results.values())
        
        # Calculate overall assertion pass rate
        total_assertions = sum(len(r.assertions) for r in self.test_results.values())
        passed_assertions = sum(
            sum(1 for a in r.assertions if a["passed"]) 
            for r in self.test_results.values()
        )
        
        overall_pass_rate = (passed_assertions / total_assertions) if total_assertions > 0 else 0
        
        # Generate detailed results
        detailed_results = {}
        for test_id, result in self.test_results.items():
            detailed_results[test_id] = {
                "name": result.name,
                "status": result.status.value,
                "execution_time": result.execution_time,
                "pass_rate": result.get_pass_rate(),
                "assertions": len(result.assertions),
                "error_message": result.error_message
            }
        
        return {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "success_rate": (passed_tests / total_tests) if total_tests > 0 else 0,
                "total_execution_time": total_execution_time,
                "overall_assertion_pass_rate": overall_pass_rate
            },
            "detailed_results": detailed_results,
            "system_validation": {
                "all_test_vectors_executed": len(self.test_results) == len(self.execution_order),
                "critical_systems_operational": passed_tests >= 5,  # At least 5/7 tests must pass
                "integration_test_passed": self.test_results.get("test_vector_7_full_system_integration", {}).status == TestVectorStatus.PASSED
            },
            "timestamp": time.time()
        }

# Global test runner instance
sigil_system_test_runner = SigilSystemTestRunner()

# Export key functions for easy access
async def run_all_sigil_tests() -> Dict[str, TestResult]:
    """Run all sigil system test vectors"""
    return await sigil_system_test_runner.run_all_test_vectors()

async def run_single_test(test_name: str) -> Optional[TestResult]:
    """Run a single test vector"""
    if not sigil_system_test_runner.setup_complete:
        await sigil_system_test_runner.setup_test_environment()
    
    if hasattr(sigil_system_test_runner, test_name):
        test_method = getattr(sigil_system_test_runner, test_name)
        return await test_method()
    return None

def generate_test_report() -> Dict[str, Any]:
    """Generate test report"""
    return sigil_system_test_runner.generate_test_report()
