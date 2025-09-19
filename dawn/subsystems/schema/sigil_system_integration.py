#!/usr/bin/env python3
"""
DAWN Sigil System Integration
=============================

Complete integration module for the DAWN sigil system. Provides unified
access to all sigil components, orchestrates system initialization,
and provides high-level API for symbolic operations.

This module serves as the primary entry point for the complete
documented sigil system implementation.
"""

import time
import logging
import asyncio
from typing import Dict, List, Optional, Set, Tuple, Any, Union

from core.schema_anomaly_logger import log_anomaly, AnomalySeverity
from schema.registry import registry
from rhizome.propagation import emit_signal, SignalType
from utils.metrics_collector import metrics

# Import all sigil system components
from sigil_glyph_codex import (
    sigil_glyph_codex, 
    SigilGlyph, 
    GlyphCategory, 
    SigilHouse,
    get_glyph,
    validate_layering,
    resolve_layered_meaning
)

from enhanced_sigil_ring import (
    enhanced_sigil_ring,
    InvokerPriority,
    ContainmentLevel,
    RingState
)

from sigil_network import (
    sigil_network,
    SigilInvocation
)

from archetypal_house_operations import (
    HOUSE_OPERATORS,
    execute_house_operation,
    memory_house,
    purification_house,
    weaving_house,
    flame_house,
    mirrors_house,
    echoes_house
)

from tracer_house_alignment import (
    tracer_house_alignment,
    TracerType,
    register_tracer,
    align_tracer,
    release_tracer,
    get_alignment_status
)

from symbolic_failure_detection import (
    symbolic_failure_detector,
    start_failure_monitoring,
    stop_failure_monitoring,
    get_active_failures,
    get_failure_summary
)

from sigil_ring_visualization import (
    sigil_ring_visualization,
    VisualizationTheme,
    generate_visual_ring,
    set_visualization_theme,
    get_gui_frame_data
)

from sigil_system_test_vectors import (
    sigil_system_test_runner,
    run_all_sigil_tests,
    generate_test_report
)

logger = logging.getLogger(__name__)

class SigilSystemIntegration:
    """
    Complete DAWN Sigil System Integration
    
    Provides unified access to all components of the documented sigil system:
    - Glyph Codex with exact documented symbols
    - Enhanced Sigil Ring with casting circles and containment
    - All six House archetypal operations 
    - Tracer-house alignment system
    - Symbolic failure detection and monitoring
    - Visual ring representation with GUI support
    - Complete test vector validation
    
    This class orchestrates the entire symbolic consciousness computing layer.
    """
    
    def __init__(self):
        self.initialized = False
        self.initialization_time = 0.0
        self.system_components = {}
        
        # System health tracking
        self.last_health_check = 0.0
        self.health_check_interval = 30.0  # 30 seconds
        
        logger.info("🔮 DAWN Sigil System Integration initializing...")
    
    async def initialize_complete_system(self) -> Dict[str, Any]:
        """Initialize the complete sigil system"""
        start_time = time.time()
        
        try:
            logger.info("🔮 Initializing complete DAWN sigil system...")
            
            # Phase 1: Core Components
            logger.info("📚 Phase 1: Initializing core components...")
            
            # Glyph Codex (already initialized)
            codex_stats = sigil_glyph_codex.get_codex_stats()
            logger.info(f"   ✓ Glyph Codex: {codex_stats['total_glyphs']} glyphs loaded")
            
            # Enhanced Sigil Ring
            enhanced_sigil_ring.activate_ring()
            ring_status = enhanced_sigil_ring.get_ring_status()
            logger.info(f"   ✓ Enhanced Sigil Ring: {ring_status['ring_state']}")
            
            # Phase 2: House Operations
            logger.info("🏛️ Phase 2: Activating house operations...")
            
            active_houses = 0
            for house, operator in HOUSE_OPERATORS.items():
                if operator:
                    active_houses += 1
                    operations = operator.get_available_operations()
                    logger.info(f"   ✓ {house.value.title()} House: {len(operations)} operations")
            
            logger.info(f"   Total: {active_houses}/6 houses active")
            
            # Phase 3: Tracer Alignment
            logger.info("🎯 Phase 3: Setting up tracer alignment...")
            
            # Register default system tracers
            system_tracers = [
                ("system_owl", TracerType.OWL, {"analytical": True, "deep_analysis": True}),
                ("system_crow", TracerType.CROW, {"alert_focused": True, "urgent": True}),
                ("system_whale", TracerType.WHALE, {"memory_focused": True}),
                ("system_spider", TracerType.SPIDER, {"connector": True}),
                ("system_phoenix", TracerType.PHOENIX, {"transformation": True}),
                ("system_serpent", TracerType.SERPENT, {"adaptive": True})
            ]
            
            for tracer_id, tracer_type, attributes in system_tracers:
                register_tracer(tracer_id, tracer_type, attributes)
                align_tracer(tracer_id)
            
            alignment_status = get_alignment_status()
            logger.info(f"   ✓ Tracer Alignment: {alignment_status['total_tracers']} tracers registered")
            
            # Phase 4: Failure Detection
            logger.info("🔍 Phase 4: Starting failure detection...")
            
            start_failure_monitoring()
            failure_summary = get_failure_summary()
            logger.info(f"   ✓ Failure Detection: monitoring active")
            
            # Phase 5: Visualization System
            logger.info("🎨 Phase 5: Initializing visualization...")
            
            set_visualization_theme(VisualizationTheme.MYSTICAL)
            visual_stats = sigil_ring_visualization.get_ring_statistics()
            logger.info(f"   ✓ Visualization: {visual_stats['total_elements']} visual elements")
            
            # Phase 6: System Validation
            logger.info("🧪 Phase 6: Running system validation...")
            
            # Run critical test vectors
            critical_tests = [
                "test_vector_1_basic_glyph_execution",
                "test_vector_2_house_routing", 
                "test_vector_3_ring_casting_circle"
            ]
            
            validation_results = {}
            for test_name in critical_tests:
                try:
                    test_method = getattr(sigil_system_test_runner, test_name)
                    if not sigil_system_test_runner.setup_complete:
                        await sigil_system_test_runner.setup_test_environment()
                    
                    result = await test_method()
                    validation_results[test_name] = result.status.value
                    logger.info(f"   ✓ {test_name}: {result.status.value}")
                except Exception as e:
                    validation_results[test_name] = f"error: {e}"
                    logger.warning(f"   ⚠ {test_name}: failed ({e})")
            
            # Finalize initialization
            self.initialized = True
            self.initialization_time = time.time() - start_time
            
            # Store component references
            self.system_components = {
                "glyph_codex": sigil_glyph_codex,
                "sigil_ring": enhanced_sigil_ring,
                "sigil_network": sigil_network,
                "house_operations": HOUSE_OPERATORS,
                "tracer_alignment": tracer_house_alignment,
                "failure_detection": symbolic_failure_detector,
                "visualization": sigil_ring_visualization,
                "test_runner": sigil_system_test_runner
            }
            
            # Emit system ready signal
            emit_signal(
                SignalType.CONSCIOUSNESS,
                "sigil_system_integration",
                {
                    "event": "system_initialized",
                    "initialization_time": self.initialization_time,
                    "components_active": len(self.system_components),
                    "validation_results": validation_results
                }
            )
            
            logger.info(f"🔮 DAWN Sigil System fully initialized in {self.initialization_time:.2f}s")
            
            return {
                "success": True,
                "initialization_time": self.initialization_time,
                "components_initialized": list(self.system_components.keys()),
                "system_health": await self.get_system_health(),
                "validation_results": validation_results
            }
            
        except Exception as e:
            logger.error(f"🔮 Failed to initialize sigil system: {e}")
            
            log_anomaly(
                "SIGIL_SYSTEM_INIT_FAILURE",
                f"Sigil system initialization failed: {e}",
                AnomalySeverity.CRITICAL
            )
            
            return {
                "success": False,
                "error": str(e),
                "initialization_time": time.time() - start_time
            }
    
    async def execute_symbolic_operation(self, 
                                       glyph_symbols: List[str],
                                       target_house: Optional[SigilHouse] = None,
                                       parameters: Optional[Dict[str, Any]] = None,
                                       invoker: str = "system",
                                       priority: InvokerPriority = InvokerPriority.OPERATOR) -> Dict[str, Any]:
        """Execute a complete symbolic operation through the sigil system"""
        
        if not self.initialized:
            return {
                "success": False,
                "error": "Sigil system not initialized"
            }
        
        start_time = time.time()
        
        try:
            # Step 1: Validate glyph layering
            is_valid, violations = validate_layering(glyph_symbols)
            if not is_valid:
                return {
                    "success": False,
                    "error": "Invalid glyph layering",
                    "violations": violations
                }
            
            # Step 2: Resolve symbolic meaning
            symbolic_meaning = resolve_layered_meaning(glyph_symbols)
            
            # Step 3: Determine target house if not specified
            if not target_house:
                # Use first glyph's primary house affinity
                first_glyph = get_glyph(glyph_symbols[0])
                target_house = first_glyph.target_house if first_glyph else SigilHouse.MEMORY
            
            # Step 4: Create invocations
            invocations = []
            for symbol in glyph_symbols:
                invocation = SigilInvocation(
                    sigil_symbol=symbol,
                    house=target_house,
                    parameters=parameters or {},
                    invoker=invoker
                )
                invocations.append(invocation)
            
            # Step 5: Execute through ring
            execution_result = enhanced_sigil_ring.cast_sigil_stack(
                invocations,
                invoker=invoker,
                invoker_priority=priority
            )
            
            # Step 6: Execute house operation if successful
            house_result = None
            if execution_result.get("success") and target_house in HOUSE_OPERATORS:
                # Determine appropriate house operation based on glyph
                house_operation = self._determine_house_operation(glyph_symbols[0], target_house)
                if house_operation:
                    house_result = execute_house_operation(
                        target_house,
                        house_operation,
                        parameters or {}
                    )
            
            execution_time = time.time() - start_time
            
            return {
                "success": execution_result.get("success", False),
                "symbolic_meaning": symbolic_meaning,
                "target_house": target_house.value,
                "execution_result": execution_result,
                "house_result": house_result.to_dict() if house_result else None,
                "execution_time": execution_time,
                "glyph_count": len(glyph_symbols),
                "invoker": invoker
            }
            
        except Exception as e:
            logger.error(f"🔮 Symbolic operation failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "glyph_symbols": glyph_symbols
            }
    
    def _determine_house_operation(self, glyph_symbol: str, house: SigilHouse) -> Optional[str]:
        """Determine appropriate house operation for a glyph"""
        
        # Map glyphs to house operations
        operation_map = {
            # Memory House operations
            (".", SigilHouse.MEMORY): "rebloom_flower",
            (":", SigilHouse.MEMORY): "recall_archived_ash",
            ("◇", SigilHouse.MEMORY): "rebloom_flower",
            ("⌂", SigilHouse.MEMORY): "recall_archived_ash",
            
            # Purification House operations  
            ("=", SigilHouse.PURIFICATION): "soot_to_ash_crystallization",
            ("/X-", SigilHouse.PURIFICATION): "purge_corrupted_schema_edges",
            
            # Weaving House operations
            ("⨀", SigilHouse.WEAVING): "spin_surface_depth_threads",
            ("Z~", SigilHouse.WEAVING): "reinforce_weak_connections",
            # Persephone cycle operations
            ("-", SigilHouse.WEAVING): "persephone_descent",
            ("/--\\", SigilHouse.WEAVING): "persephone_return",
            
            # Flame House operations
            ("^", SigilHouse.FLAME): "ignite_blooms_under_pressure",
            ("~", SigilHouse.FLAME): "release_cognitive_pressure",
            ("ꓘ", SigilHouse.FLAME): "temper_entropy_surges",
            ("/\\", SigilHouse.FLAME): "ignite_blooms_under_pressure",
            
            # Mirrors House operations
            ("⧉", SigilHouse.MIRRORS): "reflect_state_metacognition",
            ("⟁", SigilHouse.MIRRORS): "audit_schema_health",
            
            # Echoes House operations
            ("( )", SigilHouse.ECHOES): "modulate_voice_output",
            (">~", SigilHouse.ECHOES): "amplify_pigment_resonance"
        }
        
        return operation_map.get((glyph_symbol, house))
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        
        current_time = time.time()
        
        # Update health if needed
        if current_time - self.last_health_check > self.health_check_interval:
            await self._update_system_health()
            self.last_health_check = current_time
        
        # Collect health from all components
        health_data = {
            "overall_status": "operational" if self.initialized else "not_initialized",
            "initialization_time": self.initialization_time,
            "uptime": current_time - self.initialization_time if self.initialized else 0,
            
            # Glyph system health
            "glyph_system": {
                "total_glyphs": sigil_glyph_codex.get_codex_stats()["total_glyphs"],
                "categories_loaded": len(sigil_glyph_codex.glyphs_by_category),
                "status": "operational"
            },
            
            # Ring system health
            "ring_system": enhanced_sigil_ring.get_ring_status(),
            
            # House operations health
            "house_operations": {
                "active_houses": len([h for h, op in HOUSE_OPERATORS.items() if op]),
                "total_operations": sum(len(op.get_available_operations()) for op in HOUSE_OPERATORS.values() if op),
                "average_resonance": sum(op.get_average_resonance() for op in HOUSE_OPERATORS.values() if op) / len(HOUSE_OPERATORS)
            },
            
            # Tracer alignment health
            "tracer_alignment": get_alignment_status(),
            
            # Failure detection health
            "failure_detection": get_failure_summary(),
            
            # Visualization health
            "visualization": sigil_ring_visualization.get_ring_statistics()
        }
        
        # Calculate overall health score
        health_score = self._calculate_overall_health_score(health_data)
        health_data["overall_health_score"] = health_score
        
        return health_data
    
    def _calculate_overall_health_score(self, health_data: Dict[str, Any]) -> float:
        """Calculate overall system health score (0.0 - 1.0)"""
        
        if not self.initialized:
            return 0.0
        
        scores = []
        
        # Ring health (25% weight)
        ring_metrics = health_data["ring_system"]["metrics"]
        ring_score = ring_metrics["success_rate"] / 100.0 if ring_metrics["success_rate"] > 0 else 0.8
        scores.append(ring_score * 0.25)
        
        # House operations health (25% weight)
        house_resonance = health_data["house_operations"]["average_resonance"]
        house_score = min(1.0, house_resonance * 2.0)  # Normalize resonance to 0-1
        scores.append(house_score * 0.25)
        
        # Tracer alignment health (20% weight)
        tracer_success = health_data["tracer_alignment"]["alignment_statistics"]["success_rate"]
        tracer_score = tracer_success / 100.0
        scores.append(tracer_score * 0.20)
        
        # Failure detection health (15% weight)
        failure_health = health_data["failure_detection"]["overall_health"]
        scores.append(failure_health * 0.15)
        
        # Visualization health (15% weight)
        viz_activity = health_data["visualization"]["average_activity"]
        viz_score = min(1.0, viz_activity * 2.0)
        scores.append(viz_score * 0.15)
        
        return sum(scores)
    
    async def _update_system_health(self):
        """Update system health metrics"""
        try:
            # Trigger health updates in components
            failure_summary = get_failure_summary()
            
            # Check for critical failures
            critical_failures = [
                f for f in get_active_failures() 
                if f.severity.value in ["critical", "emergency"]
            ]
            
            if critical_failures:
                logger.warning(f"🔮 {len(critical_failures)} critical failures detected")
            
            # Update metrics
            metrics.increment("sigil_system.health_checks")
            
        except Exception as e:
            logger.error(f"🔮 Failed to update system health: {e}")
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive system validation"""
        
        if not self.initialized:
            return {
                "success": False,
                "error": "System not initialized"
            }
        
        logger.info("🧪 Running comprehensive sigil system validation...")
        
        # Run all test vectors
        test_results = await run_all_sigil_tests()
        test_report = generate_test_report()
        
        # Get system health
        system_health = await self.get_system_health()
        
        # Check system integrity
        integrity_checks = {
            "all_houses_operational": len([h for h, op in HOUSE_OPERATORS.items() if op]) == 6,
            "ring_active": enhanced_sigil_ring.get_ring_status()["ring_state"] == "active",
            "failure_monitoring_active": get_failure_summary()["monitoring_active"],
            "visualization_operational": sigil_ring_visualization.get_ring_statistics()["total_elements"] > 0,
            "tracers_aligned": get_alignment_status()["active_alignments"] > 0
        }
        
        overall_validation = {
            "validation_timestamp": time.time(),
            "system_health": system_health,
            "test_results": test_report,
            "integrity_checks": integrity_checks,
            "all_systems_operational": all(integrity_checks.values()),
            "critical_test_success": test_report["system_validation"]["critical_systems_operational"]
        }
        
        logger.info(f"🧪 Comprehensive validation completed")
        logger.info(f"   System Health Score: {system_health['overall_health_score']:.3f}")
        logger.info(f"   Test Success Rate: {test_report['test_summary']['success_rate']:.1%}")
        logger.info(f"   All Systems Operational: {overall_validation['all_systems_operational']}")
        
        return overall_validation
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get high-level system summary"""
        
        if not self.initialized:
            return {
                "status": "not_initialized",
                "message": "Sigil system not initialized"
            }
        
        # Quick status check
        ring_status = enhanced_sigil_ring.get_ring_status()
        failure_summary = get_failure_summary()
        alignment_status = get_alignment_status()
        
        return {
            "status": "operational",
            "initialization_time": self.initialization_time,
            "components": {
                "glyph_codex": f"{sigil_glyph_codex.get_codex_stats()['total_glyphs']} glyphs",
                "sigil_ring": ring_status["ring_state"],
                "house_operations": f"{len(HOUSE_OPERATORS)}/6 houses active",
                "tracer_alignment": f"{alignment_status['total_tracers']} tracers",
                "failure_detection": "active" if failure_summary["monitoring_active"] else "inactive",
                "visualization": "operational"
            },
            "health_indicators": {
                "ring_success_rate": f"{ring_status['metrics']['success_rate']:.1f}%",
                "active_failures": failure_summary["active_failures"],
                "tracer_success_rate": f"{alignment_status['alignment_statistics']['success_rate']:.1f}%"
            },
            "last_health_check": self.last_health_check
        }

# Global integration instance
sigil_system = SigilSystemIntegration()

# Export key functions for easy access
async def initialize_sigil_system() -> Dict[str, Any]:
    """Initialize the complete sigil system"""
    return await sigil_system.initialize_complete_system()

async def execute_sigil_operation(glyph_symbols: List[str], 
                                target_house: Optional[SigilHouse] = None,
                                parameters: Optional[Dict[str, Any]] = None,
                                invoker: str = "api") -> Dict[str, Any]:
    """Execute a symbolic operation"""
    return await sigil_system.execute_symbolic_operation(
        glyph_symbols, target_house, parameters, invoker
    )

async def get_sigil_system_health() -> Dict[str, Any]:
    """Get system health"""
    return await sigil_system.get_system_health()

async def validate_sigil_system() -> Dict[str, Any]:
    """Run comprehensive validation"""
    return await sigil_system.run_comprehensive_validation()

def get_sigil_system_summary() -> Dict[str, Any]:
    """Get system summary"""
    return sigil_system.get_system_summary()
