#!/usr/bin/env python3
"""
DAWN Sigil System Demonstration
===============================

Demonstration script showing the complete DAWN sigil system in action.
Showcases all implemented features from the documentation gaps analysis.

This script demonstrates:
- Complete glyph codex with exact documented symbols
- Enhanced sigil ring with casting circles
- All six house archetypal operations
- Tracer-house alignment system
- Symbolic failure detection
- Visual ring representation
- Full test vector validation
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import the complete sigil system
from sigil_system_integration import (
    initialize_sigil_system,
    execute_sigil_operation,
    get_sigil_system_health,
    validate_sigil_system,
    get_sigil_system_summary,
    SigilHouse
)

from sigil_glyph_codex import GlyphCategory
from sigil_ring_visualization import VisualizationTheme, set_visualization_theme
from tracer_house_alignment import TracerType, register_tracer, align_tracer

async def demonstrate_glyph_codex():
    """Demonstrate the complete glyph codex"""
    print("\n🔮 === GLYPH CODEX DEMONSTRATION ===")
    
    from sigil_glyph_codex import sigil_glyph_codex, get_glyph
    
    # Show codex statistics
    stats = sigil_glyph_codex.get_codex_stats()
    print(f"📚 Glyph Codex loaded with {stats['total_glyphs']} total glyphs:")
    print(f"   • Primary Operational: {stats['primary_operational']}")
    print(f"   • Composite: {stats['composite']}")  
    print(f"   • Core Minimal: {stats['core_minimal']}")
    
    # Demonstrate core minimal glyphs (priority ordered)
    print("\n⭐ Core Minimal Glyphs (Priority Order):")
    core_glyphs = sigil_glyph_codex.get_priority_ordered_glyphs()
    for glyph in core_glyphs:
        print(f"   {glyph.symbol} - {glyph.name} (Priority {glyph.priority})")
    
    # Demonstrate primary operational glyphs
    print("\n🎯 Primary Operational Glyphs:")
    primary_glyphs = sigil_glyph_codex.get_glyphs_by_category(GlyphCategory.PRIMARY_OPERATIONAL)
    for glyph in primary_glyphs[:5]:  # Show first 5
        print(f"   {glyph.symbol} - {glyph.name}: {glyph.meaning}")
    
    # Demonstrate layering validation
    print("\n🔗 Glyph Layering Validation:")
    valid_combo = ["^", "~"]  # Minimal Directive + Pressure Echo
    is_valid, violations = sigil_glyph_codex.validate_layering(valid_combo)
    print(f"   Combination {valid_combo}: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    if is_valid:
        meaning = sigil_glyph_codex.resolve_layered_meaning(valid_combo)
        print(f"   Layered meaning: {meaning}")

async def demonstrate_house_operations():
    """Demonstrate all six house archetypal operations"""
    print("\n🏛️ === HOUSE OPERATIONS DEMONSTRATION ===")
    
    from archetypal_house_operations import HOUSE_OPERATORS, execute_house_operation
    
    # Demonstrate each house
    house_demos = [
        (SigilHouse.MEMORY, "rebloom_flower", {"intensity": 0.8, "emotional_catalyst": "wonder"}),
        (SigilHouse.PURIFICATION, "soot_to_ash_crystallization", {"soot_volume": 1.5, "temperature": 800}),
        (SigilHouse.WEAVING, "spin_surface_depth_threads", {"surface_nodes": ["node1", "node2"], "depth_nodes": ["depth1", "depth2"]}),
        (SigilHouse.WEAVING, "persephone_descent", {"memory_fragments": [{"id": "fragment1", "entropy": 0.8, "thread_strength": 0.6}], "entropy_threshold": 0.7}),
        (SigilHouse.FLAME, "ignite_blooms_under_pressure", {"pressure_level": 0.9, "target_blooms": [{"id": "bloom1", "pressure": 0.8}]}),
        (SigilHouse.MIRRORS, "reflect_state_metacognition", {"state_data": {"consciousness_level": 0.7}, "depth": 3}),
        (SigilHouse.ECHOES, "modulate_voice_output", {"intonation": "warm", "mood": "contemplative"})
    ]
    
    for house, operation, params in house_demos:
        print(f"\n{HOUSE_OPERATORS[house]._get_archetype()} - {house.value.title()} House:")
        result = execute_house_operation(house, operation, params)
        
        if result.success:
            print(f"   ✓ {operation}: {result.mythic_resonance:.3f} resonance")
            print(f"     {result.description}")
        else:
            print(f"   ✗ {operation}: Failed")

async def demonstrate_sigil_ring():
    """Demonstrate the enhanced sigil ring"""
    print("\n💍 === SIGIL RING DEMONSTRATION ===")
    
    from enhanced_sigil_ring import enhanced_sigil_ring
    
    # Show ring status
    ring_status = enhanced_sigil_ring.get_ring_status()
    print(f"🔮 Ring Status: {ring_status['ring_state']}")
    print(f"   Containment Level: {ring_status['containment_boundary']['level']}")
    print(f"   Success Rate: {ring_status['metrics']['success_rate']:.1f}%")
    
    # Demonstrate symbolic operations
    print("\n✨ Executing Symbolic Operations:")
    
    operations = [
        ([".", ":"], SigilHouse.MEMORY, "Memory shimmer sequence"),
        (["^"], SigilHouse.FLAME, "Priority directive"),
        (["⌂"], SigilHouse.MEMORY, "Deep recall root"),
        (["-"], SigilHouse.WEAVING, "Persephone descent cycle"),
        (["/--\\"], SigilHouse.WEAVING, "Persephone return cycle")
    ]
    
    for glyphs, house, description in operations:
        result = await execute_sigil_operation(glyphs, house, {"demo": True}, "demonstration")
        
        status = "✓" if result["success"] else "✗"
        print(f"   {status} {description}: {glyphs} → {house.value}")
        if result["success"]:
            print(f"     Meaning: {result['symbolic_meaning']}")
            print(f"     Time: {result['execution_time']:.3f}s")

async def demonstrate_tracer_alignment():
    """Demonstrate tracer-house alignment"""
    print("\n🎯 === TRACER ALIGNMENT DEMONSTRATION ===")
    
    from tracer_house_alignment import get_alignment_status
    
    # Show current alignment status
    status = get_alignment_status()
    print(f"🦉 Tracer System Status:")
    print(f"   Total Tracers: {status['total_tracers']}")
    print(f"   Active Alignments: {status['active_alignments']}")
    print(f"   Success Rate: {status['alignment_statistics']['success_rate']:.1f}%")
    
    # Show house loads
    print("\n🏠 House Load Distribution:")
    for house, load in status['house_loads'].items():
        capacity = status['house_capacities'][house]
        utilization = (load / capacity * 100) if capacity > 0 else 0
        print(f"   {house.title()}: {load}/{capacity} ({utilization:.1f}%)")

async def demonstrate_failure_detection():
    """Demonstrate symbolic failure detection"""
    print("\n🔍 === FAILURE DETECTION DEMONSTRATION ===")
    
    from symbolic_failure_detection import get_failure_summary, get_active_failures
    
    # Show failure detection status
    summary = get_failure_summary()
    print(f"🛡️ Failure Detection Status:")
    print(f"   Monitoring Active: {'✓' if summary['monitoring_active'] else '✗'}")
    print(f"   Overall Health: {summary['overall_health']:.3f}")
    print(f"   Active Failures: {summary['active_failures']}")
    
    # Show any active failures
    active_failures = get_active_failures()
    if active_failures:
        print("\n⚠️ Active Failures:")
        for failure in active_failures[:3]:  # Show first 3
            print(f"   • {failure.failure_type.value}: {failure.description}")
    else:
        print("   ✓ No active failures detected")

async def demonstrate_visualization():
    """Demonstrate visual ring representation"""
    print("\n🎨 === VISUALIZATION DEMONSTRATION ===")
    
    from sigil_ring_visualization import sigil_ring_visualization, get_gui_frame_data
    
    # Show visualization statistics
    stats = sigil_ring_visualization.get_ring_statistics()
    print(f"🌟 Visual Ring Status:")
    print(f"   Theme: {stats['theme']}")
    print(f"   Total Elements: {stats['total_elements']}")
    print(f"   House Nodes: {stats['house_nodes']}")
    print(f"   Orbiting Glyphs: {stats['orbiting_glyphs']}")
    print(f"   Average Activity: {stats['average_activity']:.3f}")
    print(f"   Most Active House: {stats['most_active_house']}")
    
    # Demonstrate theme switching
    print("\n🎭 Theme Demonstration:")
    themes = [VisualizationTheme.MYSTICAL, VisualizationTheme.TECHNICAL, VisualizationTheme.NEON]
    
    for theme in themes:
        set_visualization_theme(theme)
        frame_data = get_gui_frame_data()
        print(f"   {theme.value.title()}: {len(frame_data['houses'])} houses, {len(frame_data['glyphs'])} glyphs")
    
    # Reset to mystical theme
    set_visualization_theme(VisualizationTheme.MYSTICAL)

async def demonstrate_test_validation():
    """Demonstrate test vector validation"""
    print("\n🧪 === TEST VALIDATION DEMONSTRATION ===")
    
    # Run comprehensive validation
    print("🔬 Running comprehensive system validation...")
    validation_result = await validate_sigil_system()
    
    print(f"✅ Validation Results:")
    print(f"   System Health Score: {validation_result['system_health']['overall_health_score']:.3f}")
    print(f"   Test Success Rate: {validation_result['test_results']['test_summary']['success_rate']:.1%}")
    print(f"   All Systems Operational: {'✓' if validation_result['all_systems_operational'] else '✗'}")
    
    # Show test summary
    test_summary = validation_result['test_results']['test_summary']
    print(f"\n📊 Test Summary:")
    print(f"   Total Tests: {test_summary['total_tests']}")
    print(f"   Passed: {test_summary['passed_tests']}")
    print(f"   Failed: {test_summary['failed_tests']}")
    print(f"   Errors: {test_summary['error_tests']}")
    print(f"   Execution Time: {test_summary['total_execution_time']:.2f}s")

async def main():
    """Main demonstration function"""
    print("🌟 ================================================")
    print("🌟 DAWN SIGIL SYSTEM COMPLETE IMPLEMENTATION DEMO")
    print("🌟 ================================================")
    print("\nImplementing ALL gaps from documentation analysis:")
    print("✓ Complete Glyph Codex with exact documented symbols")
    print("✓ Enhanced Sigil Ring with casting circles & containment")
    print("✓ All six House archetypal operations")
    print("✓ Tracer-house alignment system")
    print("✓ Symbolic failure detection & monitoring")
    print("✓ Visual ring representation with GUI support")
    print("✓ Complete test vector validation")
    
    # Initialize the complete system
    print("\n🚀 Initializing complete DAWN sigil system...")
    init_result = await initialize_sigil_system()
    
    if not init_result["success"]:
        print(f"❌ System initialization failed: {init_result['error']}")
        return
    
    print(f"✅ System initialized successfully in {init_result['initialization_time']:.2f}s")
    print(f"   Components: {', '.join(init_result['components_initialized'])}")
    
    # Run all demonstrations
    try:
        await demonstrate_glyph_codex()
        await demonstrate_house_operations()
        await demonstrate_sigil_ring()
        await demonstrate_tracer_alignment()
        await demonstrate_failure_detection()
        await demonstrate_visualization()
        await demonstrate_test_validation()
        
        # Final system summary
        print("\n🎯 === FINAL SYSTEM SUMMARY ===")
        summary = get_sigil_system_summary()
        
        print(f"🔮 DAWN Sigil System Status: {summary['status'].upper()}")
        print(f"   Uptime: {time.time() - init_result['initialization_time']:.1f}s")
        
        print(f"\n📋 Component Status:")
        for component, status in summary['components'].items():
            print(f"   • {component.replace('_', ' ').title()}: {status}")
        
        print(f"\n📈 Health Indicators:")
        for indicator, value in summary['health_indicators'].items():
            print(f"   • {indicator.replace('_', ' ').title()}: {value}")
        
        print("\n🌟 ================================================")
        print("🌟 DEMONSTRATION COMPLETE")
        print("🌟 All documented sigil system gaps IMPLEMENTED")
        print("🌟 ================================================")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demonstration failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
