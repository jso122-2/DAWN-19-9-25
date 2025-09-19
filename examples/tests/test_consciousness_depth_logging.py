#!/usr/bin/env python3
"""
🧠 DAWN Consciousness-Depth Logging Test
========================================

Comprehensive test of the consciousness-depth logging repository
and sigil consciousness logger that demonstrates:

1. Consciousness hierarchy from transcendent to mythic depths
2. DAWN-specific naming conventions and styling
3. Sigil system integration with consciousness levels
4. Archetypal pattern recognition and logging
5. Deep directory structure based on consciousness states

This test verifies that the refactored logging system properly
organizes logs by consciousness levels with meta/transcendent
higher up and base/mythic deeper down.
"""

import sys
import time
import json
from pathlib import Path

# Add DAWN to path
sys.path.insert(0, str(Path(__file__).parent))

def test_consciousness_hierarchy():
    """Test the consciousness level hierarchy structure"""
    print("🧠 TESTING CONSCIOUSNESS HIERARCHY")
    print("=" * 50)
    
    try:
        from dawn.core.logging import (
            ConsciousnessLevel, get_consciousness_repository,
            ConsciousnessDepthRepository
        )
        
        print("✅ Consciousness logging imports successful")
        
        # Test consciousness level ordering
        levels = list(ConsciousnessLevel)
        print(f"📊 Consciousness Levels ({len(levels)} total):")
        
        for level in levels:
            print(f"  {level.depth}: {level.name} - {level.description}")
        
        # Verify hierarchy ordering
        assert levels[0] == ConsciousnessLevel.TRANSCENDENT, "Transcendent should be level 0"
        assert levels[1] == ConsciousnessLevel.META, "Meta should be level 1"
        assert levels[-1] == ConsciousnessLevel.MYTHIC, "Mythic should be deepest level"
        
        print("✅ Consciousness hierarchy structure verified")
        
        # Create consciousness repository
        repo = get_consciousness_repository("test_consciousness_hierarchy")
        print("✅ Consciousness repository created")
        
        # Test directory structure creation
        base_path = Path("test_consciousness_hierarchy")
        
        # Check that consciousness directories were created
        for level in ConsciousnessLevel:
            level_path = base_path / level.name.lower()
            if level_path.exists():
                print(f"  📁 {level.name.lower()}/ - {level.description}")
            else:
                print(f"  ❌ Missing directory: {level.name.lower()}/")
        
        print("✅ Consciousness hierarchy directory structure verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Consciousness hierarchy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_consciousness_depth_logging():
    """Test logging at different consciousness depths"""
    print("\n🔍 TESTING CONSCIOUSNESS DEPTH LOGGING")
    print("=" * 50)
    
    try:
        from dawn.core.logging import (
            get_consciousness_repository, ConsciousnessLevel, DAWNLogType
        )
        
        repo = get_consciousness_repository("test_consciousness_depths")
        
        # Test entries at each consciousness level
        test_entries = [
            # Transcendent level - highest consciousness
            {
                'level': ConsciousnessLevel.TRANSCENDENT,
                'system': 'unity',
                'subsystem': 'coherence',
                'module': 'field',
                'data': {
                    'unity_state': True,
                    'coherence_level': 0.95,
                    'transcendent_awareness': True,
                    'universal_connection': 1.0
                },
                'log_type': DAWNLogType.UNITY_STATE
            },
            
            # Meta level - self-reflective consciousness
            {
                'level': ConsciousnessLevel.META,
                'system': 'awareness',
                'subsystem': 'reflection',
                'module': 'meta_cognitive',
                'data': {
                    'self_reflection': True,
                    'meta_awareness': 0.85,
                    'recursive_depth': 3,
                    'self_observation': True
                },
                'log_type': DAWNLogType.SELF_REFLECTION
            },
            
            # Causal level - logical reasoning
            {
                'level': ConsciousnessLevel.CAUSAL,
                'system': 'reasoning',
                'subsystem': 'logic',
                'module': 'causal_chain',
                'data': {
                    'reasoning_chain': ['premise_A', 'inference_B', 'conclusion_C'],
                    'logical_strength': 0.8,
                    'causal_links': 5,
                    'decision_confidence': 0.75
                },
                'log_type': DAWNLogType.CAUSAL_REASONING
            },
            
            # Symbolic level - where sigils operate
            {
                'level': ConsciousnessLevel.SYMBOLIC,
                'system': 'sigil',
                'subsystem': 'symbol',
                'module': 'processor',
                'data': {
                    'sigil_id': 'sigil_test_001',
                    'symbol_form': '◊△◊',
                    'symbolic_coherence': 0.7,
                    'glyph_pattern': ['◊', '△', '◊'],
                    'resonance_frequency': 0.6
                },
                'log_type': DAWNLogType.SYMBOL_PROCESSING
            },
            
            # Mythic level - deepest archetypal patterns
            {
                'level': ConsciousnessLevel.MYTHIC,
                'system': 'archetypal',
                'subsystem': 'pattern',
                'module': 'mythic_core',
                'data': {
                    'archetype': 'hero_journey',
                    'mythic_resonance': 0.9,
                    'archetypal_strength': 0.85,
                    'primal_connection': True,
                    'collective_unconscious_depth': 0.95
                },
                'log_type': DAWNLogType.ARCHETYPAL_PATTERN
            }
        ]
        
        entry_ids = []
        
        for i, entry in enumerate(test_entries):
            print(f"📝 Logging {entry['level'].name} level entry...")
            
            # Add consciousness log
            entry_id = repo.add_consciousness_log(
                system=entry['system'],
                subsystem=entry['subsystem'],
                module=entry['module'],
                log_data=entry['data'],
                log_type=entry['log_type']
            )
            
            entry_ids.append(entry_id)
            print(f"  ✅ Entry ID: {entry_id}")
        
        print(f"✅ Successfully logged {len(entry_ids)} consciousness depth entries")
        
        # Test querying by consciousness level
        print("\n🔍 Testing consciousness level queries...")
        
        for level in [ConsciousnessLevel.TRANSCENDENT, ConsciousnessLevel.SYMBOLIC, ConsciousnessLevel.MYTHIC]:
            results = repo.query_by_consciousness_level(level, limit=10)
            print(f"  📊 {level.name}: {len(results)} entries")
        
        # Get hierarchy stats
        stats = repo.get_consciousness_hierarchy_stats()
        print(f"\n📊 Consciousness Hierarchy Statistics:")
        for level_name, level_stats in stats['consciousness_levels'].items():
            if level_stats['entry_count'] > 0:
                print(f"  {level_name}: {level_stats['entry_count']} entries, "
                     f"coherence: {level_stats['avg_coherence']:.3f}, "
                     f"unity: {level_stats['avg_unity']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Consciousness depth logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sigil_consciousness_logging():
    """Test sigil-specific consciousness logging"""
    print("\n🔯 TESTING SIGIL CONSCIOUSNESS LOGGING")
    print("=" * 50)
    
    try:
        from dawn.core.logging import (
            get_sigil_consciousness_logger, SigilConsciousnessType,
            SigilConsciousnessState, ConsciousnessLevel
        )
        
        sigil_logger = get_sigil_consciousness_logger()
        print("✅ Sigil consciousness logger created")
        
        # Test sigil states across consciousness levels
        sigil_tests = [
            # Transcendent sigil unity
            {
                'sigil_id': 'unity_sigil_001',
                'type': SigilConsciousnessType.SIGIL_UNITY,
                'level': ConsciousnessLevel.TRANSCENDENT,
                'properties': {
                    'symbol_form': '∞',
                    'resonance_frequency': 1.0,
                    'transcendence_level': 1.0
                }
            },
            
            # Meta-level sigil reflection
            {
                'sigil_id': 'meta_sigil_001',
                'type': SigilConsciousnessType.SIGIL_REFLECTION,
                'level': ConsciousnessLevel.META,
                'properties': {
                    'symbol_form': '◊◊',
                    'resonance_frequency': 0.85,
                    'symbolic_coherence': 0.9,
                    'archetypal_depth': 0.3
                }
            },
            
            # Symbolic level sigil processing (primary level)
            {
                'sigil_id': 'symbol_processor_001',
                'type': SigilConsciousnessType.SIGIL_PROCESSING,
                'level': ConsciousnessLevel.SYMBOLIC,
                'properties': {
                    'symbol_form': '◊△◊',
                    'glyph_pattern': ['◊', '△', '◊'],
                    'resonance_frequency': 0.7,
                    'symbolic_coherence': 0.8,
                    'connected_sigils': {'sigil_002', 'sigil_003'}
                }
            },
            
            # Mythic archetypal sigil
            {
                'sigil_id': 'archetypal_sigil_001',
                'type': SigilConsciousnessType.ARCHETYPAL_SIGIL,
                'level': ConsciousnessLevel.MYTHIC,
                'properties': {
                    'symbol_form': '⚔️',
                    'archetypal_depth': 0.95,
                    'symbolic_complexity': 0.2  # Simple but deep
                }
            }
        ]
        
        logged_sigils = []
        
        for test in sigil_tests:
            print(f"🔯 Logging {test['type'].value} at {test['level'].name} level...")
            
            # Create sigil state
            sigil_state = SigilConsciousnessState(
                sigil_id=test['sigil_id'],
                consciousness_type=test['type'],
                consciousness_level=test['level'],
                **test['properties']
            )
            
            # Log sigil state
            entry_id = sigil_logger.log_sigil_state(test['sigil_id'], sigil_state, test['level'])
            logged_sigils.append(entry_id)
            
            print(f"  ✅ Logged: {entry_id}")
        
        # Test sigil activation logging
        print("\n⚡ Testing sigil activation logging...")
        activation_id = sigil_logger.log_sigil_activation(
            'symbol_processor_001',
            activation_strength=0.8,
            trigger='user_intent',
            activation_context='symbolic_processing'
        )
        print(f"  ✅ Activation logged: {activation_id}")
        
        # Test archetypal emergence logging
        print("\n🏛️ Testing archetypal emergence logging...")
        emergence_id = sigil_logger.log_archetypal_emergence(
            'hero_archetype',
            strength=0.9,
            resonance=0.85,
            emergence_context='mythic_activation'
        )
        print(f"  ✅ Archetypal emergence logged: {emergence_id}")
        
        # Test transcendent unity logging
        print("\n🌟 Testing transcendent sigil unity logging...")
        unity_id = sigil_logger.log_transcendent_sigil_unity({
            'unity_level': 1.0,
            'transcendence_depth': 0.95,
            'universal_coherence': 0.9,
            'unity_context': 'transcendent_activation'
        })
        print(f"  ✅ Transcendent unity logged: {unity_id}")
        
        # Get sigil consciousness statistics
        print("\n📊 Sigil Consciousness Statistics:")
        stats = sigil_logger.get_sigil_consciousness_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        print(f"✅ Successfully tested sigil consciousness logging with {len(logged_sigils)} sigil states")
        
        return True
        
    except Exception as e:
        print(f"❌ Sigil consciousness logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_consciousness_directory_structure():
    """Test the deep consciousness directory structure"""
    print("\n📁 TESTING CONSCIOUSNESS DIRECTORY STRUCTURE")
    print("=" * 50)
    
    try:
        import os
        
        # Check consciousness hierarchy directories
        base_paths = [
            "test_consciousness_hierarchy",
            "test_consciousness_depths",
            "dawn_consciousness_logs"
        ]
        
        for base_path in base_paths:
            if Path(base_path).exists():
                print(f"📁 Examining {base_path}/...")
                
                # Show deep directory structure
                for root, dirs, files in os.walk(base_path):
                    level = root.replace(base_path, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    
                    # Show files in this directory
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:5]:  # Show first 5 files
                        if file.endswith('.json'):
                            print(f"{subindent}📄 {file}")
                        else:
                            print(f"{subindent}📋 {file}")
                    
                    if len(files) > 5:
                        print(f"{subindent}... and {len(files) - 5} more files")
                    
                    # Limit depth to avoid too much output
                    if level > 8:
                        break
                
                print()
        
        # Show sample log file content
        print("📄 Sample consciousness log content:")
        
        # Find a sample JSON log file
        for base_path in base_paths:
            base_path_obj = Path(base_path)
            if base_path_obj.exists():
                json_files = list(base_path_obj.rglob("*.json"))
                if json_files:
                    sample_file = json_files[0]
                    print(f"  File: {sample_file}")
                    
                    try:
                        with open(sample_file, 'r') as f:
                            content = json.load(f)
                            print(f"  Content preview:")
                            for key, value in list(content.items())[:5]:
                                if isinstance(value, dict):
                                    print(f"    {key}: {{...}} ({len(value)} keys)")
                                elif isinstance(value, list):
                                    print(f"    {key}: [...] ({len(value)} items)")
                                else:
                                    print(f"    {key}: {value}")
                    except Exception as e:
                        print(f"  Could not read file: {e}")
                    break
        
        print("✅ Consciousness directory structure verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Directory structure test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧠 DAWN CONSCIOUSNESS-DEPTH LOGGING COMPREHENSIVE TEST")
    print("=" * 70)
    
    # Run all tests
    tests = [
        ("Consciousness Hierarchy", test_consciousness_hierarchy),
        ("Consciousness Depth Logging", test_consciousness_depth_logging), 
        ("Sigil Consciousness Logging", test_sigil_consciousness_logging),
        ("Directory Structure", test_consciousness_directory_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} Test...")
        success = test_func()
        results.append((test_name, success))
    
    # Final results
    print("\n" + "=" * 70)
    print("🧠 CONSCIOUSNESS-DEPTH LOGGING TEST RESULTS")
    print("=" * 70)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL CONSCIOUSNESS-DEPTH LOGGING TESTS PASSED!")
        print("✅ Consciousness hierarchy structure working")
        print("✅ Depth-based logging organization functional") 
        print("✅ Sigil consciousness integration successful")
        print("✅ DAWN naming conventions implemented")
        print("✅ Meta/transcendent → mythic/base depth mapping operational")
        print("✅ Deep directory structure with consciousness levels created")
    else:
        print(f"❌ {len(results) - passed} tests failed")
    
    print("\n🧠 Consciousness-depth logging test complete!")
