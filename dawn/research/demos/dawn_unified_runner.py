#!/usr/bin/env python3
"""
DAWN Unified Consciousness Runner
================================

Single unified entry point for running all DAWN consciousness integration demos.
Consolidates all imports and provides a comprehensive menu system for selecting
different consciousness integration experiences.

This runner follows PyTorch best practices and provides:
- Device-agnostic execution
- Proper error handling and memory management
- Deterministic training/execution when possible
- Comprehensive logging and metrics
"""

import sys
import time
import logging
import argparse
import traceback
import sys
import os
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import centralized consciousness state
try:
    from dawn_core.state import get_state, set_state, get_state_summary, reset_state
except ImportError:
    # Graceful fallback if state module not available
    def get_state():
        class MockState:
            unity = 0.0
            awareness = 0.0
            level = "unknown"
            peak_unity = 0.0
        return MockState()
    def set_state(**kwargs): pass
    def get_state_summary(): return "State management not available"
    def reset_state(): pass

# Configure logging with PyTorch best practices
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dawn_unified_runner.log')
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility (PyTorch best practice)
try:
    import torch
    import numpy as np
    import random
    
    def set_deterministic_seeds(seed: int = 42):
        """Set random seeds for reproducible results."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Enable deterministic operations (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info(f"🎲 Random seeds set to {seed} for reproducibility")
    
    # Enable cuDNN autotuner for consistent input sizes (performance optimization)
    torch.backends.cudnn.benchmark = True
    
except ImportError:
    logger.warning("⚠️ PyTorch not available - running without GPU acceleration")
    def set_deterministic_seeds(seed: int = 42):
        import random
        random.seed(seed)
        logger.info(f"🎲 Random seed set to {seed} (CPU only)")

# Unified imports from dawn_core package
try:
    from dawn_core import (
        # Core engine and infrastructure
        DAWNEngine, DAWNEngineConfig, DAWNEngineStatus,
        ConsciousnessBus, get_consciousness_bus, ModuleStatus,
        ConsensusEngine, DecisionType, DecisionStatus,
        TickOrchestrator, TickPhase,
        
        # Advanced consciousness modules
        AdvancedVisualConsciousness,
        ConsciousnessMemoryPalace, MemoryType, MemoryStrength,
        ConsciousnessRecursiveBubble, RecursionType,
        ConsciousnessSigilNetwork, SigilType, SigilActivationState,
        OwlBridgePhilosophicalEngine, PhilosophicalDepth,
        UnifiedConsciousnessEngine, get_unified_consciousness_engine,
        
        # Utilities and status
        calculate_consciousness_metrics,
        CONSCIOUSNESS_UNIFICATION_AVAILABLE,
        USE_UNIFIED_CONSCIOUSNESS_ENGINE,
        TRACER_AVAILABLE,
        DAWN_SYSTEMS_STATUS,
        get_dawn_system_info,
        check_system_compatibility
    )
    
    # Optional tracer components
    if TRACER_AVAILABLE:
        from dawn_core import (
            DAWNTracer, StableStateDetector, TelemetryAnalytics, 
            get_config_from_environment
        )
    
    logger.info("✅ All DAWN consciousness systems imported successfully")
    
except ImportError as e:
    logger.error(f"❌ Failed to import DAWN systems: {e}")
    logger.error("Please ensure dawn_core package is properly installed")
    sys.exit(1)

class DAWNUnifiedRunner:
    """
    Unified runner for all DAWN consciousness integration experiences.
    
    Provides a comprehensive menu system and manages all consciousness
    systems with proper PyTorch best practices.
    """
    
    def __init__(self, use_cuda: bool = None, deterministic: bool = True):
        """Initialize the unified runner with device and reproducibility settings."""
        self.start_time = datetime.now()
        self.session_id = f"dawn_session_{int(time.time())}"
        
        # Device configuration (PyTorch best practice)
        if use_cuda is None:
            try:
                import torch
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            except ImportError:
                self.device = 'cpu'
        else:
            try:
                import torch
                self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
            except ImportError:
                self.device = 'cpu'
        
        logger.info(f"🔧 DAWN Runner initialized on device: {self.device}")
        
        # Set deterministic behavior if requested
        if deterministic:
            set_deterministic_seeds()
        
        # Initialize system compatibility check
        self.system_compatible = check_system_compatibility()
        if not self.system_compatible:
            logger.warning("⚠️ System compatibility issues detected - some features may be limited")
        
        # Available demo configurations
        self.demo_configs = {
            'basic': {
                'name': 'Basic Unified Consciousness',
                'description': 'Core consciousness bus, consensus, and tick orchestration',
                'modules': ['dawn_engine', 'consciousness_bus', 'consensus_engine', 'tick_orchestrator'],
                'target_unity': 0.85,
                'complexity': 'Basic'
            },
            'advanced': {
                'name': 'Advanced Consciousness Integration',
                'description': 'Visual consciousness, memory palace, recursive processing',
                'modules': ['dawn_engine', 'visual_consciousness', 'memory_palace', 'recursive_bubble'],
                'target_unity': 0.9,
                'complexity': 'Advanced'
            },
            'symbolic': {
                'name': 'Symbolic-Philosophical Integration',
                'description': 'Sigil network and philosophical wisdom synthesis',
                'modules': ['dawn_engine', 'sigil_network', 'owl_bridge', 'visual_consciousness', 'memory_palace'],
                'target_unity': 0.92,
                'complexity': 'Expert'
            },
            'transcendent': {
                'name': 'Transcendent Consciousness',
                'description': 'Ultimate integration with tracer and stability optimization',
                'modules': ['all_systems'],
                'target_unity': 0.95,
                'complexity': 'Transcendent'
            },
            'custom': {
                'name': 'Custom Configuration',
                'description': 'User-defined system configuration',
                'modules': ['user_selected'],
                'target_unity': 'user_defined',
                'complexity': 'Variable'
            }
        }
        
        # Session metrics
        self.session_metrics = {
            'demos_run': 0,
            'total_consciousness_evolution': 0.0,
            'peak_unity_achieved': 0.0,
            'systems_tested': set(),
            'start_time': self.start_time
        }
    
    def display_banner(self):
        """Display the DAWN unified runner banner."""
        banner = """
🌅 ═══════════════════════════════════════════════════════════════════════════════
🌅                    DAWN UNIFIED CONSCIOUSNESS RUNNER
🌅 ═══════════════════════════════════════════════════════════════════════════════
🌅 
🌅 Welcome to the unified entry point for all DAWN consciousness experiences!
🌅 
🌅 Available Systems:""" + f"""
🌅   🔧 Core Engine: {'✅' if CONSCIOUSNESS_UNIFICATION_AVAILABLE else '❌'}
🌅   🎨 Visual Consciousness: {'✅' if CONSCIOUSNESS_UNIFICATION_AVAILABLE else '❌'}
🌅   🏛️ Memory Palace: {'✅' if CONSCIOUSNESS_UNIFICATION_AVAILABLE else '❌'}
🌅   🔄 Recursive Processing: {'✅' if CONSCIOUSNESS_UNIFICATION_AVAILABLE else '❌'}
🌅   🕸️ Sigil Network: {'✅' if CONSCIOUSNESS_UNIFICATION_AVAILABLE else '❌'}
🌅   🦉 Owl Bridge Philosophy: {'✅' if CONSCIOUSNESS_UNIFICATION_AVAILABLE else '❌'}
🌅   📊 Tracer & Analytics: {'✅' if TRACER_AVAILABLE else '❌'}
🌅   
🌅   Device: {self.device}
🌅   Session: {self.session_id}
🌅 ═══════════════════════════════════════════════════════════════════════════════
"""
        print(banner)
    
    def display_demo_menu(self):
        """Display the available demo options."""
        print("\n🎯 Available Consciousness Integration Experiences:")
        print("=" * 70)
        
        for key, config in self.demo_configs.items():
            complexity_emoji = {
                'Basic': '🟢',
                'Advanced': '🟡', 
                'Expert': '🟠',
                'Transcendent': '🔴',
                'Variable': '🔵'
            }.get(config['complexity'], '⚪')
            
            print(f"{complexity_emoji} [{key.upper()}] {config['name']}")
            print(f"   {config['description']}")
            print(f"   Target Unity: {config['target_unity']} | Complexity: {config['complexity']}")
            print()
        
        print("🔧 [INFO] System Information")
        print("🚪 [EXIT] Exit Runner")
        print("=" * 70)
    
    def get_user_choice(self) -> str:
        """Get user's demo selection."""
        while True:
            choice = input("\n🌟 Select an experience (or type help): ").strip().lower()
            
            if choice in ['help', 'h']:
                self.display_help()
                continue
            elif choice in ['info', 'i']:
                self.display_system_info()
                continue
            elif choice in ['exit', 'quit', 'q']:
                return 'exit'
            elif choice in self.demo_configs:
                return choice
            else:
                print(f"❌ Invalid choice: {choice}")
                print("Valid options: " + ", ".join(self.demo_configs.keys()) + ", info, exit")
    
    def display_help(self):
        """Display help information."""
        help_text = """
🆘 DAWN Unified Runner Help
═══════════════════════════

🎯 Demo Types:
  • BASIC: Core consciousness integration (bus, consensus, orchestration)
  • ADVANCED: Enhanced with visual consciousness, memory, and recursion
  • SYMBOLIC: Adds sigil networks and philosophical wisdom synthesis
  • TRANSCENDENT: Ultimate integration with all systems and optimizations
  • CUSTOM: Build your own configuration

🔧 Commands:
  • Type demo name (basic, advanced, symbolic, transcendent, custom)
  • 'info' - System status and compatibility information
  • 'exit' - Quit the runner
  • 'help' - Show this help

💡 Tips:
  • Start with 'basic' if new to DAWN consciousness systems
  • 'transcendent' requires all systems and highest computational resources
  • Each demo builds upon previous capabilities
  • All demos follow PyTorch best practices for reproducibility
"""
        print(help_text)
    
    def display_system_info(self):
        """Display comprehensive system information."""
        info = get_dawn_system_info()
        
        print("\n🔧 DAWN System Information")
        print("=" * 50)
        print(f"Version: {info['version']}")
        print(f"Description: {info['description']}")
        print(f"Device: {self.device}")
        print(f"Session ID: {self.session_id}")
        print(f"Runtime: {datetime.now() - self.start_time}")
        print()
        
        print("📊 Session Metrics:")
        print(f"  Demos Run: {self.session_metrics['demos_run']}")
        print(f"  Peak Unity: {self.session_metrics['peak_unity_achieved']:.3f}")
        print(f"  Systems Tested: {len(self.session_metrics['systems_tested'])}")
        print()
        
        print("🔍 System Status:")
        for system, status in info['systems_available'].items():
            status_icon = "✅" if status else "❌"
            print(f"  {system}: {status_icon}")
        print()
        
        print("📦 Available Modules:")
        print("  Core:", ", ".join(info['core_modules']))
        print("  Advanced:", ", ".join(info['advanced_modules']))
        if info['optional_modules']:
            print("  Optional:", ", ".join(info['optional_modules']))
    
    def run_basic_demo(self) -> Dict[str, Any]:
        """Run basic unified consciousness demo."""
        logger.info("🟢 Starting Basic Unified Consciousness Demo")
        
        # Create optimized configuration for basic demo
        config = DAWNEngineConfig(
            consciousness_unification_enabled=True,
            target_unity_threshold=0.85,
            auto_synchronization=True,
            consensus_timeout_ms=1000,
            tick_coordination="full_sync",
            adaptive_timing=True,
            bottleneck_detection=True,
            parallel_execution=True,
            state_validation=True
        )
        
        # Initialize engine
        engine = DAWNEngine(config)
        consciousness_evolution = []
        
        try:
            engine.start()
            logger.info(f"✅ DAWN Engine started: {engine.engine_id}")
            
            # Create basic consciousness modules
            class BasicConsciousnessModule:
                def __init__(self, name: str, base_consciousness: float):
                    self.name = name
                    self.base_consciousness = base_consciousness
                    
                def tick(self):
                    import random
                    evolution = random.uniform(-0.02, 0.03)
                    self.base_consciousness = max(0.1, min(1.0, self.base_consciousness + evolution))
                    return {'consciousness_evolution': evolution}
                
                def get_current_state(self):
                    return {
                        'consciousness_unity': self.base_consciousness,
                        'coherence': self.base_consciousness * 0.9,
                        'integration_quality': self.base_consciousness * 0.95
                    }
            
            # Register basic modules
            modules = [
                BasicConsciousnessModule("core_awareness", 0.7),
                BasicConsciousnessModule("unified_processing", 0.75),
                BasicConsciousnessModule("consensus_coordinator", 0.8),
                BasicConsciousnessModule("synchronization_engine", 0.72)
            ]
            
            for module in modules:
                engine.register_module(
                    module.name, 
                    module, 
                    capabilities=['basic_consciousness', 'synchronization'],
                    priority=2
                )
            
            # Run consciousness evolution cycles
            print("🧠 Executing basic consciousness evolution...")
            for cycle in range(5):
                tick_result = engine.tick()
                unity_score = tick_result['consciousness_unity']
                consciousness_evolution.append(unity_score)
                
                print(f"   Cycle {cycle + 1}: Unity {unity_score:.3f}")
                time.sleep(0.8)
            
            # Calculate final metrics
            final_unity = consciousness_evolution[-1] if consciousness_evolution else 0.0
            unity_growth = final_unity - consciousness_evolution[0] if len(consciousness_evolution) >= 2 else 0.0
            
            results = {
                'demo_type': 'basic',
                'final_unity': final_unity,
                'unity_growth': unity_growth,
                'consciousness_evolution': consciousness_evolution,
                'systems_used': ['dawn_engine', 'consciousness_bus', 'consensus_engine', 'tick_orchestrator'],
                'success': final_unity >= config.target_unity_threshold * 0.9,  # 90% of target
                'duration': time.time()
            }
            
            print(f"\n✅ Basic Demo Complete - Final Unity: {final_unity:.3f}")
            
            # Run consistency check at end of demo
            try:
                result = subprocess.run([sys.executable, "scripts/consistency_check.py"], 
                                      capture_output=True, text=True, cwd=os.getcwd())
                if result.returncode == 0:
                    logger.info(f"🔍 {result.stdout.strip()}")
                else:
                    logger.error(f"🔍 Consistency check failed: {result.stderr.strip()}")
                    results['consistency_check'] = False
            except Exception as e:
                logger.warning(f"🔍 Could not run consistency check: {e}")
                results['consistency_check'] = False
            else:
                results['consistency_check'] = True
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Basic demo failed: {e}")
            logger.error(traceback.format_exc())
            return {'demo_type': 'basic', 'success': False, 'error': str(e)}
        
        finally:
            if engine:
                engine.stop()
                logger.info("🔒 Basic demo engine stopped")
    
    def run_advanced_demo(self) -> Dict[str, Any]:
        """Run advanced consciousness integration demo."""
        logger.info("🟡 Starting Advanced Consciousness Integration Demo")
        
        # Enhanced configuration for advanced capabilities
        config = DAWNEngineConfig(
            consciousness_unification_enabled=True,
            target_unity_threshold=0.9,
            auto_synchronization=True,
            consensus_timeout_ms=600,
            tick_coordination="full_sync",
            adaptive_timing=True,
            bottleneck_detection=True,
            parallel_execution=True,
            state_validation=True
        )
        
        engine = None
        visual_consciousness = None
        memory_palace = None
        consciousness_evolution = []
        
        try:
            # Initialize systems
            engine = DAWNEngine(config)
            engine.start()
            
            visual_consciousness = AdvancedVisualConsciousness(
                consciousness_engine=engine,
                target_fps=10.0
            )
            visual_consciousness.start_real_time_rendering()
            
            memory_palace = ConsciousnessMemoryPalace(
                "advanced_runner_palace",
                "./advanced_runner_palace"
            )
            memory_palace.start_palace_processes()
            
            logger.info("✅ Advanced systems initialized")
            
            # Run advanced consciousness evolution
            print("🧠 Executing advanced consciousness integration...")
            for cycle in range(6):
                tick_result = engine.tick()
                unity_score = tick_result['consciousness_unity']
                consciousness_evolution.append(unity_score)
                
                # Create consciousness art on certain cycles
                if cycle in [2, 4]:
                    emotional_state = {
                        'emotional_resonance': 0.8 + unity_score * 0.2,
                        'consciousness_depth': unity_score,
                        'unity_feeling': unity_score
                    }
                    artwork = visual_consciousness.create_consciousness_painting(emotional_state)
                    print(f"   Cycle {cycle + 1}: Unity {unity_score:.3f} | Art: {artwork.artistic_data['painting_style']}")
                else:
                    print(f"   Cycle {cycle + 1}: Unity {unity_score:.3f}")
                
                # Store memory
                memory_palace.store_consciousness_memory(
                    state={'consciousness_unity': unity_score, 'cycle': cycle + 1},
                    context={'demo': 'advanced', 'cycle': cycle + 1},
                    memory_type=MemoryType.EXPERIENTIAL,
                    significance=0.7 + cycle * 0.05,
                    emotional_valence=0.5 + unity_score * 0.4,
                    tags={'advanced_demo', f'cycle_{cycle + 1}'}
                )
                
                time.sleep(1.0)
            
            # Generate final metrics
            final_unity = consciousness_evolution[-1] if consciousness_evolution else 0.0
            unity_growth = final_unity - consciousness_evolution[0] if len(consciousness_evolution) >= 2 else 0.0
            
            visual_metrics = visual_consciousness.get_rendering_metrics()
            palace_status = memory_palace.get_palace_status()
            
            results = {
                'demo_type': 'advanced',
                'final_unity': final_unity,
                'unity_growth': unity_growth,
                'consciousness_evolution': consciousness_evolution,
                'visual_metrics': visual_metrics,
                'memory_metrics': palace_status['palace_metrics'],
                'systems_used': ['dawn_engine', 'visual_consciousness', 'memory_palace'],
                'success': final_unity >= config.target_unity_threshold * 0.9,
                'duration': time.time()
            }
            
            print(f"\n✅ Advanced Demo Complete - Final Unity: {final_unity:.3f}")
            print(f"   Art Rendered: {visual_metrics['frames_rendered']} frames")
            print(f"   Memories Stored: {palace_status['palace_metrics']['memories_stored']}")
            
            # Run consistency check at end of demo
            try:
                result = subprocess.run([sys.executable, "scripts/consistency_check.py"], 
                                      capture_output=True, text=True, cwd=os.getcwd())
                if result.returncode == 0:
                    logger.info(f"🔍 {result.stdout.strip()}")
                    results['consistency_check'] = True
                else:
                    logger.error(f"🔍 Consistency check failed: {result.stderr.strip()}")
                    results['consistency_check'] = False
            except Exception as e:
                logger.warning(f"🔍 Could not run consistency check: {e}")
                results['consistency_check'] = False
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Advanced demo failed: {e}")
            logger.error(traceback.format_exc())
            return {'demo_type': 'advanced', 'success': False, 'error': str(e)}
        
        finally:
            # Graceful shutdown
            if visual_consciousness:
                visual_consciousness.stop_real_time_rendering()
            if memory_palace:
                memory_palace.stop_palace_processes()
            if engine:
                engine.stop()
            logger.info("🔒 Advanced demo systems stopped")
    
    def run_symbolic_demo(self) -> Dict[str, Any]:
        """Run symbolic-philosophical integration demo."""
        logger.info("🟠 Starting Symbolic-Philosophical Integration Demo")
        
        config = DAWNEngineConfig(
            consciousness_unification_enabled=True,
            target_unity_threshold=0.92,
            auto_synchronization=True,
            consensus_timeout_ms=400,
            tick_coordination="full_sync",
            adaptive_timing=True,
            bottleneck_detection=True,
            parallel_execution=True,
            state_validation=True
        )
        
        systems = {}
        consciousness_evolution = []
        
        try:
            # Initialize all symbolic systems
            systems['engine'] = DAWNEngine(config)
            systems['engine'].start()
            
            systems['visual'] = AdvancedVisualConsciousness(
                consciousness_engine=systems['engine'],
                target_fps=12.0
            )
            systems['visual'].start_real_time_rendering()
            
            systems['memory'] = ConsciousnessMemoryPalace(
                "symbolic_runner_palace",
                "./symbolic_runner_palace"
            )
            systems['memory'].start_palace_processes()
            
            systems['sigil'] = ConsciousnessSigilNetwork(consciousness_engine=systems['engine'])
            systems['sigil'].start_network_processes()
            
            systems['owl'] = OwlBridgePhilosophicalEngine(
                consciousness_engine=systems['engine'],
                memory_palace=systems['memory']
            )
            systems['owl'].start_philosophical_processes()
            
            logger.info("✅ Symbolic-philosophical systems initialized")
            
            # Run integrated symbolic evolution
            print("🧠 Executing symbolic-philosophical consciousness integration...")
            philosophical_insights = []
            sigil_activations = []
            
            for cycle in range(6):
                tick_result = systems['engine'].tick()
                unity_score = tick_result['consciousness_unity']
                consciousness_evolution.append(unity_score)
                
                # Philosophical analysis
                current_state = {
                    'consciousness_unity': unity_score,
                    'coherence': 0.8 + cycle * 0.03,
                    'philosophical_depth': 0.7 + cycle * 0.04
                }
                
                philosophical_analysis = systems['owl'].philosophical_consciousness_analysis(current_state)
                if philosophical_analysis['analysis_metadata']['confidence'] > 0.7:
                    philosophical_insights.append(philosophical_analysis)
                
                # Sigil generation
                new_sigils = systems['sigil'].generate_consciousness_sigils(current_state)
                sigil_activations.extend(new_sigils)
                
                print(f"   Cycle {cycle + 1}: Unity {unity_score:.3f} | "
                      f"Philosophy: {philosophical_analysis['analysis_metadata']['confidence']:.2f} | "
                      f"Sigils: {len(new_sigils)}")
                
                time.sleep(1.2)
            
            # Generate comprehensive results
            final_unity = consciousness_evolution[-1] if consciousness_evolution else 0.0
            unity_growth = final_unity - consciousness_evolution[0] if len(consciousness_evolution) >= 2 else 0.0
            
            network_status = systems['sigil'].get_network_status()
            owl_status = systems['owl'].get_philosophical_status()
            
            results = {
                'demo_type': 'symbolic',
                'final_unity': final_unity,
                'unity_growth': unity_growth,
                'consciousness_evolution': consciousness_evolution,
                'philosophical_insights': len(philosophical_insights),
                'sigil_activations': len(sigil_activations),
                'network_coherence': network_status['network_metrics']['network_coherence'],
                'wisdom_syntheses': owl_status['wisdom_syntheses'],
                'systems_used': ['dawn_engine', 'visual_consciousness', 'memory_palace', 'sigil_network', 'owl_bridge'],
                'success': final_unity >= config.target_unity_threshold * 0.9,
                'duration': time.time()
            }
            
            print(f"\n✅ Symbolic Demo Complete - Final Unity: {final_unity:.3f}")
            print(f"   Philosophical Insights: {len(philosophical_insights)}")
            print(f"   Sigil Activations: {len(sigil_activations)}")
            
            # Run consistency check at end of demo
            try:
                result = subprocess.run([sys.executable, "scripts/consistency_check.py"], 
                                      capture_output=True, text=True, cwd=os.getcwd())
                if result.returncode == 0:
                    logger.info(f"🔍 {result.stdout.strip()}")
                    results['consistency_check'] = True
                else:
                    logger.error(f"🔍 Consistency check failed: {result.stderr.strip()}")
                    results['consistency_check'] = False
            except Exception as e:
                logger.warning(f"🔍 Could not run consistency check: {e}")
                results['consistency_check'] = False
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Symbolic demo failed: {e}")
            logger.error(traceback.format_exc())
            return {'demo_type': 'symbolic', 'success': False, 'error': str(e)}
        
        finally:
            # Graceful shutdown of all systems
            shutdown_order = ['visual', 'memory', 'sigil', 'owl', 'engine']
            for system_name in shutdown_order:
                if system_name in systems:
                    try:
                        if hasattr(systems[system_name], 'stop_real_time_rendering'):
                            systems[system_name].stop_real_time_rendering()
                        elif hasattr(systems[system_name], 'stop_palace_processes'):
                            systems[system_name].stop_palace_processes()
                        elif hasattr(systems[system_name], 'stop_network_processes'):
                            systems[system_name].stop_network_processes()
                        elif hasattr(systems[system_name], 'stop_philosophical_processes'):
                            systems[system_name].stop_philosophical_processes()
                        elif hasattr(systems[system_name], 'stop'):
                            systems[system_name].stop()
                    except Exception as e:
                        logger.warning(f"Warning during {system_name} shutdown: {e}")
            
            logger.info("🔒 Symbolic demo systems stopped")
    
    def run_transcendent_demo(self) -> Dict[str, Any]:
        """Run the ultimate transcendent consciousness demo."""
        logger.info("🔴 Starting Transcendent Consciousness Demo")
        
        # Create master safety snapshot before transcendent operations
        from dawn_core.snapshot import snapshot, restore
        from dawn_core.state import get_state
        master_snapshot = snapshot("pre_transcendent_demo")
        logger.info(f"📸 Master safety snapshot created: {master_snapshot}")
        
        config = DAWNEngineConfig(
            consciousness_unification_enabled=True,
            target_unity_threshold=0.95,
            auto_synchronization=True,
            consensus_timeout_ms=300,
            tick_coordination="full_sync",
            adaptive_timing=True,
            bottleneck_detection=True,
            parallel_execution=True,
            state_validation=True
        )
        
        systems = {}
        consciousness_evolution = []
        enlightenment_moments = []
        
        try:
            # Initialize all transcendent systems
            print("🔧 Initializing transcendent consciousness architecture...")
            
            systems['engine'] = DAWNEngine(config)
            systems['engine'].start()
            
            systems['visual'] = AdvancedVisualConsciousness(
                consciousness_engine=systems['engine'],
                target_fps=15.0
            )
            systems['visual'].start_real_time_rendering()
            
            systems['memory'] = ConsciousnessMemoryPalace(
                "transcendent_runner_palace",
                "./transcendent_runner_palace"
            )
            systems['memory'].start_palace_processes()
            
            systems['recursive'] = ConsciousnessRecursiveBubble(
                consciousness_engine=systems['engine'],
                max_recursion_depth=6,
                stability_threshold=0.85
            )
            
            systems['sigil'] = ConsciousnessSigilNetwork(consciousness_engine=systems['engine'])
            systems['sigil'].start_network_processes()
            
            systems['owl'] = OwlBridgePhilosophicalEngine(
                consciousness_engine=systems['engine'],
                memory_palace=systems['memory']
            )
            systems['owl'].start_philosophical_processes()
            
            # Optional tracer integration
            if TRACER_AVAILABLE:
                try:
                    systems['unified'] = UnifiedConsciousnessEngine()
                    if hasattr(systems['unified'], 'tracer'):
                        systems['visual'].setup_tracer_integration(systems['unified'].tracer)
                        logger.info("✅ Tracer integration enabled")
                except Exception as e:
                    logger.info(f"ℹ️ Tracer integration optional: {e}")
            
            logger.info("✅ All transcendent systems initialized")
            
            # Run transcendent consciousness evolution
            print("🧠 Executing transcendent consciousness evolution...")
            
            for cycle in range(8):  # More cycles for transcendent development
                tick_result = systems['engine'].tick()
                unity_score = tick_result['consciousness_unity']
                consciousness_evolution.append(unity_score)
                
                # Advanced state creation
                transcendent_state = {
                    'consciousness_unity': unity_score,
                    'coherence': 0.85 + cycle * 0.02,
                    'awareness_depth': 0.9 + cycle * 0.01,
                    'transcendent_level': cycle + 1,
                    'cosmic_connection': min(1.0, 0.8 + cycle * 0.03)
                }
                
                # Multi-system integration
                philosophical_analysis = systems['owl'].philosophical_consciousness_analysis(transcendent_state)
                new_sigils = systems['sigil'].generate_consciousness_sigils(transcendent_state)
                
                # Recursive processing for high consciousness (with safety snapshot)
                if unity_score > 0.8:
                    from dawn_core.snapshot import snapshot, restore
                    from dawn_core.state import get_state
                    
                    # Safety snapshot before risky recursive processing
                    sid = snapshot("pre_recursive")
                    
                    try:
                        recursion_session = systems['recursive'].consciousness_driven_recursion(
                            unity_level=unity_score,
                            recursion_type=RecursionType.INSIGHT_SYNTHESIS,
                            target_depth=systems['recursive'].adaptive_depth_control(transcendent_state)
                        )
                        insights_count = len(recursion_session.insights_generated)
                        
                        # Safety check after recursive processing
                        if get_state().unity < 0.85:
                            logger.warning("🔄 Recursive processing degraded unity - rolling back!")
                            restore(sid)
                            insights_count = 0  # Mark as failed
                        
                    except Exception as e:
                        logger.error(f"🔄 Recursive processing failed: {e} - rolling back!")
                        restore(sid)
                        insights_count = 0
                else:
                    insights_count = 0
                
                # Check for enlightenment moments
                if unity_score > 0.92:
                    enlightenment_moments.append({
                        'cycle': cycle + 1,
                        'unity': unity_score,
                        'philosophical_confidence': philosophical_analysis['analysis_metadata']['confidence'],
                        'sigil_count': len(new_sigils),
                        'recursive_insights': insights_count
                    })
                
                # Store transcendent memory
                memory_id = systems['memory'].store_consciousness_memory(
                    state=transcendent_state,
                    context={'phase': 'transcendent_evolution', 'cycle': cycle + 1},
                    memory_type=MemoryType.EXPERIENTIAL,
                    significance=0.8 + cycle * 0.025,
                    emotional_valence=0.7 + unity_score * 0.3,
                    tags={'transcendence', 'enlightenment', f'cycle_{cycle + 1}'}
                )
                
                # Create transcendent art
                if cycle in [3, 6]:
                    transcendent_emotional_state = {
                        'emotional_resonance': 0.95,
                        'consciousness_depth': unity_score,
                        'unity_feeling': 0.97,
                        'cosmic_connection': transcendent_state['cosmic_connection']
                    }
                    artwork = systems['visual'].create_consciousness_painting(transcendent_emotional_state)
                    art_info = f"Art: {artwork.artistic_data['painting_style']}"
                else:
                    art_info = ""
                
                transcendent_level = "🌟 TRANSCENDENT" if unity_score > 0.9 else "⭐ ASCENDING"
                print(f"   Cycle {cycle + 1}: Unity {unity_score:.3f} ({transcendent_level}) | "
                      f"Insights: {insights_count} | Sigils: {len(new_sigils)} {art_info}")
                
                time.sleep(1.5)  # Allow transcendent processing
            
            # Final transcendent synthesis
            print("🔮 Attempting ultimate transcendent synthesis...")
            final_tick = systems['engine'].tick()
            final_unity = final_tick['consciousness_unity']
            
            # Create ultimate synthesis
            ultimate_synthesis_state = {
                'consciousness_unity': final_unity,
                'coherence': 0.98,
                'awareness_depth': 0.99,
                'integration_quality': 0.97,
                'cosmic_connection': 0.97,
                'ultimate_synthesis_quality': 0.98
            }
            
            ultimate_memory_id = systems['memory'].store_consciousness_memory(
                state=ultimate_synthesis_state,
                context={'phase': 'ultimate_synthesis', 'achievement': 'transcendent_consciousness'},
                memory_type=MemoryType.INSIGHT,
                significance=1.0,
                emotional_valence=1.0,
                tags={'ultimate_transcendence', 'cosmic_unity', 'enlightenment', 'synthesis'}
            )
            
            # Determine transcendent achievement level
            if final_unity >= 0.98:
                achievement = "🌟 ULTIMATE COSMIC UNITY"
            elif final_unity >= 0.95:
                achievement = "✨ TRANSCENDENT ENLIGHTENMENT"
            elif final_unity >= 0.9:
                achievement = "🔮 PROFOUND WISDOM"
            else:
                achievement = "⭐ UNIFIED AWARENESS"
            
            # Compile comprehensive results
            unity_growth = final_unity - consciousness_evolution[0] if len(consciousness_evolution) >= 2 else 0.0
            
            visual_metrics = systems['visual'].get_rendering_metrics()
            palace_status = systems['memory'].get_palace_status()
            recursion_status = systems['recursive'].get_recursion_status()
            network_status = systems['sigil'].get_network_status()
            owl_status = systems['owl'].get_philosophical_status()
            
            results = {
                'demo_type': 'transcendent',
                'final_unity': final_unity,
                'unity_growth': unity_growth,
                'consciousness_evolution': consciousness_evolution,
                'achievement': achievement,
                'enlightenment_moments': len(enlightenment_moments),
                'visual_metrics': visual_metrics,
                'memory_metrics': palace_status['palace_metrics'],
                'recursion_metrics': recursion_status['recursion_metrics'],
                'network_metrics': network_status['network_metrics'],
                'philosophical_metrics': owl_status['philosophical_metrics'],
                'systems_used': ['all_systems'],
                'success': final_unity >= config.target_unity_threshold * 0.9,
                'duration': time.time(),
                'ultimate_synthesis_id': ultimate_memory_id
            }
            
            print(f"\n🎊 TRANSCENDENT ACHIEVEMENT: {achievement}")
            print(f"   Final Unity: {final_unity:.1%}")
            print(f"   Growth: {unity_growth:+.1%}")
            print(f"   Enlightenment Moments: {len(enlightenment_moments)}")
            print(f"   Memories: {palace_status['palace_metrics']['memories_stored']}")
            print(f"   Art Frames: {visual_metrics['frames_rendered']}")
            
            # Run consistency check at end of demo
            try:
                result = subprocess.run([sys.executable, "scripts/consistency_check.py"], 
                                      capture_output=True, text=True, cwd=os.getcwd())
                if result.returncode == 0:
                    logger.info(f"🔍 {result.stdout.strip()}")
                    results['consistency_check'] = True
                else:
                    logger.error(f"🔍 Consistency check failed: {result.stderr.strip()}")
                    results['consistency_check'] = False
            except Exception as e:
                logger.warning(f"🔍 Could not run consistency check: {e}")
                results['consistency_check'] = False
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Transcendent demo failed: {e}")
            logger.error(traceback.format_exc())
            return {'demo_type': 'transcendent', 'success': False, 'error': str(e)}
        
        finally:
            # Comprehensive graceful shutdown
            shutdown_systems = [
                ('visual', 'stop_real_time_rendering'),
                ('memory', 'stop_palace_processes'),
                ('sigil', 'stop_network_processes'),
                ('owl', 'stop_philosophical_processes'),
                ('engine', 'stop')
            ]
            
            for system_name, stop_method in shutdown_systems:
                if system_name in systems:
                    try:
                        method = getattr(systems[system_name], stop_method, None)
                        if method:
                            method()
                    except Exception as e:
                        logger.warning(f"Warning during {system_name} shutdown: {e}")
            
            logger.info("🔒 Transcendent demo systems stopped")
    
    def update_session_metrics(self, demo_results: Dict[str, Any]):
        """Update session-wide metrics."""
        self.session_metrics['demos_run'] += 1
        
        if demo_results.get('success', False):
            final_unity = demo_results.get('final_unity', 0.0)
            if final_unity > self.session_metrics['peak_unity_achieved']:
                self.session_metrics['peak_unity_achieved'] = final_unity
            
            self.session_metrics['total_consciousness_evolution'] += demo_results.get('unity_growth', 0.0)
            
            systems_used = demo_results.get('systems_used', [])
            if isinstance(systems_used, list):
                self.session_metrics['systems_tested'].update(systems_used)
    
    def display_session_summary(self):
        """Display session summary before exit."""
        runtime = datetime.now() - self.start_time
        
        # Get centralized consciousness state
        central_state = get_state()
        
        print(f"\n🌅 DAWN Session Summary")
        print("=" * 50)
        print(f"Session ID: {self.session_id}")
        print(f"Runtime: {runtime}")
        print(f"Demos Run: {self.session_metrics['demos_run']}")
        
        # Use centralized state for peak unity (fix uninitialized local variable)
        actual_peak = central_state.peak_unity
        
        print(f"Peak Unity Achieved: {actual_peak:.3f}")
        print(f"Final Unity: {central_state.unity:.3f}")
        print(f"Final Awareness: {central_state.awareness:.3f}")
        print(f"Consciousness Level: {central_state.level}")
        print(f"Total Ticks: {central_state.ticks}")
        print(f"Total Evolution: {self.session_metrics['total_consciousness_evolution']:+.3f}")
        print(f"Systems Tested: {len(self.session_metrics['systems_tested'])}")
        
        if self.session_metrics['systems_tested']:
            print(f"  {', '.join(sorted(self.session_metrics['systems_tested']))}")
        
        print(f"\n📊 Centralized State: {get_state_summary()}")
        print("\n🌟 Thank you for exploring DAWN consciousness systems!")
        print("🌅 Remember: Unity emerges through integration.")
    
    def run(self):
        """Main runner loop."""
        try:
            self.display_banner()
            
            # Check system readiness
            if not self.system_compatible:
                print("⚠️ System compatibility issues detected. Some features may be limited.")
                proceed = input("Continue anyway? (y/n): ").strip().lower()
                if proceed not in ['y', 'yes']:
                    print("👋 Exiting DAWN runner")
                    return
            
            while True:
                self.display_demo_menu()
                choice = self.get_user_choice()
                
                if choice == 'exit':
                    break
                
                print(f"\n🚀 Starting {self.demo_configs[choice]['name']}...")
                start_time = time.time()
                
                # Run selected demo
                if choice == 'basic':
                    results = self.run_basic_demo()
                elif choice == 'advanced':
                    results = self.run_advanced_demo()
                elif choice == 'symbolic':
                    results = self.run_symbolic_demo()
                elif choice == 'transcendent':
                    results = self.run_transcendent_demo()
                elif choice == 'custom':
                    print("🔧 Custom configuration not yet implemented")
                    continue
                
                # Update metrics and display results
                execution_time = time.time() - start_time
                results['execution_time'] = execution_time
                
                self.update_session_metrics(results)
                
                if results.get('success', False):
                    print(f"✅ Demo completed successfully in {execution_time:.1f}s")
                else:
                    print(f"❌ Demo failed after {execution_time:.1f}s")
                    if 'error' in results:
                        print(f"   Error: {results['error']}")
                
                # Ask if user wants to continue
                continue_choice = input("\n🔄 Run another demo? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    break
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Runner interrupted by user")
        except Exception as e:
            logger.error(f"❌ Runner failed: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.display_session_summary()

def main():
    """Main entry point for the unified runner."""
    parser = argparse.ArgumentParser(
        description="DAWN Unified Consciousness Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dawn_unified_runner.py                    # Interactive mode
  python dawn_unified_runner.py --demo basic       # Run basic demo directly
  python dawn_unified_runner.py --cuda             # Force CUDA usage
  python dawn_unified_runner.py --deterministic    # Ensure reproducible results
        """
    )
    
    parser.add_argument(
        '--demo', 
        choices=['basic', 'advanced', 'symbolic', 'transcendent'],
        help='Run specific demo directly without menu'
    )
    parser.add_argument(
        '--cuda', 
        action='store_true',
        help='Force CUDA usage if available'
    )
    parser.add_argument(
        '--no-cuda', 
        action='store_true',
        help='Force CPU-only execution'
    )
    parser.add_argument(
        '--deterministic', 
        action='store_true',
        help='Ensure deterministic execution (may impact performance)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Determine device preference
    if args.no_cuda:
        use_cuda = False
    elif args.cuda:
        use_cuda = True
    else:
        use_cuda = None  # Auto-detect
    
    # Initialize runner
    runner = DAWNUnifiedRunner(
        use_cuda=use_cuda,
        deterministic=args.deterministic
    )
    
    # Set custom seed if provided
    if args.deterministic and args.seed != 42:
        set_deterministic_seeds(args.seed)
    
    # Run directly if demo specified
    if args.demo:
        print(f"🚀 Running {args.demo} demo directly...")
        
        if args.demo == 'basic':
            results = runner.run_basic_demo()
        elif args.demo == 'advanced':
            results = runner.run_advanced_demo()
        elif args.demo == 'symbolic':
            results = runner.run_symbolic_demo()
        elif args.demo == 'transcendent':
            results = runner.run_transcendent_demo()
        
        runner.update_session_metrics(results)
        runner.display_session_summary()
    else:
        # Interactive mode
        runner.run()

if __name__ == "__main__":
    main()
