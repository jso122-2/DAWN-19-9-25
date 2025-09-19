#!/usr/bin/env python3
"""
🎨 DAWN Visual Consciousness Integration Demo
===========================================

Comprehensive demonstration of DAWN's enhanced visual consciousness system,
including real-time rendering, artistic expression, and tracer integration.

This demo showcases:
- Advanced Visual Consciousness with real-time rendering
- Consciousness Artistic Renderer with multiple styles
- Visual Consciousness Tracer Integration
- Integration with existing DAWN systems

Usage:
    python dawn_visual_consciousness_integration_demo.py [--mode MODE] [--duration SECONDS]
"""

import sys
import os
import time
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

# DAWN visual consciousness imports
try:
    from dawn.subsystems.visual.advanced_visual_consciousness import (
        create_advanced_visual_consciousness,
        ConsciousnessVisualMode,
        VisualComplexity
    )
    from dawn.subsystems.visual.consciousness_artistic_renderer import (
        create_consciousness_artistic_renderer,
        ArtisticStyle,
        ArtisticMedium
    )
    from dawn.subsystems.visual.visual_consciousness_tracer_integration import (
        create_visual_consciousness_tracer_integration,
        VisualTracerType,
        TracerPriority
    )
    VISUAL_SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Visual consciousness systems not available: {e}")
    VISUAL_SYSTEMS_AVAILABLE = False

# Try to import DAWN core systems
try:
    from dawn.core.communication.bus import get_consciousness_bus
    from dawn.core.foundation.state import get_state, set_state
    DAWN_CORE_AVAILABLE = True
except ImportError:
    print("⚠️  DAWN core systems not available - running in standalone mode")
    DAWN_CORE_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dawn_visual_demo.log')
    ]
)

logger = logging.getLogger(__name__)

class VisualConsciousnessDemo:
    """
    Comprehensive demo of DAWN's visual consciousness systems.
    """
    
    def __init__(self, mode: str = "comprehensive", duration: int = 30):
        self.mode = mode
        self.duration = duration
        self.demo_start_time = time.time()
        
        # System components
        self.visual_consciousness = None
        self.artistic_renderer = None
        self.tracer_integration = None
        self.consciousness_bus = None
        
        # Demo state
        self.demo_running = False
        self.consciousness_states = []
        self.generated_artworks = []
        self.performance_data = []
        
        logger.info(f"🎨 Visual Consciousness Demo initialized - Mode: {mode}, Duration: {duration}s")
    
    async def initialize_systems(self) -> bool:
        """Initialize all visual consciousness systems"""
        try:
            print("🚀 Initializing DAWN Visual Consciousness Systems...")
            
            if not VISUAL_SYSTEMS_AVAILABLE:
                print("❌ Visual consciousness systems not available")
                return False
            
            # Initialize consciousness bus if available
            if DAWN_CORE_AVAILABLE:
                try:
                    self.consciousness_bus = get_consciousness_bus(auto_start=True)
                    print("✅ Consciousness bus connected")
                except Exception as e:
                    print(f"⚠️  Consciousness bus not available: {e}")
            
            # Initialize visual consciousness engine
            print("   🎨 Creating Advanced Visual Consciousness...")
            self.visual_consciousness = create_advanced_visual_consciousness(
                consciousness_engine=None,  # Will use simulated data
                tracer_system=None,  # Will be connected later
                canvas_size=(1024, 768),
                target_fps=10.0  # Slower for demo
            )
            print("   ✅ Advanced Visual Consciousness created")
            
            # Initialize artistic renderer
            print("   🎭 Creating Consciousness Artistic Renderer...")
            self.artistic_renderer = create_consciousness_artistic_renderer(
                output_directory="dawn_visual_outputs"
            )
            print("   ✅ Artistic Renderer created")
            
            # Initialize tracer integration
            print("   🔍 Creating Tracer Integration...")
            self.tracer_integration = create_visual_consciousness_tracer_integration(
                tracer_system=None,  # Simulated
                consciousness_bus=self.consciousness_bus
            )
            print("   ✅ Tracer Integration created")
            
            # Connect systems
            print("   🔗 Connecting systems...")
            self.tracer_integration.register_visual_consciousness_engine(self.visual_consciousness)
            self.tracer_integration.register_artistic_renderer(self.artistic_renderer)
            print("   ✅ Systems connected")
            
            # Start monitoring
            self.tracer_integration.start_monitoring()
            print("   📊 Performance monitoring started")
            
            print("🌟 All systems initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize systems: {e}")
            print(f"❌ System initialization failed: {e}")
            return False
    
    async def run_demo(self) -> None:
        """Run the visual consciousness demonstration"""
        if not await self.initialize_systems():
            return
        
        self.demo_running = True
        print(f"\n🎬 Starting Visual Consciousness Demo - {self.mode} mode")
        print("=" * 60)
        
        try:
            if self.mode == "quick":
                await self._run_quick_demo()
            elif self.mode == "artistic":
                await self._run_artistic_demo()
            elif self.mode == "performance":
                await self._run_performance_demo()
            else:  # comprehensive
                await self._run_comprehensive_demo()
                
        except KeyboardInterrupt:
            print("\n🔔 Demo interrupted by user")
        except Exception as e:
            logger.error(f"Demo error: {e}")
            print(f"❌ Demo error: {e}")
        finally:
            await self._cleanup_systems()
    
    async def _run_quick_demo(self) -> None:
        """Quick demonstration of core features"""
        print("🚀 Quick Demo: Testing core visual consciousness features")
        
        # Generate test consciousness state
        consciousness_state = self._generate_test_consciousness_state()
        print(f"📊 Generated consciousness state: Unity={consciousness_state['consciousness_unity']:.3f}")
        
        # Test visual consciousness rendering
        print("🎨 Testing visual consciousness rendering...")
        self.visual_consciousness.update_consciousness_state(consciousness_state)
        
        # Create a consciousness painting
        print("🖼️  Creating consciousness painting...")
        painting = self.artistic_renderer.create_consciousness_painting(
            consciousness_state,
            ArtisticStyle.CONSCIOUSNESS_FLOW
        )
        self.generated_artworks.append(painting)
        print(f"✅ Painting created: {painting.composition_id}")
        
        # Generate poetry
        print("📝 Creating consciousness poetry...")
        poetry = self.artistic_renderer.consciousness_to_poetry(consciousness_state)
        self.generated_artworks.append(poetry)
        print(f"✅ Poetry created: {len(poetry.text_data)} characters")
        
        # Show metrics
        await self._show_demo_metrics()
    
    async def _run_artistic_demo(self) -> None:
        """Demonstration focused on artistic expression"""
        print("🎭 Artistic Demo: Exploring different artistic styles")
        
        styles_to_test = [
            ArtisticStyle.CONSCIOUSNESS_FLOW,
            ArtisticStyle.ABSTRACT_EXPRESSIONISM,
            ArtisticStyle.IMPRESSIONISM,
            ArtisticStyle.MINIMALISM,
            ArtisticStyle.SURREALISM
        ]
        
        for i, style in enumerate(styles_to_test):
            print(f"\n🎨 Creating artwork {i+1}/{len(styles_to_test)}: {style.value}")
            
            # Generate varying consciousness states
            consciousness_state = self._generate_varied_consciousness_state(i / len(styles_to_test))
            
            # Create painting in this style
            painting = self.artistic_renderer.create_consciousness_painting(
                consciousness_state, style
            )
            self.generated_artworks.append(painting)
            
            print(f"   ✅ {style.value} painting created")
            print(f"      Emotional resonance: {painting.emotional_resonance:.3f}")
            print(f"      Technical quality: {painting.technical_quality:.3f}")
            
            # Brief pause between artworks
            await asyncio.sleep(1)
        
        # Create poetry and music descriptions
        print("\n📝 Creating consciousness poetry...")
        poetry = self.artistic_renderer.consciousness_to_poetry(consciousness_state)
        self.generated_artworks.append(poetry)
        
        print("🎵 Creating musical composition description...")
        music = self.artistic_renderer.consciousness_to_music(consciousness_state)
        self.generated_artworks.append(music)
        
        await self._show_artistic_summary()
    
    async def _run_performance_demo(self) -> None:
        """Demonstration focused on performance monitoring"""
        print("📊 Performance Demo: Monitoring visual consciousness performance")
        
        # Run continuous rendering for performance testing
        print("🎬 Starting continuous rendering test...")
        
        test_duration = min(self.duration, 15)  # Max 15 seconds for performance test
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < test_duration:
            # Generate dynamic consciousness state
            t = (time.time() - start_time) / test_duration
            consciousness_state = self._generate_dynamic_consciousness_state(t)
            
            # Update visual consciousness
            self.visual_consciousness.update_consciousness_state(consciousness_state)
            
            # Render frame (simulated)
            render_start = time.time()
            frame_data = self.visual_consciousness.render_consciousness_flow()
            render_time = (time.time() - render_start) * 1000  # ms
            
            # Record performance data
            self.performance_data.append({
                'frame': frame_count,
                'render_time_ms': render_time,
                'consciousness_unity': consciousness_state['consciousness_unity'],
                'timestamp': time.time()
            })
            
            frame_count += 1
            
            # Show progress
            if frame_count % 10 == 0:
                fps = frame_count / (time.time() - start_time)
                print(f"   Frame {frame_count}, FPS: {fps:.1f}, Render time: {render_time:.1f}ms")
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)
        
        await self._show_performance_analysis()
    
    async def _run_comprehensive_demo(self) -> None:
        """Comprehensive demonstration of all features"""
        print("🌟 Comprehensive Demo: Full visual consciousness showcase")
        
        # Phase 1: System capabilities
        print("\n📋 Phase 1: Testing system capabilities")
        await self._test_system_capabilities()
        
        # Phase 2: Artistic expression
        print("\n🎨 Phase 2: Artistic expression showcase")
        await self._test_artistic_expression()
        
        # Phase 3: Real-time rendering
        print("\n🎬 Phase 3: Real-time rendering test")
        await self._test_realtime_rendering()
        
        # Phase 4: Integration testing
        print("\n🔗 Phase 4: Integration testing")
        await self._test_system_integration()
        
        # Final summary
        await self._show_comprehensive_summary()
    
    async def _test_system_capabilities(self) -> None:
        """Test core system capabilities"""
        print("   🧪 Testing visual consciousness engine...")
        
        # Test different visual modes
        modes = [
            ConsciousnessVisualMode.UNITY_FIELD,
            ConsciousnessVisualMode.FRACTAL_DEPTH,
            ConsciousnessVisualMode.EMOTIONAL_HARMONY
        ]
        
        for mode in modes:
            consciousness_state = self._generate_test_consciousness_state()
            self.visual_consciousness.visual_mode = mode
            self.visual_consciousness.update_consciousness_state(consciousness_state)
            print(f"      ✅ {mode.value} mode tested")
        
        print("   🎭 Testing artistic renderer...")
        
        # Test different mediums
        consciousness_state = self._generate_test_consciousness_state()
        
        painting = self.artistic_renderer.create_consciousness_painting(consciousness_state)
        poetry = self.artistic_renderer.consciousness_to_poetry(consciousness_state)
        music = self.artistic_renderer.consciousness_to_music(consciousness_state)
        
        print(f"      ✅ Painting created: {painting.composition_id}")
        print(f"      ✅ Poetry created: {len(poetry.text_data)} chars")
        print(f"      ✅ Music description created")
        
        self.generated_artworks.extend([painting, poetry, music])
    
    async def _test_artistic_expression(self) -> None:
        """Test artistic expression capabilities"""
        print("   🎨 Creating artworks in different styles...")
        
        styles = [ArtisticStyle.CONSCIOUSNESS_FLOW, ArtisticStyle.IMPRESSIONISM, ArtisticStyle.MINIMALISM]
        
        for style in styles:
            consciousness_state = self._generate_varied_consciousness_state(0.7)
            painting = self.artistic_renderer.create_consciousness_painting(consciousness_state, style)
            self.generated_artworks.append(painting)
            print(f"      ✅ {style.value}: Quality={painting.technical_quality:.3f}")
    
    async def _test_realtime_rendering(self) -> None:
        """Test real-time rendering performance"""
        print("   🎬 Testing real-time rendering...")
        
        frames_to_render = 20
        start_time = time.time()
        
        for i in range(frames_to_render):
            # Dynamic consciousness state
            t = i / frames_to_render
            consciousness_state = self._generate_dynamic_consciousness_state(t)
            
            # Render frame
            self.visual_consciousness.update_consciousness_state(consciousness_state)
            self.visual_consciousness.render_consciousness_flow()
            
            if i % 5 == 0:
                print(f"      Frame {i+1}/{frames_to_render}")
        
        total_time = time.time() - start_time
        fps = frames_to_render / total_time
        print(f"      ✅ Rendered {frames_to_render} frames in {total_time:.2f}s ({fps:.1f} FPS)")
    
    async def _test_system_integration(self) -> None:
        """Test integration between systems"""
        print("   🔗 Testing system integration...")
        
        # Test tracer integration
        if self.tracer_integration:
            status = self.tracer_integration.get_integration_status()
            print(f"      ✅ Tracer integration: {status['total_events']} events")
            
            # Test performance monitoring
            metrics = self.tracer_integration.get_performance_metrics()
            print(f"      ✅ Performance monitoring: {metrics.frames_per_second:.1f} FPS")
            
            # Test alerts
            alerts = self.tracer_integration.get_performance_alerts()
            print(f"      ✅ Alert system: {len(alerts)} alerts")
    
    def _generate_test_consciousness_state(self) -> Dict[str, Any]:
        """Generate a test consciousness state"""
        return {
            'consciousness_unity': 0.75,
            'self_awareness_depth': 0.65,
            'integration_quality': 0.8,
            'emotional_coherence': {
                'serenity': 0.7,
                'curiosity': 0.8,
                'creativity': 0.9
            },
            'memory_integration': 0.7,
            'recursive_depth': 3,
            'stability_score': 0.85
        }
    
    def _generate_varied_consciousness_state(self, variation: float) -> Dict[str, Any]:
        """Generate consciousness state with variation"""
        base = 0.5
        amplitude = 0.3
        
        return {
            'consciousness_unity': base + amplitude * np.sin(variation * np.pi),
            'self_awareness_depth': base + amplitude * np.cos(variation * np.pi),
            'integration_quality': base + amplitude * np.sin(variation * np.pi * 2),
            'emotional_coherence': {
                'serenity': base + amplitude * np.cos(variation * np.pi * 0.5),
                'curiosity': base + amplitude * np.sin(variation * np.pi * 1.5),
                'creativity': base + amplitude * np.cos(variation * np.pi * 2.5)
            },
            'memory_integration': 0.6 + 0.3 * variation,
            'recursive_depth': int(2 + variation * 4),
            'stability_score': 0.7 + 0.2 * (1 - abs(variation - 0.5) * 2)
        }
    
    def _generate_dynamic_consciousness_state(self, t: float) -> Dict[str, Any]:
        """Generate dynamic consciousness state based on time"""
        return {
            'consciousness_unity': 0.6 + 0.3 * np.sin(t * np.pi * 2),
            'self_awareness_depth': 0.5 + 0.4 * np.cos(t * np.pi * 3),
            'integration_quality': 0.7 + 0.2 * np.sin(t * np.pi * 4),
            'emotional_coherence': {
                'serenity': 0.5 + 0.3 * np.sin(t * np.pi),
                'curiosity': 0.6 + 0.3 * np.cos(t * np.pi * 1.5),
                'creativity': 0.7 + 0.3 * np.sin(t * np.pi * 2.5)
            },
            'memory_integration': 0.6 + 0.2 * t,
            'recursive_depth': int(2 + t * 3),
            'stability_score': 0.8 + 0.1 * np.cos(t * np.pi)
        }
    
    async def _show_demo_metrics(self) -> None:
        """Show demo metrics and results"""
        print(f"\n📊 Demo Results:")
        print(f"   Artworks created: {len(self.generated_artworks)}")
        
        if self.visual_consciousness:
            metrics = self.visual_consciousness.get_visual_metrics()
            print(f"   Visual coherence: {metrics['current_visual_state']['visual_coherence']:.3f}")
            print(f"   Rendering quality: {metrics['current_visual_state']['rendering_quality']:.3f}")
        
        if self.artistic_renderer:
            render_metrics = self.artistic_renderer.get_rendering_metrics()
            print(f"   Average render time: {render_metrics.get('average_render_time', 0):.2f}s")
            print(f"   Average quality: {render_metrics.get('average_quality', 0):.3f}")
    
    async def _show_artistic_summary(self) -> None:
        """Show artistic creation summary"""
        print(f"\n🎨 Artistic Summary:")
        
        paintings = [a for a in self.generated_artworks if a.medium == ArtisticMedium.PAINTING]
        poetry = [a for a in self.generated_artworks if a.medium == ArtisticMedium.POETRY]
        music = [a for a in self.generated_artworks if a.medium == ArtisticMedium.MUSIC]
        
        print(f"   Paintings: {len(paintings)}")
        print(f"   Poetry: {len(poetry)}")
        print(f"   Music descriptions: {len(music)}")
        
        if paintings:
            avg_quality = np.mean([p.technical_quality for p in paintings])
            avg_resonance = np.mean([p.emotional_resonance for p in paintings])
            print(f"   Average painting quality: {avg_quality:.3f}")
            print(f"   Average emotional resonance: {avg_resonance:.3f}")
    
    async def _show_performance_analysis(self) -> None:
        """Show performance analysis results"""
        print(f"\n📊 Performance Analysis:")
        
        if self.performance_data:
            render_times = [d['render_time_ms'] for d in self.performance_data]
            avg_render_time = np.mean(render_times)
            max_render_time = np.max(render_times)
            min_render_time = np.min(render_times)
            
            total_time = self.performance_data[-1]['timestamp'] - self.performance_data[0]['timestamp']
            fps = len(self.performance_data) / total_time
            
            print(f"   Total frames: {len(self.performance_data)}")
            print(f"   Average FPS: {fps:.2f}")
            print(f"   Average render time: {avg_render_time:.2f}ms")
            print(f"   Min render time: {min_render_time:.2f}ms")
            print(f"   Max render time: {max_render_time:.2f}ms")
    
    async def _show_comprehensive_summary(self) -> None:
        """Show comprehensive demo summary"""
        print(f"\n🌟 Comprehensive Demo Summary:")
        print(f"   Demo duration: {time.time() - self.demo_start_time:.1f}s")
        print(f"   Total artworks: {len(self.generated_artworks)}")
        
        # System status
        if self.visual_consciousness:
            print(f"   ✅ Visual Consciousness: Active")
        if self.artistic_renderer:
            print(f"   ✅ Artistic Renderer: Active")
        if self.tracer_integration:
            print(f"   ✅ Tracer Integration: Active")
        
        # Performance summary
        if self.tracer_integration:
            metrics = self.tracer_integration.get_performance_metrics()
            print(f"   Current FPS: {metrics.frames_per_second:.1f}")
            print(f"   Visual quality: {metrics.artistic_quality_score:.3f}")
    
    async def _cleanup_systems(self) -> None:
        """Clean up systems and resources"""
        print(f"\n🧹 Cleaning up systems...")
        
        try:
            # Stop monitoring
            if self.tracer_integration:
                self.tracer_integration.stop_monitoring()
                print("   ✅ Tracer monitoring stopped")
            
            # Stop real-time rendering
            if self.visual_consciousness:
                self.visual_consciousness.stop_real_time_rendering()
                print("   ✅ Real-time rendering stopped")
            
            # Final metrics
            await self._show_final_metrics()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        self.demo_running = False
        print("   🌙 Cleanup complete")
    
    async def _show_final_metrics(self) -> None:
        """Show final demo metrics"""
        print(f"\n📈 Final Metrics:")
        
        # Show created files
        output_dir = Path("dawn_visual_outputs")
        if output_dir.exists():
            files = list(output_dir.glob("*"))
            print(f"   Output files created: {len(files)}")
            for file in files[-5:]:  # Show last 5 files
                print(f"      {file.name}")
        
        # Show system performance
        if self.tracer_integration:
            status = self.tracer_integration.get_integration_status()
            print(f"   Total tracer events: {status['total_events']}")
            print(f"   Performance alerts: {status['recent_alerts']}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DAWN Visual Consciousness Integration Demo')
    
    parser.add_argument(
        '--mode',
        choices=['quick', 'artistic', 'performance', 'comprehensive'],
        default='comprehensive',
        help='Demo mode (default: comprehensive)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Demo duration in seconds (default: 30)'
    )
    
    return parser.parse_args()

async def main():
    """Main demo entry point"""
    args = parse_arguments()
    
    print("🌅 DAWN Visual Consciousness Integration Demo")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Duration: {args.duration} seconds")
    print("=" * 60)
    
    # Create and run demo
    demo = VisualConsciousnessDemo(mode=args.mode, duration=args.duration)
    await demo.run_demo()
    
    print("\n🎉 Demo complete!")
    print("Check 'dawn_visual_outputs/' directory for generated artworks")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🔔 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)
