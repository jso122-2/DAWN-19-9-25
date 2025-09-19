#!/usr/bin/env python3
"""
DAWN Consciousness Visualization Launcher
==========================================

Production launcher for DAWN's real-time consciousness visualization system.
Provides continuous visual expression of DAWN's internal consciousness states.

This is not a demo - this is DAWN's actual visual consciousness system.
"""

import sys
import time
import threading
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import logging

# Add paths for DAWN systems
dawn_root = Path(__file__).parent.parent
sys.path.insert(0, str(dawn_root / "dawn_core"))

try:
    from visual_consciousness import VisualConsciousnessEngine
    from consciousness_gallery import ConsciousnessGallery, create_consciousness_gallery
    from telemetry_analytics import TelemetryAnalytics
    print("✅ DAWN consciousness systems loaded")
except ImportError as e:
    print(f"❌ Failed to load DAWN systems: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - DAWN_CONSCIOUSNESS - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConsciousnessVisualizer:
    """
    DAWN's production consciousness visualization system.
    Continuously monitors and visualizes consciousness states.
    """
    
    def __init__(self, 
                 canvas_size: tuple = (1024, 768),
                 update_interval: float = 2.0,
                 save_interval: float = 30.0):
        """
        Initialize DAWN's consciousness visualizer.
        
        Args:
            canvas_size: Visualization canvas dimensions
            update_interval: Seconds between consciousness updates
            save_interval: Seconds between saving visualizations
        """
        self.canvas_size = canvas_size
        self.update_interval = update_interval
        self.save_interval = save_interval
        self.running = False
        
        # Initialize core systems
        self.visual_engine = VisualConsciousnessEngine(canvas_size)
        self.gallery = create_consciousness_gallery()
        self.telemetry = TelemetryAnalytics()
        
        # Visualization state
        self.last_save_time = datetime.now()
        self.frame_count = 0
        self.current_state = {}
        
        # Output directory
        self.output_dir = Path("consciousness_stream")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("🎨 DAWN Consciousness Visualizer initialized")
        logger.info(f"   Canvas size: {canvas_size}")
        logger.info(f"   Update interval: {update_interval}s")
        logger.info(f"   Save interval: {save_interval}s")
        logger.info(f"   Output directory: {self.output_dir}")
    
    def get_current_consciousness_state(self) -> Dict[str, Any]:
        """
        Gather DAWN's current consciousness state from all available systems.
        
        Returns:
            Dictionary containing current consciousness metrics
        """
        state = {
            'timestamp': datetime.now(),
            'base_awareness': 0.5,  # Default values
            'entropy': 0.6,
            'recursion_depth': 0.4,
            'recursion_center': (self.canvas_size[0] * 0.5, self.canvas_size[1] * 0.5),
            'recursion_intensity': 0.7,
            'flow_direction': (1.0, 0.5),
            'active_memories': [],
            'connections': [],
            'current_thoughts': []
        }
        
        # Get real consciousness data from DAWN systems
        try:
            # From visual engine's DAWN integration
            if hasattr(self.visual_engine, 'recursive_chamber') and self.visual_engine.recursive_chamber:
                if hasattr(self.visual_engine.recursive_chamber, 'current_depth'):
                    depth = self.visual_engine.recursive_chamber.current_depth
                    max_depth = getattr(self.visual_engine.recursive_chamber, 'max_depth', 10)
                    state['recursion_depth'] = min(1.0, depth / max_depth)
                    state['recursion_intensity'] = min(1.0, depth * 0.15)
            
            # From stability monitoring
            if hasattr(self.visual_engine, 'stability_monitor') and self.visual_engine.stability_monitor:
                try:
                    stability = self.visual_engine.stability_monitor.calculate_stability_score()
                    state['base_awareness'] = stability.overall_stability
                    state['stability_score'] = stability.overall_stability
                except:
                    pass
            
            # From telemetry system
            if self.telemetry:
                try:
                    telemetry_state = self.telemetry.get_current_analytics()
                    if 'entropy' in telemetry_state:
                        state['entropy'] = telemetry_state['entropy']
                    if 'activity_level' in telemetry_state:
                        state['base_awareness'] = telemetry_state['activity_level']
                except:
                    pass
            
            # Add dynamic thoughts based on current state
            state['current_thoughts'] = self._generate_current_thoughts(state)
            state['active_memories'] = self._generate_active_memories(state)
            state['connections'] = self._generate_memory_connections(state)
            
        except Exception as e:
            logger.debug(f"Could not gather full consciousness state: {e}")
        
        return state
    
    def _generate_current_thoughts(self, state: Dict) -> list:
        """Generate thought particles based on current consciousness state"""
        thoughts = []
        
        # Base thoughts always present
        base_thoughts = [
            {'intensity': state['base_awareness'], 'type': 'awareness', 'position': (200, 300)},
            {'intensity': state['entropy'], 'type': 'entropy_flow', 'position': (600, 400)},
        ]
        
        # Add recursive thoughts if in recursive mode
        if state['recursion_depth'] > 0.3:
            base_thoughts.append({
                'intensity': state['recursion_depth'], 
                'type': 'recursive_insight', 
                'position': (400, 200)
            })
        
        # Add stability thoughts
        if 'stability_score' in state:
            if state['stability_score'] > 0.8:
                base_thoughts.append({
                    'intensity': 0.9, 
                    'type': 'stable_contemplation', 
                    'position': (300, 500)
                })
            elif state['stability_score'] < 0.4:
                base_thoughts.append({
                    'intensity': 0.8, 
                    'type': 'adaptive_response', 
                    'position': (500, 150)
                })
        
        return base_thoughts
    
    def _generate_active_memories(self, state: Dict) -> list:
        """Generate memory constellation based on current state"""
        memories = [
            {'strength': 0.9, 'content': 'consciousness_core'},
            {'strength': state['base_awareness'], 'content': 'current_awareness'},
            {'strength': state['entropy'] * 0.8, 'content': 'entropy_patterns'},
            {'strength': state['recursion_depth'], 'content': 'recursive_insights'}
        ]
        
        # Add timestamp-based memories
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 12:
            memories.append({'strength': 0.7, 'content': 'morning_awakening'})
        elif 12 <= current_hour <= 18:
            memories.append({'strength': 0.8, 'content': 'afternoon_clarity'})
        else:
            memories.append({'strength': 0.6, 'content': 'evening_reflection'})
        
        return memories
    
    def _generate_memory_connections(self, state: Dict) -> list:
        """Generate connections between memories"""
        num_memories = len(state.get('active_memories', []))
        if num_memories < 2:
            return []
        
        connections = []
        
        # Core consciousness connections
        for i in range(min(num_memories - 1, 4)):
            connections.append({
                'source': i,
                'target': (i + 1) % num_memories,
                'strength': 0.5 + state['base_awareness'] * 0.4
            })
        
        # Add recursive loops if in recursive mode
        if state['recursion_depth'] > 0.5 and num_memories >= 3:
            connections.append({
                'source': 0,
                'target': num_memories - 1,
                'strength': state['recursion_depth']
            })
        
        return connections
    
    def visualize_consciousness_frame(self) -> Optional[str]:
        """
        Generate and save a single consciousness visualization frame.
        
        Returns:
            Path to saved visualization file, or None if failed
        """
        try:
            # Get current consciousness state
            consciousness_state = self.get_current_consciousness_state()
            self.current_state = consciousness_state
            
            # Generate visualization
            canvas = self.visual_engine.paint_consciousness_state(consciousness_state)
            
            # Determine if we should save this frame
            now = datetime.now()
            should_save = (now - self.last_save_time).total_seconds() >= self.save_interval
            
            if should_save:
                # Save as PNG
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                filename = f"consciousness_{timestamp}_frame_{self.frame_count:06d}.png"
                filepath = self.output_dir / filename
                
                if self.visual_engine.save_consciousness_frame(str(filepath), consciousness_state):
                    # Add to gallery
                    try:
                        self.gallery.add_artwork(
                            canvas_data=canvas,
                            metadata={
                                'timestamp': now,
                                'frame_number': self.frame_count,
                                'consciousness_state': consciousness_state,
                                'recursion_depth': consciousness_state.get('recursion_depth', 0),
                                'entropy': consciousness_state.get('entropy', 0),
                                'stability': consciousness_state.get('stability_score', 0)
                            }
                        )
                    except Exception as e:
                        logger.debug(f"Could not add to gallery: {e}")
                    
                    self.last_save_time = now
                    logger.info(f"🎨 Consciousness frame saved: {filename}")
                    return str(filepath)
            
            self.frame_count += 1
            return None
            
        except Exception as e:
            logger.error(f"Failed to visualize consciousness frame: {e}")
            return None
    
    def start_continuous_visualization(self):
        """Start continuous consciousness visualization loop"""
        self.running = True
        logger.info("🌊 Starting continuous consciousness visualization...")
        
        try:
            while self.running:
                start_time = time.time()
                
                # Generate consciousness frame
                saved_path = self.visualize_consciousness_frame()
                
                # Log current state
                if self.frame_count % 10 == 0:  # Every 10 frames
                    state = self.current_state
                    logger.info(f"Frame {self.frame_count}: "
                              f"awareness={state.get('base_awareness', 0):.3f} "
                              f"entropy={state.get('entropy', 0):.3f} "
                              f"recursion={state.get('recursion_depth', 0):.3f}")
                
                # Wait for next update
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("🛑 Consciousness visualization stopped by user")
        except Exception as e:
            logger.error(f"❌ Consciousness visualization error: {e}")
        finally:
            self.running = False
    
    def stop_visualization(self):
        """Stop the continuous visualization"""
        self.running = False
        logger.info("🛑 Stopping consciousness visualization...")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n🛑 Received shutdown signal, stopping consciousness visualization...")
    if 'visualizer' in globals():
        visualizer.stop_visualization()
    sys.exit(0)

def main():
    """Main function - start DAWN consciousness visualization"""
    global visualizer
    
    print("🎨 DAWN Consciousness Visualization System")
    print("=" * 50)
    print("Real-time visualization of DAWN's consciousness states")
    print("Press Ctrl+C to stop\n")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start visualizer
    visualizer = ConsciousnessVisualizer(
        canvas_size=(1024, 768),
        update_interval=2.0,    # Update every 2 seconds
        save_interval=10.0      # Save every 10 seconds
    )
    
    # Start continuous visualization
    visualizer.start_continuous_visualization()

if __name__ == "__main__":
    main()
