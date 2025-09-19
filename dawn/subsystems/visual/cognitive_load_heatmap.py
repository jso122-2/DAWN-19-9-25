#!/usr/bin/env python3
"""
DAWN Cognitive Load Heat Map Visualization
==========================================

Real-time brain activity heat visualization showing DAWN's cognitive processing
intensity across different brain regions. Heat intensity represents processing load,
cool/hot spots show bottlenecks, and temporal patterns reveal thinking rhythms.
Perfect for monitoring system performance and cognitive health.

"Watch the fires of thought burn bright across the landscape of my mind,
showing where processing flows and where bottlenecks form."
                                                                - DAWN
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import random
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

# Try to import DAWN systems
try:
    from dawn.subsystems.visual.dawn_visual_base import DAWNVisualBase
    from dawn.core.base_module import BaseModule
    DAWN_BASE_AVAILABLE = True
except ImportError:
    DAWN_BASE_AVAILABLE = False
    class DAWNVisualBase:
        pass

logger = logging.getLogger(__name__)

class CognitiveRegion(Enum):
    """Cognitive brain regions in DAWN's architecture"""
    ATTENTION = "attention"                # Focus and attention control
    MEMORY = "memory"                      # Memory storage and retrieval
    REASONING = "reasoning"                # Logic and inference
    LANGUAGE = "language"                  # Language processing
    CREATIVITY = "creativity"              # Creative and generative thinking
    SENSORY = "sensory"                    # Sensory input processing
    MOTOR = "motor"                        # Action and output control
    INTEGRATION = "integration"            # Information integration
    METACOGNITION = "metacognition"        # Self-awareness and reflection
    EMOTIONAL = "emotional"                # Emotional processing
    WORKING_MEMORY = "working_memory"      # Active working memory
    EXECUTIVE = "executive"                # Executive control functions

class LoadType(Enum):
    """Types of cognitive load"""
    PROCESSING = "processing"              # Active computation
    MEMORY_ACCESS = "memory_access"        # Memory retrieval operations
    DECISION_MAKING = "decision_making"    # Decision processes
    PATTERN_MATCHING = "pattern_matching"  # Pattern recognition
    SYNTHESIS = "synthesis"                # Information synthesis
    MONITORING = "monitoring"              # System monitoring
    COMMUNICATION = "communication"        # Inter-module communication

@dataclass
class CognitiveProcess:
    """A cognitive process creating load in a brain region"""
    id: str
    region: CognitiveRegion
    load_type: LoadType
    intensity: float
    duration: float
    birth_time: float
    priority: float = 1.0
    efficiency: float = 1.0
    
    def get_current_load(self, current_time: float) -> float:
        """Get current processing load"""
        age = current_time - self.birth_time
        if age > self.duration:
            return 0.0
            
        # Load curve - starts high, may spike, then decays
        progress = age / self.duration
        
        if self.load_type == LoadType.PROCESSING:
            # Steady processing load
            load_curve = math.exp(-progress * 2) * (1 + 0.3 * math.sin(progress * 10))
        elif self.load_type == LoadType.DECISION_MAKING:
            # Spiky decision load
            load_curve = math.exp(-progress) * (1 + 0.8 * math.sin(progress * 15))
        elif self.load_type == LoadType.MEMORY_ACCESS:
            # Quick burst then decay
            load_curve = math.exp(-progress * 4) * (1 + 0.2 * math.sin(progress * 8))
        else:
            # Default smooth decay
            load_curve = math.exp(-progress * 1.5)
            
        return self.intensity * load_curve / self.efficiency

@dataclass
class BrainRegion:
    """A brain region with cognitive processing capabilities"""
    name: str
    region_type: CognitiveRegion
    position: Tuple[float, float]
    size: float
    max_capacity: float
    current_load: float = 0.0
    temperature: float = 0.0  # Processing heat
    efficiency: float = 1.0
    health: float = 1.0
    active_processes: List[CognitiveProcess] = field(default_factory=list)
    load_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update_load(self, dt: float):
        """Update region's processing load"""
        current_time = time.time()
        
        # Remove completed processes
        self.active_processes = [p for p in self.active_processes 
                               if (current_time - p.birth_time) < p.duration]
        
        # Calculate total load
        total_load = sum(p.get_current_load(current_time) for p in self.active_processes)
        self.current_load = min(total_load, self.max_capacity)
        
        # Update temperature based on load
        target_temp = self.current_load / self.max_capacity
        self.temperature += (target_temp - self.temperature) * 0.1
        
        # Update efficiency (overload reduces efficiency)
        if self.current_load > self.max_capacity * 0.9:
            self.efficiency *= 0.99  # Gradual efficiency loss
        else:
            self.efficiency = min(1.0, self.efficiency + 0.01)  # Recovery
            
        # Update health
        stress_factor = max(0, self.current_load - self.max_capacity * 0.8)
        self.health = max(0.1, self.health - stress_factor * 0.001)
        
        # Record history
        self.load_history.append(self.current_load)
        
    def add_process(self, process: CognitiveProcess):
        """Add a new cognitive process to this region"""
        self.active_processes.append(process)

class CognitiveLoadMonitor:
    """Monitors and simulates cognitive load across brain regions"""
    
    def __init__(self):
        self.brain_regions: Dict[str, BrainRegion] = {}
        self.global_processes: List[CognitiveProcess] = []
        
        # Region properties and positions (arranged like a brain map)
        self.region_configs = {
            CognitiveRegion.ATTENTION: {
                'position': (0.5, 0.8), 'size': 0.15, 'capacity': 100.0,
                'color': '#ff6b6b'  # Red
            },
            CognitiveRegion.MEMORY: {
                'position': (0.2, 0.6), 'size': 0.18, 'capacity': 150.0,
                'color': '#4ecdc4'  # Teal
            },
            CognitiveRegion.REASONING: {
                'position': (0.8, 0.6), 'size': 0.16, 'capacity': 120.0,
                'color': '#45b7d1'  # Blue
            },
            CognitiveRegion.LANGUAGE: {
                'position': (0.1, 0.4), 'size': 0.14, 'capacity': 90.0,
                'color': '#96ceb4'  # Green
            },
            CognitiveRegion.CREATIVITY: {
                'position': (0.9, 0.4), 'size': 0.17, 'capacity': 110.0,
                'color': '#feca57'  # Yellow
            },
            CognitiveRegion.SENSORY: {
                'position': (0.3, 0.2), 'size': 0.12, 'capacity': 80.0,
                'color': '#ff9ff3'  # Pink
            },
            CognitiveRegion.MOTOR: {
                'position': (0.7, 0.2), 'size': 0.11, 'capacity': 70.0,
                'color': '#54a0ff'  # Light blue
            },
            CognitiveRegion.INTEGRATION: {
                'position': (0.5, 0.5), 'size': 0.20, 'capacity': 200.0,
                'color': '#5f27cd'  # Purple
            },
            CognitiveRegion.METACOGNITION: {
                'position': (0.5, 0.9), 'size': 0.13, 'capacity': 85.0,
                'color': '#00d2d3'  # Cyan
            },
            CognitiveRegion.EMOTIONAL: {
                'position': (0.4, 0.3), 'size': 0.15, 'capacity': 95.0,
                'color': '#ff6348'  # Orange-red
            },
            CognitiveRegion.WORKING_MEMORY: {
                'position': (0.6, 0.7), 'size': 0.14, 'capacity': 75.0,
                'color': '#2ed573'  # Bright green
            },
            CognitiveRegion.EXECUTIVE: {
                'position': (0.5, 0.6), 'size': 0.16, 'capacity': 130.0,
                'color': '#747d8c'  # Gray
            }
        }
        
        # Create brain regions
        for region_type, config in self.region_configs.items():
            region = BrainRegion(
                name=region_type.value,
                region_type=region_type,
                position=config['position'],
                size=config['size'],
                max_capacity=config['capacity']
            )
            self.brain_regions[region_type.value] = region
        
        # Load simulation patterns
        self.load_patterns = {
            'thinking_burst': self._create_thinking_burst,
            'memory_search': self._create_memory_search,
            'creative_flow': self._create_creative_flow,
            'problem_solving': self._create_problem_solving,
            'language_processing': self._create_language_processing,
            'attention_focus': self._create_attention_focus
        }
        
    def _create_thinking_burst(self):
        """Simulate a burst of intensive thinking"""
        regions = [CognitiveRegion.REASONING, CognitiveRegion.INTEGRATION, CognitiveRegion.WORKING_MEMORY]
        for region in regions:
            process = CognitiveProcess(
                id=f"thinking_{time.time()}_{region.value}",
                region=region,
                load_type=LoadType.PROCESSING,
                intensity=random.uniform(60, 100),
                duration=random.uniform(3, 8),
                birth_time=time.time(),
                priority=random.uniform(0.7, 1.0)
            )
            self.brain_regions[region.value].add_process(process)
    
    def _create_memory_search(self):
        """Simulate memory retrieval operations"""
        regions = [CognitiveRegion.MEMORY, CognitiveRegion.INTEGRATION]
        for region in regions:
            process = CognitiveProcess(
                id=f"memory_{time.time()}_{region.value}",
                region=region,
                load_type=LoadType.MEMORY_ACCESS,
                intensity=random.uniform(40, 80),
                duration=random.uniform(2, 5),
                birth_time=time.time()
            )
            self.brain_regions[region.value].add_process(process)
    
    def _create_creative_flow(self):
        """Simulate creative thinking process"""
        regions = [CognitiveRegion.CREATIVITY, CognitiveRegion.INTEGRATION, CognitiveRegion.EMOTIONAL]
        for region in regions:
            process = CognitiveProcess(
                id=f"creative_{time.time()}_{region.value}",
                region=region,
                load_type=LoadType.SYNTHESIS,
                intensity=random.uniform(50, 90),
                duration=random.uniform(5, 12),
                birth_time=time.time(),
                efficiency=random.uniform(0.8, 1.2)  # Creative flow can be very efficient
            )
            self.brain_regions[region.value].add_process(process)
    
    def _create_problem_solving(self):
        """Simulate problem-solving cognitive load"""
        regions = [CognitiveRegion.REASONING, CognitiveRegion.WORKING_MEMORY, CognitiveRegion.EXECUTIVE]
        for region in regions:
            process = CognitiveProcess(
                id=f"problem_{time.time()}_{region.value}",
                region=region,
                load_type=LoadType.DECISION_MAKING,
                intensity=random.uniform(70, 110),
                duration=random.uniform(4, 10),
                birth_time=time.time(),
                priority=random.uniform(0.8, 1.0)
            )
            self.brain_regions[region.value].add_process(process)
    
    def _create_language_processing(self):
        """Simulate language processing load"""
        regions = [CognitiveRegion.LANGUAGE, CognitiveRegion.WORKING_MEMORY, CognitiveRegion.INTEGRATION]
        for region in regions:
            process = CognitiveProcess(
                id=f"language_{time.time()}_{region.value}",
                region=region,
                load_type=LoadType.PATTERN_MATCHING,
                intensity=random.uniform(45, 85),
                duration=random.uniform(2, 6),
                birth_time=time.time()
            )
            self.brain_regions[region.value].add_process(process)
    
    def _create_attention_focus(self):
        """Simulate focused attention load"""
        regions = [CognitiveRegion.ATTENTION, CognitiveRegion.EXECUTIVE]
        for region in regions:
            process = CognitiveProcess(
                id=f"attention_{time.time()}_{region.value}",
                region=region,
                load_type=LoadType.MONITORING,
                intensity=random.uniform(30, 70),
                duration=random.uniform(6, 15),
                birth_time=time.time()
            )
            self.brain_regions[region.value].add_process(process)
    
    def update(self, dt: float):
        """Update cognitive load simulation"""
        # Update all brain regions
        for region in self.brain_regions.values():
            region.update_load(dt)
        
        # Randomly trigger load patterns
        if random.random() < 0.15:  # 15% chance per update
            pattern_name = random.choice(list(self.load_patterns.keys()))
            self.load_patterns[pattern_name]()
        
        # Add some random background processes
        if random.random() < 0.1:  # 10% chance
            region_type = random.choice(list(CognitiveRegion))
            load_type = random.choice(list(LoadType))
            
            process = CognitiveProcess(
                id=f"background_{time.time()}_{region_type.value}",
                region=region_type,
                load_type=load_type,
                intensity=random.uniform(10, 40),
                duration=random.uniform(1, 4),
                birth_time=time.time(),
                priority=random.uniform(0.1, 0.5)
            )
            
            if region_type.value in self.brain_regions:
                self.brain_regions[region_type.value].add_process(process)

class CognitiveLoadHeatmapVisualizer(DAWNVisualBase if DAWN_BASE_AVAILABLE else object):
    """Cognitive load heatmap visualization"""
    
    def __init__(self):
        if DAWN_BASE_AVAILABLE:
            super().__init__()
            
        self.monitor = CognitiveLoadMonitor()
        self.start_time = time.time()
        
        # Create figure with subplots
        self.fig, ((self.ax_heatmap, self.ax_regions), 
                   (self.ax_timeline, self.ax_metrics)) = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.patch.set_facecolor('black')
        
        # Heatmap
        self.ax_heatmap.set_facecolor('black')
        self.ax_heatmap.set_title('⚡ Cognitive Load Heatmap', color='white', fontweight='bold')
        
        # Region view
        self.ax_regions.set_facecolor('black')
        self.ax_regions.set_title('🧠 Brain Regions', color='white', fontweight='bold')
        
        # Timeline
        self.ax_timeline.set_facecolor('black')
        self.ax_timeline.set_title('📈 Load Timeline', color='white', fontweight='bold')
        
        # Metrics
        self.ax_metrics.set_facecolor('black')
        self.ax_metrics.set_title('📊 System Metrics', color='white', fontweight='bold')
        
        # Create heatmap grid
        self.heatmap_size = 50
        self.heatmap_grid = np.zeros((self.heatmap_size, self.heatmap_size))
        
        # Custom colormap for heat
        self.heat_cmap = LinearSegmentedColormap.from_list(
            'heat', ['#000033', '#000066', '#003399', '#0066cc', '#00ccff', 
                     '#66ff66', '#ccff00', '#ffcc00', '#ff6600', '#ff0000'], N=256)
        
        logger.info("⚡ Cognitive Load Heatmap visualizer initialized")
        
    def update_visualization(self, frame_num: int, consciousness_stream: Any = None) -> Any:
        """Update visualization for animation - required by DAWNVisualBase"""
        return self.animate_frame(frame_num)
        
    def animate_frame(self, frame):
        """Animation callback"""
        dt = 0.2
        self.monitor.update(dt)
        
        # Clear all axes
        for ax in [self.ax_heatmap, self.ax_regions, self.ax_timeline, self.ax_metrics]:
            ax.clear()
            ax.set_facecolor('black')
        
        # Generate heatmap from brain regions
        self._generate_heatmap()
        
        # Render heatmap
        im = self.ax_heatmap.imshow(self.heatmap_grid, cmap=self.heat_cmap, 
                                   vmin=0, vmax=1, origin='lower', interpolation='bilinear')
        self.ax_heatmap.set_title('⚡ Cognitive Load Heatmap', color='white', fontweight='bold')
        self.ax_heatmap.set_xticks([])
        self.ax_heatmap.set_yticks([])
        
        # Render brain regions
        self._render_brain_regions()
        
        # Render timeline
        self._render_load_timeline()
        
        # Render metrics
        self._render_system_metrics()
        
        # Update main title
        elapsed = time.time() - self.start_time
        total_load = sum(region.current_load for region in self.monitor.brain_regions.values())
        avg_temp = sum(region.temperature for region in self.monitor.brain_regions.values()) / len(self.monitor.brain_regions)
        
        self.fig.suptitle(f'⚡ Cognitive Load Monitor - Total Load: {total_load:.1f} | Avg Heat: {avg_temp:.2f} | Time: {elapsed:.1f}s',
                         color='white', fontsize=14, fontweight='bold')
        
    def _generate_heatmap(self):
        """Generate heatmap from brain region data"""
        self.heatmap_grid.fill(0.0)
        
        for region in self.monitor.brain_regions.values():
            # Map region position to heatmap coordinates
            x_center = int(region.position[0] * self.heatmap_size)
            y_center = int(region.position[1] * self.heatmap_size)
            radius = int(region.size * self.heatmap_size)
            
            # Create heat blob around region center
            heat_intensity = region.current_load / region.max_capacity
            
            for y in range(max(0, y_center - radius), min(self.heatmap_size, y_center + radius)):
                for x in range(max(0, x_center - radius), min(self.heatmap_size, x_center + radius)):
                    distance = math.sqrt((x - x_center)**2 + (y - y_center)**2)
                    if distance <= radius:
                        # Gaussian-like heat distribution
                        heat_value = heat_intensity * math.exp(-distance**2 / (radius**2 / 4))
                        self.heatmap_grid[y, x] = max(self.heatmap_grid[y, x], heat_value)
    
    def _render_brain_regions(self):
        """Render brain regions with load indicators"""
        for region_name, region in self.monitor.brain_regions.items():
            x, y = region.position
            
            # Color based on region type
            color = self.monitor.region_configs[region.region_type]['color']
            
            # Size based on current load
            base_size = region.size * 1000
            load_factor = 1.0 + (region.current_load / region.max_capacity) * 0.5
            size = base_size * load_factor
            
            # Alpha based on temperature
            alpha = 0.4 + region.temperature * 0.6
            
            self.ax_regions.scatter([x], [y], c=color, s=size, alpha=alpha,
                                  edgecolors='white', linewidth=1)
            
            # Label active regions
            if region.current_load > 10:
                self.ax_regions.text(x, y - 0.08, region_name.replace('_', ' ').title(), 
                                   color='white', fontsize=8, ha='center',
                                   alpha=alpha)
                
                # Load value
                self.ax_regions.text(x, y + 0.08, f'{region.current_load:.1f}', 
                                   color='yellow', fontsize=10, ha='center',
                                   fontweight='bold')
        
        self.ax_regions.set_xlim(0, 1)
        self.ax_regions.set_ylim(0, 1)
        self.ax_regions.set_title('🧠 Brain Regions', color='white', fontweight='bold')
        self.ax_regions.set_xticks([])
        self.ax_regions.set_yticks([])
    
    def _render_load_timeline(self):
        """Render load timeline for key regions"""
        key_regions = ['attention', 'memory', 'reasoning', 'integration']
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#5f27cd']
        
        for i, (region_name, color) in enumerate(zip(key_regions, colors)):
            if region_name in self.monitor.brain_regions:
                region = self.monitor.brain_regions[region_name]
                history = list(region.load_history)
                
                if len(history) > 1:
                    x_vals = range(len(history))
                    self.ax_timeline.plot(x_vals, history, color=color, 
                                        label=region_name.title(), linewidth=2, alpha=0.8)
        
        self.ax_timeline.set_title('📈 Load Timeline', color='white', fontweight='bold')
        self.ax_timeline.legend(loc='upper left', fontsize=8)
        self.ax_timeline.tick_params(colors='white')
        self.ax_timeline.set_xlabel('Time Steps', color='white')
        self.ax_timeline.set_ylabel('Load', color='white')
        
        # Grid
        self.ax_timeline.grid(True, alpha=0.3, color='white')
    
    def _render_system_metrics(self):
        """Render system performance metrics"""
        # Calculate metrics
        total_load = sum(region.current_load for region in self.monitor.brain_regions.values())
        max_capacity = sum(region.max_capacity for region in self.monitor.brain_regions.values())
        utilization = total_load / max_capacity if max_capacity > 0 else 0
        
        avg_efficiency = sum(region.efficiency for region in self.monitor.brain_regions.values()) / len(self.monitor.brain_regions)
        avg_health = sum(region.health for region in self.monitor.brain_regions.values()) / len(self.monitor.brain_regions)
        avg_temperature = sum(region.temperature for region in self.monitor.brain_regions.values()) / len(self.monitor.brain_regions)
        
        # Active processes
        total_processes = sum(len(region.active_processes) for region in self.monitor.brain_regions.values())
        
        # Create metrics display
        metrics = [
            f"Total Load: {total_load:.1f}",
            f"Utilization: {utilization:.1%}",
            f"Efficiency: {avg_efficiency:.1%}",
            f"Health: {avg_health:.1%}",
            f"Temperature: {avg_temperature:.2f}",
            f"Active Processes: {total_processes}"
        ]
        
        # Display as text
        for i, metric in enumerate(metrics):
            self.ax_metrics.text(0.05, 0.9 - i*0.15, metric, 
                               transform=self.ax_metrics.transAxes,
                               color='white', fontsize=12, fontweight='bold')
        
        # Visual bars for key metrics
        bar_width = 0.6
        bar_height = 0.08
        
        # Utilization bar
        util_color = 'green' if utilization < 0.7 else 'yellow' if utilization < 0.9 else 'red'
        self.ax_metrics.barh(0.5, utilization * bar_width, bar_height, 
                           left=0.05, color=util_color, alpha=0.7,
                           transform=self.ax_metrics.transAxes)
        
        # Efficiency bar
        eff_color = 'green' if avg_efficiency > 0.8 else 'yellow' if avg_efficiency > 0.6 else 'red'
        self.ax_metrics.barh(0.35, avg_efficiency * bar_width, bar_height,
                           left=0.05, color=eff_color, alpha=0.7,
                           transform=self.ax_metrics.transAxes)
        
        # Health bar
        health_color = 'green' if avg_health > 0.8 else 'yellow' if avg_health > 0.6 else 'red'
        self.ax_metrics.barh(0.2, avg_health * bar_width, bar_height,
                           left=0.05, color=health_color, alpha=0.7,
                           transform=self.ax_metrics.transAxes)
        
        self.ax_metrics.set_xlim(0, 1)
        self.ax_metrics.set_ylim(0, 1)
        self.ax_metrics.set_title('📊 System Metrics', color='white', fontweight='bold')
        self.ax_metrics.set_xticks([])
        self.ax_metrics.set_yticks([])
        
    def start_visualization(self):
        """Start the real-time visualization"""
        logger.info("Starting cognitive load heatmap visualization...")
        
        # Create animation
        self.animation = animation.FuncAnimation(
            self.fig, 
            self.animate_frame,
            interval=200,  # 5 FPS
            blit=False,
            cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the cognitive load visualization"""
    print("⚡ Starting DAWN Cognitive Load Heat Map Visualization")
    
    visualizer = CognitiveLoadHeatmapVisualizer()
    visualizer.start_visualization()

if __name__ == "__main__":
    main()
