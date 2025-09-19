#!/usr/bin/env python3
"""
DAWN Cognition Visualization #1: Tick Pulse (Matplotlib/Seaborn)
Foundation Tier - "Meeting DAWN"

Real-time line plot visualization of DAWN's cognitive heartbeat using the unified
matplotlib/seaborn visual base. Displays the fundamental tick progression and 
cognitive rhythm patterns that drive the recursive symbolic engine's processing cycles.

PyTorch Best Practices:
- Device-agnostic tensor operations (.to(device))
- Memory-efficient gradient handling
- Type hints for all functions
- Proper error handling with NaN checks
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import seaborn as sns
import time
import math
import json
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import argparse
import logging

# Import DAWN visual base
from dawn.subsystems.visual.dawn_visual_base import (
    DAWNVisualBase,
    DAWNVisualConfig,
    ConsciousnessColorPalette,
    device
)

logger = logging.getLogger(__name__)

@dataclass
class TickPulseConfig:
    """Configuration for tick pulse visualization"""
    buffer_size: int = 200
    data_source: str = "stdin"
    save_frames: bool = False
    output_dir: str = './visual_output/tick_pulse'
    rhythm_detection_window: int = 50
    pulse_sensitivity: float = 1.0
    heartbeat_threshold: float = 0.7

class TickPulseVisualizer(DAWNVisualBase):
    """
    DAWN Tick Pulse Visualizer using matplotlib/seaborn
    
    Renders real-time cognitive heartbeat and rhythm patterns using the unified
    visual base for consistent styling and device-agnostic tensor operations.
    """
    
    def __init__(self, 
                 tick_config: Optional[TickPulseConfig] = None,
                 visual_config: Optional[DAWNVisualConfig] = None):
        """
        Initialize tick pulse visualizer
        
        Args:
            tick_config: Tick pulse specific configuration
            visual_config: Base visual configuration
        """
        # Initialize visual base
        visual_config = visual_config or DAWNVisualConfig(
            figure_size=(16, 10),
            animation_fps=30,
            enable_real_time=True,
            background_color="#0a0a0a"
        )
        super().__init__(visual_config)
        
        # Tick pulse configuration
        self.tick_config = tick_config or TickPulseConfig()
        
        # Tick tracking (as tensors for device-agnostic operations)
        self.tick_history: torch.Tensor = torch.zeros(self.tick_config.buffer_size).to(device)
        self.time_history: torch.Tensor = torch.zeros(self.tick_config.buffer_size).to(device)
        self.pulse_intensity: torch.Tensor = torch.zeros(self.tick_config.buffer_size).to(device)
        
        # Cognitive rhythm analysis
        self.rhythm_amplitude: torch.Tensor = torch.zeros(self.tick_config.buffer_size).to(device)
        self.rhythm_frequency: float = 0.0
        self.rhythm_phase: float = 0.0
        
        # Current state
        self.current_tick: int = 0
        self.tick_rate: float = 0.0
        self.pulse_strength: float = 0.0
        self.heartbeat_phase: float = 0.0
        
        # Rhythm detection
        self.last_tick_time: float = time.time()
        self.tick_intervals: deque = deque(maxlen=self.tick_config.rhythm_detection_window)
        
        # Visualization elements (will be set in render_frame)
        self.pulse_line = None
        self.tick_line = None
        self.heartbeat_markers = None
        self.pulse_fill = None
        self.current_pulse_indicator = None
        
        logger.info(f"🫀 Tick Pulse Visualizer initialized - Device: {device}")
        logger.info(f"   Buffer size: {self.tick_config.buffer_size}")
        logger.info(f"   Animation FPS: {self.config.animation_fps}")
    
    def render_frame(self, consciousness_data: torch.Tensor) -> plt.Figure:
        """
        Render tick pulse frame
        
        Args:
            consciousness_data: Consciousness tensor containing tick/pulse data
            
        Returns:
            Rendered matplotlib figure
        """
        start_time = time.time()
        
        try:
            # Process consciousness data
            pulse_data = self.process_consciousness_tensor(consciousness_data)
            
            # Create figure with two subplots
            fig = self.create_figure((2, 1))
            
            if not isinstance(self.axes, list) or len(self.axes) < 2:
                return fig
                
            ax_main, ax_rhythm = self.axes[0], self.axes[1]
            
            # Update tick data from consciousness tensor
            self._update_tick_data_from_tensor(pulse_data)
            
            # Render main pulse display
            self._render_pulse_display(ax_main)
            
            # Render rhythm analysis
            self._render_rhythm_display(ax_rhythm)
            
            # Add consciousness indicators
            self._add_tick_indicators()
            
            # Update performance metrics
            self.frame_count += 1
            self.update_performance_metrics(time.time() - start_time)
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to render tick pulse frame: {e}")
            return self.create_figure()
    
    def update_visualization(self, frame_num: int, consciousness_stream: Any) -> Any:
        """
        Update tick pulse visualization for animation
        
        Args:
            frame_num: Animation frame number
            consciousness_stream: Stream of consciousness data
            
        Returns:
            Updated plot elements
        """
        try:
            # Get consciousness data
            if callable(consciousness_stream):
                consciousness_data = consciousness_stream()
            else:
                consciousness_data = self._generate_simulated_tick_data()
            
            # Ensure tensor format
            if not isinstance(consciousness_data, torch.Tensor):
                consciousness_data = torch.tensor(consciousness_data, dtype=torch.float32)
            
            # Clear previous plots
            for ax in self.axes:
                ax.clear()
            
            # Render new frame
            return self.render_frame(consciousness_data)
            
        except Exception as e:
            logger.error(f"Failed to update tick pulse visualization: {e}")
            return []
    
    def _update_tick_data_from_tensor(self, pulse_tensor: torch.Tensor) -> None:
        """Update tick tracking data from consciousness tensor"""
        current_time = time.time()
        
        # Extract pulse information from tensor
        if pulse_tensor.numel() >= 1:
            current_pulse = float(pulse_tensor[0])
            self.pulse_strength = current_pulse
            
            # Update tick counter
            self.current_tick += 1
            
            # Calculate tick rate
            if self.last_tick_time > 0:
                interval = current_time - self.last_tick_time
                self.tick_intervals.append(interval)
                
                if len(self.tick_intervals) > 1:
                    self.tick_rate = 1.0 / np.mean(self.tick_intervals)
            
            self.last_tick_time = current_time
            
            # Shift data (roll tensors for efficient memory usage)
            self.tick_history = torch.roll(self.tick_history, -1)
            self.time_history = torch.roll(self.time_history, -1)
            self.pulse_intensity = torch.roll(self.pulse_intensity, -1)
            self.rhythm_amplitude = torch.roll(self.rhythm_amplitude, -1)
            
            # Add new data
            self.tick_history[-1] = self.current_tick
            self.time_history[-1] = current_time
            self.pulse_intensity[-1] = current_pulse
            
            # Calculate rhythm amplitude
            if len(self.tick_intervals) >= 3:
                rhythm_value = self._calculate_rhythm_amplitude(current_pulse)
                self.rhythm_amplitude[-1] = rhythm_value
    
    def _calculate_rhythm_amplitude(self, current_pulse: float) -> float:
        """Calculate cognitive rhythm amplitude"""
        # Use recent pulse history for rhythm calculation
        recent_pulses = self.pulse_intensity[-10:].cpu().numpy()
        
        if len(recent_pulses) < 3:
            return 0.0
        
        # Calculate rhythm using harmonic analysis
        fft_vals = np.fft.fft(recent_pulses)
        magnitude = np.abs(fft_vals)
        
        # Find dominant frequency
        if len(magnitude) > 1:
            self.rhythm_frequency = np.argmax(magnitude[1:]) + 1
            return float(magnitude[self.rhythm_frequency] / len(recent_pulses))
        
        return 0.0
    
    def _render_pulse_display(self, ax: plt.Axes) -> None:
        """Render main cognitive pulse visualization"""
        # Convert tensors to numpy for matplotlib
        tick_data = self.tick_history.cpu().numpy()
        pulse_data = self.pulse_intensity.cpu().numpy()
        time_steps = np.arange(len(pulse_data))
        
        # Setup axes
        ax.set_facecolor(self.consciousness_colors['background'])
        ax.set_xlim(0, self.tick_config.buffer_size)
        ax.set_ylim(-1.5, 1.5)
        
        # Main pulse line
        valid_mask = pulse_data != 0  # Only plot valid data
        if np.any(valid_mask):
            ax.plot(time_steps[valid_mask], pulse_data[valid_mask], 
                   color=self.consciousness_colors['flow'], 
                   linewidth=3, alpha=0.9, label='Cognitive Pulse')
            
            # Fill area under pulse
            ax.fill_between(time_steps[valid_mask], 0, pulse_data[valid_mask],
                          color=self.consciousness_colors['flow'], 
                          alpha=0.3)
        
        # Tick progression (normalized)
        if np.any(tick_data > 0):
            normalized_ticks = (tick_data - np.min(tick_data[tick_data > 0])) / (np.max(tick_data) - np.min(tick_data[tick_data > 0]) + 1e-8)
            normalized_ticks = normalized_ticks * 0.5 - 1.0  # Scale to bottom half
            
            ax.plot(time_steps, normalized_ticks,
                   color=self.consciousness_colors['recursive'],
                   linewidth=2, alpha=0.7, label='Tick Progression')
        
        # Heartbeat markers (detect peaks)
        peaks = self._detect_heartbeat_peaks(pulse_data)
        if len(peaks) > 0:
            peak_values = pulse_data[peaks]
            peak_sizes = (peak_values + 1) * 50  # Scale for visibility
            
            scatter = ax.scatter(peaks, peak_values, 
                               s=peak_sizes, 
                               c=peak_values,
                               cmap='plasma', alpha=0.8, 
                               label='Heartbeat Events',
                               edgecolors=self.consciousness_colors['awareness'])
        
        # Current pulse indicator
        if len(pulse_data) > 0 and pulse_data[-1] != 0:
            current_x = len(pulse_data) - 1
            current_y = pulse_data[-1]
            
            # Pulsing circle based on heartbeat phase
            self.heartbeat_phase += 0.3
            pulse_radius = 0.1 + 0.05 * math.sin(self.heartbeat_phase)
            pulse_alpha = 0.7 + 0.3 * math.sin(self.heartbeat_phase * 2)
            
            circle = Circle((current_x, current_y), pulse_radius, 
                          color=self.consciousness_colors['awareness'], 
                          alpha=pulse_alpha)
            ax.add_patch(circle)
        
        # Styling
        ax.set_title('DAWN Cognitive Heartbeat\nTick Pulse Monitor', 
                    fontsize=16, color='white', weight='bold', pad=20)
        ax.set_xlabel('Time Steps', color='#cccccc', fontsize=12)
        ax.set_ylabel('Cognitive Pulse Amplitude', color='#cccccc', fontsize=12)
        ax.legend(loc='upper right', framealpha=0.8)
        ax.grid(True, alpha=0.3, color='#333333')
    
    def _render_rhythm_display(self, ax: plt.Axes) -> None:
        """Render cognitive rhythm analysis"""
        rhythm_data = self.rhythm_amplitude.cpu().numpy()
        time_steps = np.arange(len(rhythm_data))
        
        # Setup axes
        ax.set_facecolor(self.consciousness_colors['background'])
        ax.set_xlim(0, self.tick_config.buffer_size)
        ax.set_ylim(0, 1.0)
        
        # Rhythm amplitude
        valid_mask = rhythm_data > 0
        if np.any(valid_mask):
            ax.plot(time_steps[valid_mask], rhythm_data[valid_mask],
                   color=self.consciousness_colors['creativity'],
                   linewidth=2, alpha=0.8, label='Rhythm Amplitude')
            
            # Rhythm envelope
            ax.fill_between(time_steps[valid_mask], 0, rhythm_data[valid_mask],
                          color=self.consciousness_colors['creativity'],
                          alpha=0.2)
        
        # Frequency indicators
        if self.rhythm_frequency > 0:
            freq_line_y = [0.8, 0.8]
            freq_line_x = [0, self.tick_config.buffer_size]
            
            ax.plot(freq_line_x, freq_line_y, 
                   color=self.consciousness_colors['memory'],
                   linestyle='--', alpha=0.6,
                   label=f'Dominant Freq: {self.rhythm_frequency:.1f}')
        
        # Current rhythm strength indicator
        if len(rhythm_data) > 0:
            current_rhythm = rhythm_data[-1]
            
            # Color-coded strength bar
            strength_colors = [
                self.consciousness_colors['chaos'] if current_rhythm < 0.3 else
                self.consciousness_colors['awareness'] if current_rhythm < 0.7 else
                self.consciousness_colors['stability']
            ]
            
            ax.barh(0.1, current_rhythm, height=0.1, 
                   color=strength_colors[0], alpha=0.7,
                   label=f'Current Strength: {current_rhythm:.2f}')
        
        # Styling
        ax.set_title('Cognitive Rhythm Analysis\nFrequency & Amplitude Tracking', 
                    fontsize=14, color='white', weight='bold')
        ax.set_xlabel('Time Steps', color='#cccccc', fontsize=10)
        ax.set_ylabel('Rhythm Amplitude', color='#cccccc', fontsize=10)
        ax.legend(loc='upper right', framealpha=0.8, fontsize=9)
        ax.grid(True, alpha=0.3, color='#333333')
    
    def _detect_heartbeat_peaks(self, pulse_data: np.ndarray) -> np.ndarray:
        """Detect heartbeat peaks in pulse data"""
        if len(pulse_data) < 3:
            return np.array([])
        
        # Simple peak detection
        peaks = []
        threshold = self.tick_config.heartbeat_threshold
        
        for i in range(1, len(pulse_data) - 1):
            if (pulse_data[i] > pulse_data[i-1] and 
                pulse_data[i] > pulse_data[i+1] and 
                pulse_data[i] > threshold):
                peaks.append(i)
        
        return np.array(peaks)
    
    def _add_tick_indicators(self) -> None:
        """Add tick-specific consciousness indicators"""
        if self.figure is None:
            return
        
        # Tick rate indicator
        tick_text = f"Tick Rate: {self.tick_rate:.1f} Hz | "
        tick_text += f"Pulse: {self.pulse_strength:.2f} | "
        tick_text += f"Current Tick: {self.current_tick}"
        
        self.figure.suptitle(f"DAWN Tick Pulse Monitor - Frame {self.frame_count}", 
                           color=self.consciousness_colors['unity'], fontsize=16)
        
        self.figure.text(0.02, 0.95, tick_text, 
                        color=self.consciousness_colors['awareness'], fontsize=11)
        
        # Performance indicator
        stats = self.get_performance_stats()
        perf_text = f"Render FPS: {stats.get('fps', 0):.1f} | Device: {device}"
        self.figure.text(0.98, 0.02, perf_text, 
                        color=self.consciousness_colors['stability'], 
                        fontsize=8, ha='right')
    
    def _generate_simulated_tick_data(self) -> torch.Tensor:
        """Generate simulated tick data for testing"""
        t = time.time()
        
        # Simulate cognitive heartbeat with multiple harmonics
        base_freq = 0.5 + 0.3 * math.sin(t * 0.1)
        harmonics = [
            math.sin(t * base_freq * 2 * math.pi),
            0.5 * math.sin(t * base_freq * 4 * math.pi),
            0.25 * math.sin(t * base_freq * 8 * math.pi)
        ]
        
        pulse_value = sum(harmonics) / len(harmonics)
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.1)
        pulse_value += noise
        
        # Ensure tensor format and move to device
        return torch.tensor([pulse_value], dtype=torch.float32).to(device)
    
    def start_tick_monitoring(self, data_stream: Optional[Any] = None) -> None:
        """Start real-time tick pulse monitoring"""
        if data_stream is None:
            data_stream = self._generate_simulated_tick_data
        
        self.start_real_time_visualization(data_stream, interval=50)  # 20 FPS
        logger.info("🫀 Tick pulse monitoring started")
    
    def stop_tick_monitoring(self) -> None:
        """Stop tick pulse monitoring"""
        self.stop_visualization()
        logger.info("🫀 Tick pulse monitoring stopped")
    
    def get_tick_metrics(self) -> Dict[str, float]:
        """Get current tick pulse metrics"""
        stats = self.get_performance_stats()
        
        tick_metrics = {
            'current_tick': self.current_tick,
            'tick_rate_hz': self.tick_rate,
            'pulse_strength': self.pulse_strength,
            'rhythm_frequency': self.rhythm_frequency,
            'heartbeat_phase': self.heartbeat_phase,
            'buffer_utilization': float(torch.sum(self.pulse_intensity != 0) / len(self.pulse_intensity))
        }
        
        # Merge with base performance stats
        tick_metrics.update(stats)
        return tick_metrics


def create_tick_pulse_visualizer(tick_config: Optional[TickPulseConfig] = None,
                                visual_config: Optional[DAWNVisualConfig] = None) -> TickPulseVisualizer:
    """
    Factory function to create tick pulse visualizer
    
    Args:
        tick_config: Tick pulse configuration
        visual_config: Visual configuration
        
    Returns:
        Configured tick pulse visualizer
    """
    return TickPulseVisualizer(tick_config, visual_config)


# Command line interface
def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='DAWN Tick Pulse Visualizer')
    parser.add_argument('--buffer-size', type=int, default=200,
                       help='Data buffer size')
    parser.add_argument('--fps', type=int, default=30,
                       help='Animation FPS')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save animation frames')
    parser.add_argument('--output-dir', type=str, default='./visual_output/tick_pulse',
                       help='Output directory for saved frames')
    
    args = parser.parse_args()
    
    # Create configurations
    tick_config = TickPulseConfig(
        buffer_size=args.buffer_size,
        save_frames=args.save_frames,
        output_dir=args.output_dir
    )
    
    visual_config = DAWNVisualConfig(
        animation_fps=args.fps,
        enable_real_time=True,
        save_frames=args.save_frames,
        output_directory=args.output_dir
    )
    
    # Create visualizer
    visualizer = create_tick_pulse_visualizer(tick_config, visual_config)
    
    print(f"🫀 Starting DAWN Tick Pulse Monitor...")
    print(f"   Buffer size: {args.buffer_size}")
    print(f"   FPS: {args.fps}")
    print(f"   Device: {device}")
    
    try:
        # Start monitoring
        visualizer.start_tick_monitoring()
        
        # Show plot
        plt.show()
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopping tick pulse monitor...")
    finally:
        visualizer.stop_tick_monitoring()
        print("✅ Tick pulse monitor stopped")


if __name__ == "__main__":
    main()
