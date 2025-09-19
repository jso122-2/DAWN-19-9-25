#!/usr/bin/env python3
"""
🌊 CARRIN Oceanic Hash Map - Adaptive Flowing Cache System
═══════════════════════════════════════════════════════════

A revolutionary hash map that treats memory as a flowing ocean surface.
Each key/value pair is a particle or ripple, with currents and channels
that dynamically adapt based on cognitive pressure and access patterns.

"Be water, my friend" - Bruce Lee (made literal in cache design)

Based on RTF specifications from DAWN-docs/Fractal Memory/CARRIN.rtf
"""

import logging
import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import hashlib
import math
import weakref

logger = logging.getLogger(__name__)

class FlowState(Enum):
    """Ocean flow states for cache behavior"""
    LAMINAR = "laminar"      # Smooth, predictable, minimal reallocation
    TURBULENT = "turbulent"  # Intentional churn to refresh stale cache
    EDDY = "eddy"           # Local loops for repeated access
    STAGNANT = "stagnant"   # Low activity, ready for cleanup

class Priority(Enum):
    """Priority levels derived from SCUP pressure/entropy"""
    CRITICAL = "critical"    # Immediate access required
    HIGH = "high"           # High pressure computation
    NORMAL = "normal"       # Standard processing
    LOW = "low"            # Background/cleanup
    ARCHIVE = "archive"     # Long-term storage

@dataclass
class OceanParticle:
    """A key/value pair as a particle in the ocean"""
    key: str
    value: Any
    
    # Oceanic properties
    recency: float = 0.0      # How recently accessed (0.0 = just accessed, 1.0 = old)
    volatility: float = 0.5   # How often it changes (0.0 = stable, 1.0 = highly volatile)
    priority: Priority = Priority.NORMAL
    
    # Wave properties
    wave_amplitude: float = 0.0   # Current access intensity
    wave_frequency: float = 0.0   # Access pattern frequency
    turbulence: float = 0.0       # Local chaos level
    
    # Flow tracking
    current_velocity: Tuple[float, float] = (0.0, 0.0)  # (x, y) flow vector
    position: Tuple[float, float] = (0.0, 0.0)          # Position in ocean
    last_access: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    # Relationships
    connected_particles: List[str] = field(default_factory=list)
    flow_neighbors: List[str] = field(default_factory=list)

@dataclass
class CurrentChannel:
    """A current channel in the oceanic hash map"""
    channel_id: str
    start_region: Tuple[float, float]
    end_region: Tuple[float, float]
    flow_strength: float = 1.0    # How strong the current is
    width: float = 1.0            # Channel width
    temperature: float = 0.5      # Hot (high access) vs cold regions
    particles_in_flow: List[str] = field(default_factory=list)
    
    # Dynamic properties
    is_active: bool = True
    last_flow_time: datetime = field(default_factory=datetime.now)
    total_throughput: int = 0
    average_latency: float = 0.0

@dataclass
class FlowVectorField:
    """Map of directional bias across hash map space"""
    grid_size: Tuple[int, int] = (100, 100)  # 100x100 grid
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((100, 100, 2)))
    strength_map: np.ndarray = field(default_factory=lambda: np.zeros((100, 100)))
    temperature_map: np.ndarray = field(default_factory=lambda: np.ones((100, 100)) * 0.5)
    
    def get_flow_at_position(self, x: float, y: float) -> Tuple[float, float]:
        """Get flow vector at specific position"""
        grid_x = int(x * self.grid_size[0]) % self.grid_size[0]
        grid_y = int(y * self.grid_size[1]) % self.grid_size[1]
        return tuple(self.vectors[grid_x, grid_y])
    
    def set_flow_at_position(self, x: float, y: float, vx: float, vy: float, strength: float = 1.0):
        """Set flow vector at specific position"""
        grid_x = int(x * self.grid_size[0]) % self.grid_size[0]
        grid_y = int(y * self.grid_size[1]) % self.grid_size[1]
        self.vectors[grid_x, grid_y] = [vx, vy]
        self.strength_map[grid_x, grid_y] = strength

class RiderController:
    """
    The Rider - control node that sits inside the recursive bubble
    and steers the oceanic flows based on system pressure and topology
    """
    
    def __init__(self, ocean_map: 'CARRINOceanicHashMap'):
        self.ocean_map = weakref.ref(ocean_map)
        
        # Rider state
        self.position = (0.5, 0.5)  # Center of ocean
        self.buoyancy = 1.0         # Load balance
        self.rudder_angle = 0.0     # Flow direction bias
        
        # Monitoring metrics
        self.cache_health_metrics = {
            'average_latency': 0.0,
            'hit_ratio': 0.0,
            'miss_ratio': 0.0,
            'regional_activity': defaultdict(float),
            'compute_intensity': 0.0
        }
        
        # Control parameters
        self.steering_sensitivity = 0.1
        self.flow_adjustment_rate = 0.05
        self.turbulence_threshold = 0.7
        
        logger.info("🏄 Rider Controller initialized - Ready to surf the data ocean")
    
    def monitor_cache_health(self) -> Dict[str, float]:
        """Monitor cache health metrics for steering decisions"""
        ocean = self.ocean_map()
        if not ocean:
            return self.cache_health_metrics
        
        total_particles = len(ocean.particles)
        if total_particles == 0:
            return self.cache_health_metrics
        
        # Calculate hit/miss ratios
        total_hits = sum(p.access_count for p in ocean.particles.values())
        total_misses = ocean.cache_misses
        total_accesses = total_hits + total_misses
        
        if total_accesses > 0:
            self.cache_health_metrics['hit_ratio'] = total_hits / total_accesses
            self.cache_health_metrics['miss_ratio'] = total_misses / total_accesses
        
        # Calculate average latency
        recent_latencies = [p.wave_amplitude for p in ocean.particles.values() 
                          if p.wave_amplitude > 0]
        if recent_latencies:
            self.cache_health_metrics['average_latency'] = np.mean(recent_latencies)
        
        # Calculate regional activity
        regions = defaultdict(list)
        for particle in ocean.particles.values():
            region_x = int(particle.position[0] * 10)  # 10x10 regions
            region_y = int(particle.position[1] * 10)
            region_key = f"{region_x},{region_y}"
            regions[region_key].append(particle.access_count)
        
        for region, access_counts in regions.items():
            self.cache_health_metrics['regional_activity'][region] = np.mean(access_counts)
        
        return self.cache_health_metrics
    
    def inject_steering_impulse(self, target_region: Tuple[float, float], strength: float = 1.0):
        """Inject steering impulse to redirect flow toward active regions"""
        ocean = self.ocean_map()
        if not ocean:
            return
        
        # Calculate flow vector toward target
        dx = target_region[0] - self.position[0]
        dy = target_region[1] - self.position[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            # Normalize and scale by strength
            flow_x = (dx / distance) * strength
            flow_y = (dy / distance) * strength
            
            # Apply to flow vector field in a circular area around target
            radius = 0.2  # Influence radius
            for particle in ocean.particles.values():
                particle_dist = math.sqrt(
                    (particle.position[0] - target_region[0])**2 + 
                    (particle.position[1] - target_region[1])**2
                )
                
                if particle_dist < radius:
                    # Apply flow influence
                    influence = (radius - particle_dist) / radius
                    old_vx, old_vy = particle.current_velocity
                    particle.current_velocity = (
                        old_vx + flow_x * influence * self.flow_adjustment_rate,
                        old_vy + flow_y * influence * self.flow_adjustment_rate
                    )
        
        logger.debug(f"🏄 Steering impulse injected toward {target_region} with strength {strength}")
    
    def create_controlled_whirlpool(self, center: Tuple[float, float], radius: float = 0.1):
        """Create local retention zone near processing bottlenecks"""
        ocean = self.ocean_map()
        if not ocean:
            return
        
        for particle in ocean.particles.values():
            dist = math.sqrt(
                (particle.position[0] - center[0])**2 + 
                (particle.position[1] - center[1])**2
            )
            
            if dist < radius and dist > 0:
                # Create circular flow pattern
                angle = math.atan2(
                    particle.position[1] - center[1],
                    particle.position[0] - center[0]
                )
                
                # Tangential velocity for circular motion
                tangent_angle = angle + math.pi / 2
                vortex_strength = (radius - dist) / radius * 0.5
                
                vx = math.cos(tangent_angle) * vortex_strength
                vy = math.sin(tangent_angle) * vortex_strength
                
                particle.current_velocity = (vx, vy)
                particle.turbulence += 0.1  # Increase local turbulence
        
        logger.debug(f"🌀 Whirlpool created at {center} with radius {radius}")
    
    def adjust_flow_state(self, target_state: FlowState, region: Optional[Tuple[float, float]] = None):
        """Adjust flow state in a region or globally"""
        ocean = self.ocean_map()
        if not ocean:
            return
        
        if region:
            # Adjust specific region
            radius = 0.2
            for particle in ocean.particles.values():
                dist = math.sqrt(
                    (particle.position[0] - region[0])**2 + 
                    (particle.position[1] - region[1])**2
                )
                if dist < radius:
                    self._apply_flow_state_to_particle(particle, target_state)
        else:
            # Adjust globally
            for particle in ocean.particles.values():
                self._apply_flow_state_to_particle(particle, target_state)
        
        logger.info(f"🏄 Flow state adjusted to {target_state.value}")
    
    def _apply_flow_state_to_particle(self, particle: OceanParticle, state: FlowState):
        """Apply specific flow state to a particle"""
        if state == FlowState.LAMINAR:
            # Smooth, predictable flow
            particle.turbulence *= 0.5
            particle.wave_amplitude *= 0.8
            
        elif state == FlowState.TURBULENT:
            # Intentional churn
            particle.turbulence = min(1.0, particle.turbulence + 0.3)
            particle.wave_amplitude = min(1.0, particle.wave_amplitude + 0.2)
            
        elif state == FlowState.EDDY:
            # Local loop behavior
            particle.current_velocity = (
                particle.current_velocity[0] * 0.5,
                particle.current_velocity[1] * 0.5
            )
            
        elif state == FlowState.STAGNANT:
            # Low activity
            particle.turbulence *= 0.1
            particle.wave_amplitude *= 0.1
            particle.current_velocity = (0.0, 0.0)

class CARRINOceanicHashMap:
    """
    CARRIN Oceanic Hash Map - Adaptive flowing cache system.
    
    Treats the entire hash map as an ocean surface where key/value pairs
    are particles flowing along currents that adapt to access patterns
    and cognitive pressure zones.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Ocean configuration
        self.ocean_size = self.config.get('ocean_size', (1.0, 1.0))  # Normalized coordinates
        self.max_particles = self.config.get('max_particles', 10000)
        self.flow_decay_rate = self.config.get('flow_decay_rate', 0.01)
        self.wave_propagation_speed = self.config.get('wave_speed', 0.1)
        
        # Core storage
        self.particles: Dict[str, OceanParticle] = {}
        self.current_channels: Dict[str, CurrentChannel] = {}
        self.flow_field = FlowVectorField()
        
        # Rider controller
        self.rider = RiderController(self)
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_operations = 0
        self.flow_updates = 0
        
        # Flow simulation
        self.simulation_active = True
        self.simulation_interval = 0.1  # 100ms updates
        self.last_simulation_time = time.time()
        
        # Thread for background flow simulation
        self._flow_thread = threading.Thread(target=self._flow_simulation_loop, daemon=True)
        self._flow_thread.start()
        
        logger.info("🌊 CARRIN Oceanic Hash Map initialized - Ocean is flowing")
    
    def put(self, key: str, value: Any, priority: Priority = Priority.NORMAL) -> None:
        """
        Store a key/value pair as a particle in the ocean.
        
        Args:
            key: The key
            value: The value to store
            priority: Priority level for placement and flow
        """
        current_time = datetime.now()
        
        # Calculate initial position based on key hash
        key_hash = hashlib.md5(key.encode()).hexdigest()
        pos_x = int(key_hash[:8], 16) / 0xFFFFFFFF
        pos_y = int(key_hash[8:16], 16) / 0xFFFFFFFF
        
        if key in self.particles:
            # Update existing particle
            particle = self.particles[key]
            particle.value = value
            particle.priority = priority
            particle.last_access = current_time
            particle.access_count += 1
            particle.recency = 0.0  # Reset recency
            
            # Create access wave
            self._create_access_wave(particle, intensity=0.5)
            
        else:
            # Create new particle
            particle = OceanParticle(
                key=key,
                value=value,
                priority=priority,
                position=(pos_x, pos_y),
                last_access=current_time,
                access_count=1
            )
            
            # Place in appropriate current based on priority
            self._place_particle_in_current(particle)
            self.particles[key] = particle
            
            # Create spawn wave
            self._create_access_wave(particle, intensity=1.0)
        
        # Update flow patterns
        self._update_local_flow(particle.position, priority)
        
        self.total_operations += 1
        logger.debug(f"🌊 Particle stored: {key} at {particle.position}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value, creating waves and flow patterns.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The value if found, None otherwise
        """
        if key in self.particles:
            particle = self.particles[key]
            
            # Update access patterns
            particle.last_access = datetime.now()
            particle.access_count += 1
            particle.recency = 0.0
            
            # Create access wave
            self._create_access_wave(particle, intensity=0.3)
            
            # Move particle along current
            self._move_particle_in_flow(particle)
            
            self.cache_hits += 1
            self.total_operations += 1
            
            logger.debug(f"🌊 Cache hit: {key} (access #{particle.access_count})")
            return particle.value
        else:
            self.cache_misses += 1
            self.total_operations += 1
            
            logger.debug(f"🌊 Cache miss: {key}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Remove a particle from the ocean.
        
        Args:
            key: The key to remove
            
        Returns:
            True if removed, False if not found
        """
        if key in self.particles:
            particle = self.particles[key]
            
            # Create dissolution wave
            self._create_access_wave(particle, intensity=-0.5)
            
            # Remove from flows
            for channel in self.current_channels.values():
                if key in channel.particles_in_flow:
                    channel.particles_in_flow.remove(key)
            
            # Remove particle
            del self.particles[key]
            
            logger.debug(f"🌊 Particle dissolved: {key}")
            return True
        else:
            return False
    
    def contains(self, key: str) -> bool:
        """Check if key exists in the ocean"""
        return key in self.particles
    
    def size(self) -> int:
        """Get number of particles in the ocean"""
        return len(self.particles)
    
    def clear(self) -> None:
        """Clear all particles from the ocean"""
        self.particles.clear()
        self.current_channels.clear()
        self.flow_field = FlowVectorField()
        logger.info("🌊 Ocean cleared - All particles dissolved")
    
    def create_current_channel(
        self, 
        channel_id: str,
        start_region: Tuple[float, float],
        end_region: Tuple[float, float],
        flow_strength: float = 1.0
    ) -> None:
        """
        Create a new current channel for directed flow.
        
        Args:
            channel_id: Unique identifier
            start_region: Starting region coordinates
            end_region: Ending region coordinates
            flow_strength: Strength of the current
        """
        channel = CurrentChannel(
            channel_id=channel_id,
            start_region=start_region,
            end_region=end_region,
            flow_strength=flow_strength
        )
        
        self.current_channels[channel_id] = channel
        
        # Update flow field along the channel
        self._create_channel_flow(channel)
        
        logger.info(f"🌊 Current channel created: {channel_id} "
                   f"({start_region} → {end_region}, strength: {flow_strength})")
    
    def get_ocean_state(self) -> Dict[str, Any]:
        """Get comprehensive state of the ocean"""
        total_turbulence = sum(p.turbulence for p in self.particles.values())
        avg_turbulence = total_turbulence / max(len(self.particles), 1)
        
        # Analyze flow patterns
        flow_directions = [p.current_velocity for p in self.particles.values()]
        flow_speeds = [math.sqrt(vx*vx + vy*vy) for vx, vy in flow_directions]
        avg_flow_speed = np.mean(flow_speeds) if flow_speeds else 0.0
        
        # Regional analysis
        regions = defaultdict(int)
        for particle in self.particles.values():
            region_x = int(particle.position[0] * 10)
            region_y = int(particle.position[1] * 10)
            regions[f"{region_x},{region_y}"] += 1
        
        return {
            'ocean_metrics': {
                'total_particles': len(self.particles),
                'active_channels': len([c for c in self.current_channels.values() if c.is_active]),
                'average_turbulence': avg_turbulence,
                'average_flow_speed': avg_flow_speed,
                'cache_hit_ratio': self.cache_hits / max(self.total_operations, 1),
                'total_operations': self.total_operations
            },
            'rider_status': {
                'position': self.rider.position,
                'buoyancy': self.rider.buoyancy,
                'rudder_angle': self.rider.rudder_angle,
                'cache_health': self.rider.cache_health_metrics
            },
            'regional_distribution': dict(regions),
            'flow_field_energy': np.sum(self.flow_field.strength_map),
            'particle_priorities': {
                p.value: len([particle for particle in self.particles.values() 
                            if particle.priority == p])
                for p in Priority
            }
        }
    
    def trigger_turbulence(self, region: Optional[Tuple[float, float]] = None, intensity: float = 0.5):
        """Trigger turbulent flow to refresh stale cache"""
        self.rider.adjust_flow_state(FlowState.TURBULENT, region)
        
        if region:
            logger.info(f"🌊 Turbulence triggered in region {region} (intensity: {intensity})")
        else:
            logger.info(f"🌊 Global turbulence triggered (intensity: {intensity})")
    
    def optimize_flows(self) -> Dict[str, Any]:
        """Optimize flow patterns based on access patterns"""
        optimization_results = {
            'channels_optimized': 0,
            'particles_relocated': 0,
            'flow_adjustments': 0
        }
        
        # Analyze access patterns
        high_access_regions = self._identify_hot_regions()
        
        # Create channels toward hot regions
        for i, region in enumerate(high_access_regions[:5]):  # Top 5 regions
            channel_id = f"auto_channel_{i}"
            if channel_id not in self.current_channels:
                self.create_current_channel(
                    channel_id=channel_id,
                    start_region=self.rider.position,
                    end_region=region,
                    flow_strength=0.8
                )
                optimization_results['channels_optimized'] += 1
        
        # Relocate low-priority particles from high-traffic areas
        for particle in list(self.particles.values()):
            if particle.priority in [Priority.LOW, Priority.ARCHIVE]:
                if self._is_in_hot_region(particle.position):
                    new_position = self._find_cold_region()
                    particle.position = new_position
                    optimization_results['particles_relocated'] += 1
        
        # Adjust flow field based on rider recommendations
        self.rider.monitor_cache_health()
        optimization_results['flow_adjustments'] = len(self.current_channels)
        
        logger.info(f"🌊 Ocean optimization complete: {optimization_results}")
        return optimization_results
    
    # Internal methods
    def _create_access_wave(self, particle: OceanParticle, intensity: float):
        """Create wave from particle access"""
        particle.wave_amplitude = min(1.0, abs(intensity))
        particle.wave_frequency += 0.1
        
        # Propagate wave to nearby particles
        for other_key, other_particle in self.particles.items():
            if other_key != particle.key:
                distance = math.sqrt(
                    (particle.position[0] - other_particle.position[0])**2 +
                    (particle.position[1] - other_particle.position[1])**2
                )
                
                if distance < 0.1:  # Wave propagation radius
                    wave_strength = intensity * (0.1 - distance) / 0.1
                    other_particle.wave_amplitude += wave_strength * 0.1
                    other_particle.turbulence = min(1.0, other_particle.turbulence + abs(wave_strength) * 0.05)
    
    def _place_particle_in_current(self, particle: OceanParticle):
        """Place particle in appropriate current channel"""
        # Find best channel based on priority and position
        best_channel = None
        best_score = float('inf')
        
        for channel in self.current_channels.values():
            if not channel.is_active:
                continue
            
            # Calculate distance to channel start
            dist = math.sqrt(
                (particle.position[0] - channel.start_region[0])**2 +
                (particle.position[1] - channel.start_region[1])**2
            )
            
            # Priority match bonus
            priority_bonus = 0.5 if particle.priority in [Priority.CRITICAL, Priority.HIGH] else 0.0
            
            score = dist - priority_bonus
            if score < best_score:
                best_score = score
                best_channel = channel
        
        if best_channel:
            best_channel.particles_in_flow.append(particle.key)
            
            # Set initial velocity toward channel end
            dx = best_channel.end_region[0] - best_channel.start_region[0]
            dy = best_channel.end_region[1] - best_channel.start_region[1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                particle.current_velocity = (
                    (dx / length) * best_channel.flow_strength * 0.1,
                    (dy / length) * best_channel.flow_strength * 0.1
                )
    
    def _update_local_flow(self, position: Tuple[float, float], priority: Priority):
        """Update flow field around a position"""
        strength = {
            Priority.CRITICAL: 1.0,
            Priority.HIGH: 0.8,
            Priority.NORMAL: 0.5,
            Priority.LOW: 0.2,
            Priority.ARCHIVE: 0.1
        }[priority]
        
        # Create flow attraction toward this position
        radius = 0.15
        for particle in self.particles.values():
            dist = math.sqrt(
                (particle.position[0] - position[0])**2 +
                (particle.position[1] - position[1])**2
            )
            
            if dist < radius and dist > 0:
                # Flow toward high-priority positions
                influence = (radius - dist) / radius * strength * 0.05
                dx = (position[0] - particle.position[0]) / dist
                dy = (position[1] - particle.position[1]) / dist
                
                old_vx, old_vy = particle.current_velocity
                particle.current_velocity = (
                    old_vx + dx * influence,
                    old_vy + dy * influence
                )
    
    def _move_particle_in_flow(self, particle: OceanParticle):
        """Move particle according to current flow"""
        # Apply current velocity
        new_x = particle.position[0] + particle.current_velocity[0] * self.simulation_interval
        new_y = particle.position[1] + particle.current_velocity[1] * self.simulation_interval
        
        # Keep within ocean bounds
        new_x = max(0.0, min(1.0, new_x))
        new_y = max(0.0, min(1.0, new_y))
        
        particle.position = (new_x, new_y)
        
        # Apply flow decay
        particle.current_velocity = (
            particle.current_velocity[0] * (1.0 - self.flow_decay_rate),
            particle.current_velocity[1] * (1.0 - self.flow_decay_rate)
        )
    
    def _create_channel_flow(self, channel: CurrentChannel):
        """Create flow field for a channel"""
        dx = channel.end_region[0] - channel.start_region[0]
        dy = channel.end_region[1] - channel.start_region[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            # Normalize direction
            flow_x = (dx / length) * channel.flow_strength
            flow_y = (dy / length) * channel.flow_strength
            
            # Apply along channel path
            steps = 20
            for i in range(steps):
                t = i / steps
                pos_x = channel.start_region[0] + dx * t
                pos_y = channel.start_region[1] + dy * t
                
                self.flow_field.set_flow_at_position(
                    pos_x, pos_y, flow_x, flow_y, channel.flow_strength
                )
    
    def _identify_hot_regions(self) -> List[Tuple[float, float]]:
        """Identify regions with high access activity"""
        regional_activity = defaultdict(int)
        
        for particle in self.particles.values():
            region_x = int(particle.position[0] * 10) / 10.0 + 0.05  # Center of region
            region_y = int(particle.position[1] * 10) / 10.0 + 0.05
            region_key = (region_x, region_y)
            regional_activity[region_key] += particle.access_count
        
        # Sort by activity and return top regions
        sorted_regions = sorted(regional_activity.items(), key=lambda x: x[1], reverse=True)
        return [region for region, activity in sorted_regions[:10]]
    
    def _is_in_hot_region(self, position: Tuple[float, float]) -> bool:
        """Check if position is in a hot region"""
        hot_regions = self._identify_hot_regions()
        
        for hot_region in hot_regions[:3]:  # Top 3 hot regions
            dist = math.sqrt(
                (position[0] - hot_region[0])**2 +
                (position[1] - hot_region[1])**2
            )
            if dist < 0.1:  # Within hot region radius
                return True
        
        return False
    
    def _find_cold_region(self) -> Tuple[float, float]:
        """Find a cold region with low activity"""
        # Simple strategy: find area with fewest particles
        min_density = float('inf')
        best_position = (np.random.random(), np.random.random())
        
        # Sample random positions and find least dense
        for _ in range(20):
            candidate = (np.random.random(), np.random.random())
            density = sum(1 for p in self.particles.values()
                         if math.sqrt((p.position[0] - candidate[0])**2 + 
                                    (p.position[1] - candidate[1])**2) < 0.2)
            
            if density < min_density:
                min_density = density
                best_position = candidate
        
        return best_position
    
    def _flow_simulation_loop(self):
        """Background thread for flow simulation"""
        while self.simulation_active:
            try:
                current_time = time.time()
                
                # Update particle aging and flow
                for particle in self.particles.values():
                    # Age particles (increase recency)
                    age_delta = (current_time - particle.last_access.timestamp()) / 3600.0  # Hours
                    particle.recency = min(1.0, age_delta / 24.0)  # Normalize to days
                    
                    # Update volatility based on access pattern
                    if particle.access_count > 1:
                        access_frequency = particle.access_count / max(age_delta, 0.01)
                        particle.volatility = min(1.0, access_frequency / 10.0)
                    
                    # Move particle
                    self._move_particle_in_flow(particle)
                    
                    # Decay waves
                    particle.wave_amplitude *= 0.95
                    particle.wave_frequency *= 0.98
                    particle.turbulence *= 0.9
                
                # Update rider monitoring
                if self.flow_updates % 10 == 0:  # Every second
                    self.rider.monitor_cache_health()
                
                self.flow_updates += 1
                self.last_simulation_time = current_time
                
                time.sleep(self.simulation_interval)
                
            except Exception as e:
                logger.error(f"🌊 Flow simulation error: {e}")
                time.sleep(1.0)
    
    def __del__(self):
        """Cleanup when ocean is destroyed"""
        self.simulation_active = False


# Global CARRIN instance
_carrin_ocean = None

def get_carrin_ocean(config: Dict[str, Any] = None) -> CARRINOceanicHashMap:
    """Get the global CARRIN oceanic hash map instance"""
    global _carrin_ocean
    if _carrin_ocean is None:
        _carrin_ocean = CARRINOceanicHashMap(config)
    return _carrin_ocean


# Example usage and testing
if __name__ == "__main__":
    print("🌊 Testing CARRIN Oceanic Hash Map")
    print("=" * 50)
    
    # Create ocean
    ocean = CARRINOceanicHashMap()
    
    # Add some data
    ocean.put("user_123", {"name": "Alice", "age": 30}, Priority.HIGH)
    ocean.put("session_456", {"token": "abc123", "expires": "2025-01-01"}, Priority.NORMAL)
    ocean.put("cache_789", {"computed_result": 42}, Priority.LOW)
    
    # Create a current channel
    ocean.create_current_channel(
        "high_priority_channel",
        start_region=(0.2, 0.2),
        end_region=(0.8, 0.8),
        flow_strength=1.5
    )
    
    # Access data (creates waves)
    print(f"Retrieved user: {ocean.get('user_123')}")
    print(f"Retrieved session: {ocean.get('session_456')}")
    print(f"Retrieved cache: {ocean.get('cache_789')}")
    print(f"Missing key: {ocean.get('nonexistent')}")
    
    # Wait for some flow simulation
    time.sleep(0.5)
    
    # Get ocean state
    state = ocean.get_ocean_state()
    print(f"\nOcean State:")
    print(f"  Total particles: {state['ocean_metrics']['total_particles']}")
    print(f"  Cache hit ratio: {state['ocean_metrics']['cache_hit_ratio']:.2f}")
    print(f"  Average turbulence: {state['ocean_metrics']['average_turbulence']:.3f}")
    print(f"  Rider position: {state['rider_status']['position']}")
    
    # Trigger optimization
    optimization = ocean.optimize_flows()
    print(f"\nOptimization results: {optimization}")
    
    # Trigger turbulence
    ocean.trigger_turbulence(region=(0.5, 0.5), intensity=0.7)
    
    print("\n🌊 CARRIN Ocean is flowing - Be water, my friend! 🏄")
    
    # Clean shutdown
    ocean.simulation_active = False
