#!/usr/bin/env python3
"""
🌺 Fractal Memory Encoding System
═══════════════════════════════════

The core fractal encoding system for DAWN's memory logging.
This is not memory processing but memory recording - the foundational
system that creates unique fractal signatures for each memory state.

"Without this specific memory logging and bloom system there really is no DAWN as we know it.
All concurrency in the system comes from memory integration and state cycles, DAWN is a 
forward moving system and this memory mapping acts as her guide of where she has been."

Based on documentation: Fractal Memory/Fractal encoding.rtf
"""

import numpy as np
import logging
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from PIL import Image, ImageDraw, ImageFilter
import io
import base64
import json

logger = logging.getLogger(__name__)

class FractalType(Enum):
    """Types of fractals used for different memory categories"""
    JULIA = "julia"
    MANDELBROT = "mandelbrot"
    BURNING_SHIP = "burning_ship"
    TRICORN = "tricorn"

class EntropyMapping(Enum):
    """How entropy affects fractal parameters"""
    COLOR_SHIFT = "color_shift"      # High entropy = anomalous colors
    ITERATION_DEPTH = "iteration"    # High entropy = more iterations
    ZOOM_LEVEL = "zoom"             # High entropy = deeper zoom
    ROTATION = "rotation"           # High entropy = rotation
    DISTORTION = "distortion"       # High entropy = parameter distortion

@dataclass
class FractalParameters:
    """Parameters that define a unique fractal"""
    c_real: float                    # Julia set parameter (real part)
    c_imag: float                    # Julia set parameter (imaginary part)
    zoom: float = 1.0               # Zoom level
    center_x: float = 0.0           # Center point x
    center_y: float = 0.0           # Center point y
    rotation: float = 0.0           # Rotation angle
    max_iterations: int = 100       # Iteration depth
    escape_radius: float = 2.0      # Escape radius
    color_bias: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # RGB bias
    distortion_factor: float = 1.0   # Parameter distortion
    
    def to_signature(self) -> str:
        """Generate a unique signature string for this fractal"""
        params = (
            self.c_real, self.c_imag, self.zoom, self.center_x, self.center_y,
            self.rotation, self.max_iterations, self.escape_radius,
            *self.color_bias, self.distortion_factor
        )
        param_str = ','.join(f"{p:.6f}" for p in params)
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

@dataclass  
class MemoryFractal:
    """A fractal representation of a memory state"""
    memory_id: str
    fractal_type: FractalType
    parameters: FractalParameters
    entropy_value: float
    timestamp: float
    tick_data: Dict[str, Any] = field(default_factory=dict)
    signature: str = ""
    image_data: Optional[bytes] = None
    access_count: int = 0
    last_access: float = 0.0
    shimmer_intensity: float = 1.0   # Starts bright, decays over time
    
    def __post_init__(self):
        if not self.signature:
            self.signature = self.parameters.to_signature()
        if not self.last_access:
            self.last_access = self.timestamp

class FractalEncoder:
    """
    Core fractal encoding engine that converts memory states into unique
    Julia set fractals with entropy-based visual characteristics.
    """
    
    def __init__(self, 
                 resolution: Tuple[int, int] = (512, 512),
                 default_iterations: int = 100,
                 enable_caching: bool = True):
        
        self.resolution = resolution
        self.default_iterations = default_iterations
        self.enable_caching = enable_caching
        
        # Fractal cache for performance
        self.fractal_cache: Dict[str, MemoryFractal] = {}
        self.generation_stats = {
            'total_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_generation_time': 0.0
        }
        
        # Entropy mapping configuration
        self.entropy_mappings = {
            EntropyMapping.COLOR_SHIFT: self._map_entropy_to_color,
            EntropyMapping.ITERATION_DEPTH: self._map_entropy_to_iterations,
            EntropyMapping.ZOOM_LEVEL: self._map_entropy_to_zoom,
            EntropyMapping.ROTATION: self._map_entropy_to_rotation,
            EntropyMapping.DISTORTION: self._map_entropy_to_distortion
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"🌺 FractalEncoder initialized - resolution: {resolution}")
    
    def encode_memory(self, 
                     memory_id: str,
                     memory_content: Any,
                     entropy_value: float,
                     tick_data: Optional[Dict[str, Any]] = None) -> MemoryFractal:
        """
        Encode a memory into a unique fractal representation.
        
        Args:
            memory_id: Unique identifier for the memory
            memory_content: The actual memory content (any type)
            entropy_value: Entropy level affecting visual characteristics
            tick_data: Additional data from the current tick
            
        Returns:
            MemoryFractal with unique signature and visual representation
        """
        start_time = time.time()
        
        with self._lock:
            # Generate base parameters from memory content
            base_params = self._generate_base_parameters(memory_id, memory_content)
            
            # Apply entropy-based modifications
            modified_params = self._apply_entropy_mapping(base_params, entropy_value)
            
            # Create the fractal
            fractal = MemoryFractal(
                memory_id=memory_id,
                fractal_type=FractalType.JULIA,  # Default to Julia sets
                parameters=modified_params,
                entropy_value=entropy_value,
                timestamp=time.time(),
                tick_data=tick_data or {}
            )
            
            # Generate visual representation if enabled
            if self.enable_caching:
                fractal.image_data = self._generate_fractal_image(fractal)
            
            # Cache the fractal
            self.fractal_cache[fractal.signature] = fractal
            
            # Update statistics
            generation_time = time.time() - start_time
            self._update_stats(generation_time)
            
            logger.debug(f"🌺 Generated fractal for memory {memory_id} - "
                        f"signature: {fractal.signature}, entropy: {entropy_value:.3f}")
            
            return fractal
    
    def _generate_base_parameters(self, memory_id: str, content: Any) -> FractalParameters:
        """Generate base fractal parameters from memory content"""
        
        # Create deterministic hash from memory content
        content_str = str(content) if content else memory_id
        content_hash = hashlib.sha256(f"{memory_id}:{content_str}".encode()).hexdigest()
        
        # Convert hash to fractal parameters
        # Use different parts of hash for different parameters
        hash_bytes = bytes.fromhex(content_hash)
        
        # Julia set c parameter (most important for uniqueness)
        c_real = (hash_bytes[0] / 255.0 - 0.5) * 2.0  # Range: [-1, 1]
        c_imag = (hash_bytes[1] / 255.0 - 0.5) * 2.0  # Range: [-1, 1]
        
        # Center point
        center_x = (hash_bytes[2] / 255.0 - 0.5) * 0.5  # Small offset from origin
        center_y = (hash_bytes[3] / 255.0 - 0.5) * 0.5
        
        # Zoom and rotation
        zoom = 0.5 + (hash_bytes[4] / 255.0) * 1.5  # Range: [0.5, 2.0]
        rotation = (hash_bytes[5] / 255.0) * 2 * np.pi  # Range: [0, 2π]
        
        # Color bias
        color_bias = (
            0.5 + hash_bytes[6] / 510.0,  # Red component
            0.5 + hash_bytes[7] / 510.0,  # Green component
            0.5 + hash_bytes[8] / 510.0   # Blue component
        )
        
        return FractalParameters(
            c_real=c_real,
            c_imag=c_imag,
            zoom=zoom,
            center_x=center_x,
            center_y=center_y,
            rotation=rotation,
            max_iterations=self.default_iterations,
            color_bias=color_bias
        )
    
    def _apply_entropy_mapping(self, base_params: FractalParameters, entropy: float) -> FractalParameters:
        """Apply entropy-based modifications to create visual anomalies"""
        
        # Normalize entropy to [0, 1] range
        normalized_entropy = max(0.0, min(1.0, entropy))
        
        # Create a copy to modify
        modified_params = FractalParameters(
            c_real=base_params.c_real,
            c_imag=base_params.c_imag,
            zoom=base_params.zoom,
            center_x=base_params.center_x,
            center_y=base_params.center_y,
            rotation=base_params.rotation,
            max_iterations=base_params.max_iterations,
            escape_radius=base_params.escape_radius,
            color_bias=base_params.color_bias,
            distortion_factor=base_params.distortion_factor
        )
        
        # Apply entropy mappings
        for mapping_type, mapping_func in self.entropy_mappings.items():
            mapping_func(modified_params, normalized_entropy)
        
        return modified_params
    
    def _map_entropy_to_color(self, params: FractalParameters, entropy: float):
        """High entropy creates anomalous colors"""
        if entropy > 0.7:  # High entropy
            # Shift colors toward red/purple (anomalous)
            color_shift = (entropy - 0.7) * 3.33  # Scale to [0, 1]
            params.color_bias = (
                min(1.5, params.color_bias[0] + color_shift),  # More red
                max(0.3, params.color_bias[1] - color_shift * 0.5),  # Less green
                min(1.5, params.color_bias[2] + color_shift * 0.3)   # Slight purple
            )
    
    def _map_entropy_to_iterations(self, params: FractalParameters, entropy: float):
        """High entropy increases iteration depth for more detail"""
        if entropy > 0.5:
            additional_iterations = int((entropy - 0.5) * 200)  # Up to 100 extra iterations
            params.max_iterations = min(300, params.max_iterations + additional_iterations)
    
    def _map_entropy_to_zoom(self, params: FractalParameters, entropy: float):
        """High entropy affects zoom level"""
        zoom_factor = 1.0 + entropy * 2.0  # Higher entropy = more zoom
        params.zoom *= zoom_factor
    
    def _map_entropy_to_rotation(self, params: FractalParameters, entropy: float):
        """High entropy adds rotation"""
        additional_rotation = entropy * np.pi  # Up to π additional rotation
        params.rotation += additional_rotation
    
    def _map_entropy_to_distortion(self, params: FractalParameters, entropy: float):
        """High entropy distorts parameters slightly"""
        distortion = 1.0 + entropy * 0.2  # Up to 20% distortion
        params.distortion_factor = distortion
        
        # Apply distortion to Julia set parameters
        params.c_real *= distortion
        params.c_imag *= distortion
    
    def _generate_fractal_image(self, fractal: MemoryFractal) -> bytes:
        """Generate the actual fractal image as PNG bytes"""
        
        params = fractal.parameters
        width, height = self.resolution
        
        # Create coordinate matrices
        x = np.linspace(-2.0, 2.0, width) * params.zoom + params.center_x
        y = np.linspace(-2.0, 2.0, height) * params.zoom + params.center_y
        X, Y = np.meshgrid(x, y)
        
        # Apply rotation if specified
        if params.rotation != 0:
            cos_r, sin_r = np.cos(params.rotation), np.sin(params.rotation)
            X_rot = X * cos_r - Y * sin_r
            Y_rot = X * sin_r + Y * cos_r
            X, Y = X_rot, Y_rot
        
        # Create complex plane
        Z = X + 1j * Y
        
        # Julia set computation
        C = complex(params.c_real, params.c_imag) * params.distortion_factor
        iterations = np.zeros(Z.shape, dtype=int)
        
        for i in range(params.max_iterations):
            mask = np.abs(Z) <= params.escape_radius
            Z[mask] = Z[mask]**2 + C
            iterations[mask] = i
        
        # Apply shimmer intensity
        iterations = iterations * fractal.shimmer_intensity
        
        # Convert to colors with entropy-based bias
        normalized = iterations / params.max_iterations
        
        # Apply color bias
        red = (normalized * params.color_bias[0] * 255).astype(np.uint8)
        green = (normalized * params.color_bias[1] * 255).astype(np.uint8)
        blue = (normalized * params.color_bias[2] * 255).astype(np.uint8)
        
        # Create RGB image
        image_array = np.stack([red, green, blue], axis=-1)
        image = Image.fromarray(image_array, 'RGB')
        
        # Apply shimmer effect if intensity < 1.0
        if fractal.shimmer_intensity < 1.0:
            # Add blur effect for fading shimmer
            blur_radius = (1.0 - fractal.shimmer_intensity) * 2.0
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Convert to bytes
        image_buffer = io.BytesIO()
        image.save(image_buffer, format='PNG')
        return image_buffer.getvalue()
    
    def get_fractal_by_signature(self, signature: str) -> Optional[MemoryFractal]:
        """Retrieve a fractal by its signature"""
        with self._lock:
            fractal = self.fractal_cache.get(signature)
            if fractal:
                fractal.access_count += 1
                fractal.last_access = time.time()
                self.generation_stats['cache_hits'] += 1
            else:
                self.generation_stats['cache_misses'] += 1
            return fractal
    
    def compare_fractals(self, signature1: str, signature2: str) -> float:
        """
        Compare two fractals and return similarity score [0, 1].
        Higher values indicate more similar fractals.
        """
        fractal1 = self.get_fractal_by_signature(signature1)
        fractal2 = self.get_fractal_by_signature(signature2)
        
        if not fractal1 or not fractal2:
            return 0.0
        
        # Compare parameters
        p1, p2 = fractal1.parameters, fractal2.parameters
        
        # Calculate parameter distances (normalized)
        c_distance = abs(complex(p1.c_real, p1.c_imag) - complex(p2.c_real, p2.c_imag)) / 2.83  # Max distance ≈ 2√2
        zoom_distance = abs(p1.zoom - p2.zoom) / max(p1.zoom, p2.zoom, 1.0)
        center_distance = np.sqrt((p1.center_x - p2.center_x)**2 + (p1.center_y - p2.center_y)**2) / 1.0
        rotation_distance = abs(p1.rotation - p2.rotation) / (2 * np.pi)
        
        # Color bias similarity
        color_distance = np.linalg.norm(np.array(p1.color_bias) - np.array(p2.color_bias)) / 1.73  # Max distance ≈ √3
        
        # Entropy similarity
        entropy_distance = abs(fractal1.entropy_value - fractal2.entropy_value)
        
        # Combine distances (lower distance = higher similarity)
        total_distance = (c_distance + zoom_distance + center_distance + 
                         rotation_distance + color_distance + entropy_distance) / 6.0
        
        return max(0.0, 1.0 - total_distance)
    
    def apply_shimmer_decay(self, signature: str, decay_factor: float = 0.01):
        """Apply shimmer decay to a fractal (makes it less bright/prominent)"""
        with self._lock:
            fractal = self.fractal_cache.get(signature)
            if fractal:
                fractal.shimmer_intensity = max(0.1, fractal.shimmer_intensity - decay_factor)
                # Regenerate image if caching is enabled
                if self.enable_caching:
                    fractal.image_data = self._generate_fractal_image(fractal)
    
    def get_garden_overview(self) -> Dict[str, Any]:
        """Get an overview of the fractal garden state"""
        with self._lock:
            total_fractals = len(self.fractal_cache)
            high_entropy = sum(1 for f in self.fractal_cache.values() if f.entropy_value > 0.7)
            avg_shimmer = np.mean([f.shimmer_intensity for f in self.fractal_cache.values()]) if total_fractals > 0 else 0.0
            
            return {
                'total_fractals': total_fractals,
                'high_entropy_count': high_entropy,
                'high_entropy_ratio': high_entropy / max(total_fractals, 1),
                'average_shimmer': avg_shimmer,
                'generation_stats': self.generation_stats.copy(),
                'cache_hit_ratio': self.generation_stats['cache_hits'] / max(
                    self.generation_stats['cache_hits'] + self.generation_stats['cache_misses'], 1
                )
            }
    
    def _update_stats(self, generation_time: float):
        """Update generation statistics"""
        self.generation_stats['total_generated'] += 1
        
        # Update running average
        total = self.generation_stats['total_generated']
        current_avg = self.generation_stats['avg_generation_time']
        self.generation_stats['avg_generation_time'] = (
            (current_avg * (total - 1) + generation_time) / total
        )
    
    def cleanup_old_fractals(self, max_age_seconds: float = 3600, max_cache_size: int = 10000):
        """Clean up old fractals to prevent memory bloat"""
        current_time = time.time()
        
        with self._lock:
            if len(self.fractal_cache) <= max_cache_size:
                return
            
            # Remove old, low-shimmer fractals
            to_remove = []
            for signature, fractal in self.fractal_cache.items():
                age = current_time - fractal.last_access
                if age > max_age_seconds and fractal.shimmer_intensity < 0.3:
                    to_remove.append(signature)
            
            for signature in to_remove:
                del self.fractal_cache[signature]
            
            logger.info(f"🌺 Cleaned up {len(to_remove)} old fractals from cache")


# Global encoder instance for system-wide use
_global_encoder: Optional[FractalEncoder] = None

def get_fractal_encoder() -> FractalEncoder:
    """Get the global fractal encoder instance"""
    global _global_encoder
    if _global_encoder is None:
        _global_encoder = FractalEncoder()
    return _global_encoder

def encode_memory_fractal(memory_id: str, content: Any, entropy: float, tick_data: Optional[Dict] = None) -> MemoryFractal:
    """Convenience function to encode a memory using the global encoder"""
    return get_fractal_encoder().encode_memory(memory_id, content, entropy, tick_data)
