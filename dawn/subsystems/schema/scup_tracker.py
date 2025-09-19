"""
DAWN SCUP (Semantic Coherence Under Pressure) Tracker
=====================================================
Advanced SCUP tracking and computation system implementing multiple
calculation methods for comprehensive pressure-aware coherence monitoring.

SCUP represents the probability of coherence loss under cognitive pressure:
p_loss = σ(a·F* + b·P̂ + c·Δdrift + d·τ̄ - e·A - f·SHI)

Where σ(x) = 1/(1+e^(-x)) is the sigmoid function.

Author: DAWN Development Team  
Generated: 2025-09-18
"""

import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading

# Import torch with fallback for systems without PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[SCUPTracker] Warning: PyTorch not available, using numpy fallback")


class SCUPZone(Enum):
    """SCUP zone classifications based on coherence risk"""
    CALM = "calm"           # 🟢 Low risk - SCUP > 0.7
    CREATIVE = "creative"   # 🟡 Moderate risk - 0.5 < SCUP ≤ 0.7
    ACTIVE = "active"       # 🟠 High risk - 0.3 < SCUP ≤ 0.5  
    CRITICAL = "critical"   # 🔴 Critical risk - SCUP ≤ 0.3


@dataclass
class SCUPState:
    """SCUP state management and history tracking"""
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    coherence_buffer: deque = field(default_factory=lambda: deque(maxlen=20))
    recovery_momentum: float = 0.0
    breathing_phase: float = 0.0
    emergency_active: bool = False
    last_scup: float = 0.500
    recovery_count: int = 0
    stability_score: float = 1.0


@dataclass 
class SCUPInputs:
    """Input parameters for SCUP calculation"""
    alignment: float        # Semantic alignment (0-1)
    entropy: float         # System entropy (0-1+)
    pressure: float        # Cognitive pressure (0-1+)
    forecast_index: float = 0.5    # F* - smoothed forecast index
    adaptive_capacity: float = 0.5  # A - system adaptive capacity
    drift_delta: float = 0.0       # Δdrift - forecast vs realized drift
    mean_tension: float = 0.0      # τ̄ - mean tension of hot edges
    shi_value: float = 0.5         # Current SHI value


@dataclass
class SCUPResult:
    """Complete SCUP calculation results"""
    scup: float
    zone: SCUPZone
    stability: float
    recovery_potential: float
    method_used: str
    calculation_time: float
    components: Dict[str, float]
    confidence: float


class SCUPTracker:
    """
    Comprehensive SCUP tracking system with multiple computation methods
    and adaptive coherence monitoring under cognitive pressure
    """
    
    def __init__(self, vault_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.vault_path = vault_path
        self.config = config or {}
        
        # Core state
        self.state = SCUPState()
        
        # SCUP formula coefficients (a, b, c, d, e, f)
        self.coefficients = {
            'forecast_weight': self.config.get('a', 0.3),      # a
            'pressure_weight': self.config.get('b', 0.25),    # b  
            'drift_weight': self.config.get('c', 0.2),        # c
            'tension_weight': self.config.get('d', 0.15),     # d
            'capacity_weight': self.config.get('e', 0.2),     # e
            'shi_weight': self.config.get('f', 0.1)           # f
        }
        
        # Zone thresholds
        self.zone_thresholds = {
            'critical': 0.3,
            'active': 0.5,
            'creative': 0.7
        }
        
        # Performance tracking
        self.computation_count = 0
        self.total_computation_time = 0.0
        self.error_count = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        print("[SCUPTracker] 🧠 SCUP tracking system initialized")
        
    def compute_scup(self, 
                    alignment: float,
                    entropy: float, 
                    pressure: float,
                    method: str = "auto",
                    additional_inputs: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Compute SCUP using specified method
        """
        start_time = time.time()
        
        try:
            with self.lock:
                # Prepare inputs
                inputs = SCUPInputs(
                    alignment=max(0.0, min(1.0, alignment)),
                    entropy=max(0.0, entropy),
                    pressure=max(0.0, pressure)
                )
                
                if additional_inputs:
                    inputs.forecast_index = additional_inputs.get('forecast_index', 0.5)
                    inputs.adaptive_capacity = additional_inputs.get('adaptive_capacity', 0.5)
                    inputs.drift_delta = additional_inputs.get('drift_delta', 0.0)
                    inputs.mean_tension = additional_inputs.get('mean_tension', 0.0)
                    inputs.shi_value = additional_inputs.get('shi_value', 0.5)
                
                # Select method
                if method == "auto":
                    method = self._select_optimal_method(inputs)
                
                # Compute SCUP
                if method == "enhanced":
                    scup_value = self._compute_enhanced_scup(inputs)
                else:
                    scup_value = self._compute_basic_scup(inputs)
                
                # Classify zone
                zone = self._classify_zone(scup_value)
                
                # Calculate metrics
                stability = self._get_stability_score()
                recovery_potential = self._assess_recovery_potential(scup_value, inputs)
                confidence = self._calculate_confidence(scup_value, stability, method)
                
                # Update state
                self.state.history.append({
                    'timestamp': time.time(),
                    'scup': scup_value,
                    'zone': zone
                })
                self.state.last_scup = scup_value
                
                # Track performance
                calculation_time = time.time() - start_time
                self.computation_count += 1
                self.total_computation_time += calculation_time
                
                return {
                    'scup': scup_value,
                    'zone': zone.value,
                    'stability': stability,
                    'recovery_potential': recovery_potential,
                    'method_used': method,
                    'confidence': confidence
                }
                
        except Exception as e:
            self.error_count += 1
            print(f"[SCUPTracker] ❌ Error in compute_scup: {str(e)}")
            return {'scup': 0.0, 'zone': 'critical', 'stability': 0.0}
    
    def _select_optimal_method(self, inputs: SCUPInputs) -> str:
        """Select optimal computation method"""
        if inputs.entropy > 0.7 or inputs.pressure > 0.8:
            return "enhanced"
        return "basic"
    
    def _compute_basic_scup(self, inputs: SCUPInputs) -> float:
        """Basic SCUP computation"""
        pressure_entropy_factor = inputs.entropy * inputs.pressure
        alignment_factor = inputs.alignment + 0.1
        
        scup = 1.0 - (pressure_entropy_factor / (alignment_factor * 2.0))
        return max(0.0, min(1.0, scup))
    
    def _compute_enhanced_scup(self, inputs: SCUPInputs) -> float:
        """Enhanced SCUP using full mathematical formula"""
        P_hat = inputs.pressure / max(1.0, inputs.pressure + 0.5)
        
        linear_term = (
            self.coefficients['forecast_weight'] * inputs.forecast_index +
            self.coefficients['pressure_weight'] * P_hat +
            self.coefficients['drift_weight'] * abs(inputs.drift_delta) +
            self.coefficients['tension_weight'] * inputs.mean_tension -
            self.coefficients['capacity_weight'] * inputs.adaptive_capacity -
            self.coefficients['shi_weight'] * inputs.shi_value
        )
        
        # Apply sigmoid: σ(x) = 1/(1+e^(-x))
        p_loss = 1.0 / (1.0 + math.exp(-linear_term))
        scup = 1.0 - p_loss
        
        return max(0.0, min(1.0, scup))
    
    def _classify_zone(self, scup: float) -> SCUPZone:
        """Classify SCUP into zone"""
        if scup > self.zone_thresholds['creative']:
            return SCUPZone.CALM
        elif scup > self.zone_thresholds['active']:
            return SCUPZone.CREATIVE
        elif scup > self.zone_thresholds['critical']:
            return SCUPZone.ACTIVE
        else:
            return SCUPZone.CRITICAL
    
    def _get_stability_score(self) -> float:
        """Calculate stability from recent coherence variance"""
        if len(self.state.coherence_buffer) < 3:
            return 1.0
        variance = np.var(list(self.state.coherence_buffer))
        return max(0.0, 1.0 - variance * 2.0)
    
    def _assess_recovery_potential(self, scup: float, inputs: SCUPInputs) -> float:
        """Assess recovery potential"""
        base_potential = inputs.adaptive_capacity
        shi_boost = max(0.0, inputs.shi_value - 0.5) * 0.4
        pressure_penalty = inputs.pressure * 0.3
        
        recovery_potential = base_potential + shi_boost - pressure_penalty
        return max(0.0, min(1.0, recovery_potential))
    
    def _calculate_confidence(self, scup: float, stability: float, method: str) -> float:
        """Calculate confidence in calculation"""
        method_confidence = {'basic': 0.7, 'enhanced': 0.9}.get(method, 0.7)
        confidence = (stability + method_confidence) / 2.0
        return max(0.0, min(1.0, confidence))


def create_scup_tracker(config: Optional[Dict[str, Any]] = None) -> SCUPTracker:
    """Create configured SCUP tracker"""
    return SCUPTracker(config=config)


if __name__ == "__main__":
    tracker = create_scup_tracker()
    
    result = tracker.compute_scup(
        alignment=0.7,
        entropy=0.4,
        pressure=0.3,
        method="enhanced"
    )
    
    print(f"SCUP: {result['scup']:.3f} ({result['zone']})")