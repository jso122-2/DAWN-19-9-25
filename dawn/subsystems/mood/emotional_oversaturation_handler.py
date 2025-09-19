"""
DAWN Emotional Oversaturation Handler - Emotional Overflow & Stability Management

This module detects and manages emotional oversaturation states where mood
dynamics become unstable or extreme. It implements adaptive correction
mechanisms to maintain emotional coherence while preserving authentic
emotional expression.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time

class OversaturationLevel(Enum):
    """Levels of emotional oversaturation"""
    STABLE = "stable"
    ELEVATED = "elevated"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class InterventionType(Enum):
    """Types of stabilization interventions"""
    NONE = "none"
    DAMPENING = "dampening"
    CAPPING = "capping"
    CIRCUIT_BREAKER = "circuit_breaker"

@dataclass
class OversaturationReading:
    """A single oversaturation assessment"""
    timestamp: float
    arousal_level: float
    valence_level: float
    entropy_level: float
    momentum: float
    instability_score: float
    saturation_level: OversaturationLevel
    recommended_intervention: InterventionType
    
    # Component scores
    arousal_saturation: float = 0.0
    valence_saturation: float = 0.0
    entropy_saturation: float = 0.0
    momentum_saturation: float = 0.0
    
    # Intervention parameters
    dampening_factor: float = 1.0
    cap_values: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'instability_score': self.instability_score,
            'saturation_level': self.saturation_level.value,
            'intervention': self.recommended_intervention.value,
            'component_scores': {
                'arousal': self.arousal_saturation,
                'valence': self.valence_saturation,
                'entropy': self.entropy_saturation,
                'momentum': self.momentum_saturation
            },
            'dampening_factor': self.dampening_factor,
            'cap_values': self.cap_values
        }

class EmotionalOversaturationHandler:
    """
    Core handler for emotional oversaturation detection and management
    """
    
    def __init__(self, memory_window: int = 200):
        self.memory_window = memory_window
        
        # Historical tracking
        self.mood_history = deque(maxlen=memory_window)
        self.saturation_history = deque(maxlen=100)
        self.intervention_history = deque(maxlen=50)
        
        # Saturation thresholds
        self.thresholds = {
            'arousal_elevated': 0.85,
            'arousal_critical': 0.95,
            'valence_swing_elevated': 0.7,
            'valence_swing_critical': 0.9,
            'entropy_elevated': 0.8,
            'entropy_critical': 0.9,
            'momentum_elevated': 0.3,
            'momentum_critical': 0.5,
            'instability_elevated': 0.6,
            'instability_critical': 0.8,
            'instability_emergency': 0.95
        }
        
        # Intervention parameters
        self.intervention_params = {
            'dampening_base': 0.9,
            'dampening_critical': 0.7,
            'cap_arousal_max': 0.9,
            'cap_valence_max': 0.8,
            'cap_entropy_max': 0.85
        }
        
        # State tracking
        self.current_level = OversaturationLevel.STABLE
        self.circuit_breaker_active = False
        self.circuit_breaker_timestamp = 0
        self.circuit_breaker_duration = 10.0  # seconds
        
        # Constitutional constraints
        self.constitutional_limits = {
            'kindness_floor': 0.6,
            'empathy_preservation': 0.8,
            'authenticity_threshold': 0.3
        }
        
        print("[OversaturationHandler] 🛡️ Emotional stability guardian initialized")
    
    def assess_oversaturation(self, mood_state: Dict[str, float]) -> OversaturationReading:
        """Assess current emotional oversaturation level"""
        try:
            current_time = time.time()
            self.mood_history.append({
                'timestamp': current_time,
                'arousal': mood_state.get('arousal', 0.5),
                'valence': mood_state.get('valence', 0.0),
                'entropy': mood_state.get('entropy', 0.5)
            })
            
            # Calculate component saturations
            arousal_sat = self._calculate_arousal_saturation(mood_state)
            valence_sat = self._calculate_valence_saturation(mood_state)
            entropy_sat = self._calculate_entropy_saturation(mood_state)
            momentum_sat = self._calculate_momentum_saturation()
            
            # Calculate overall instability score
            instability = self._calculate_instability_score(
                arousal_sat, valence_sat, entropy_sat, momentum_sat
            )
            
            # Determine saturation level and intervention
            saturation_level = self._classify_saturation_level(instability)
            intervention = self._recommend_intervention(saturation_level, arousal_sat, valence_sat, entropy_sat)
            
            # Calculate intervention parameters
            dampening_factor = self._calculate_dampening_factor(saturation_level)
            cap_values = self._calculate_cap_values(saturation_level, mood_state)
            
            # Create reading
            reading = OversaturationReading(
                timestamp=current_time,
                arousal_level=mood_state.get('arousal', 0.5),
                valence_level=mood_state.get('valence', 0.0),
                entropy_level=mood_state.get('entropy', 0.5),
                momentum=momentum_sat,
                instability_score=instability,
                saturation_level=saturation_level,
                recommended_intervention=intervention,
                arousal_saturation=arousal_sat,
                valence_saturation=valence_sat,
                entropy_saturation=entropy_sat,
                momentum_saturation=momentum_sat,
                dampening_factor=dampening_factor,
                cap_values=cap_values
            )
            
            # Update state
            self.saturation_history.append(reading)
            self._update_state(reading)
            
            return reading
            
        except Exception as e:
            print(f"[OversaturationHandler] ❌ Assessment error: {e}")
            return self._create_safe_reading(mood_state)
    
    def _calculate_arousal_saturation(self, mood_state: Dict[str, float]) -> float:
        """Calculate arousal oversaturation score"""
        arousal = mood_state.get('arousal', 0.5)
        base_sat = max(0, (arousal - 0.7) / 0.3)
        
        # Check for sustained high arousal
        if len(self.mood_history) >= 10:
            recent_arousal = [m['arousal'] for m in list(self.mood_history)[-10:]]
            sustained_high = sum(a > 0.8 for a in recent_arousal) / len(recent_arousal)
            base_sat += sustained_high * 0.3
        
        return min(1.0, base_sat)
    
    def _calculate_valence_saturation(self, mood_state: Dict[str, float]) -> float:
        """Calculate valence swing oversaturation score"""
        valence = mood_state.get('valence', 0.0)
        base_sat = max(0, (abs(valence) - 0.6) / 0.4)
        
        # Check for rapid valence swings
        if len(self.mood_history) >= 5:
            recent_valence = [m['valence'] for m in list(self.mood_history)[-5:]]
            valence_range = max(recent_valence) - min(recent_valence)
            swing_sat = max(0, (valence_range - 0.5) / 0.5)
            base_sat += swing_sat * 0.4
        
        return min(1.0, base_sat)
    
    def _calculate_entropy_saturation(self, mood_state: Dict[str, float]) -> float:
        """Calculate entropy oversaturation score"""
        entropy = mood_state.get('entropy', 0.5)
        base_sat = max(0, (entropy - 0.7) / 0.3)
        
        # Check for chaotic patterns
        if len(self.mood_history) >= 15:
            recent = list(self.mood_history)[-15:]
            arousal_changes = np.diff([m['arousal'] for m in recent])
            valence_changes = np.diff([m['valence'] for m in recent])
            volatility = np.mean(np.abs(arousal_changes)) + np.mean(np.abs(valence_changes))
            
            if volatility > 0.2:
                base_sat += (volatility - 0.2) / 0.2 * 0.4
        
        return min(1.0, base_sat)
    
    def _calculate_momentum_saturation(self) -> float:
        """Calculate emotional momentum oversaturation"""
        if len(self.mood_history) < 10:
            return 0.0
        
        recent = list(self.mood_history)[-10:]
        arousal_momentum = abs(recent[-1]['arousal'] - recent[0]['arousal']) / len(recent)
        valence_momentum = abs(recent[-1]['valence'] - recent[0]['valence']) / len(recent)
        entropy_momentum = abs(recent[-1]['entropy'] - recent[0]['entropy']) / len(recent)
        
        total_momentum = arousal_momentum + valence_momentum + entropy_momentum
        momentum_sat = max(0, (total_momentum - 0.1) / 0.3)
        
        return min(1.0, momentum_sat)
    
    def _calculate_instability_score(self, arousal_sat: float, valence_sat: float,
                                   entropy_sat: float, momentum_sat: float) -> float:
        """Calculate overall emotional instability score"""
        instability = (
            arousal_sat * 0.3 +
            valence_sat * 0.25 +
            entropy_sat * 0.25 +
            momentum_sat * 0.2
        )
        
        # Amplify for multiple high components
        high_components = sum([
            arousal_sat > 0.7, valence_sat > 0.7,
            entropy_sat > 0.7, momentum_sat > 0.7
        ])
        
        if high_components >= 2:
            instability *= 1.2
        if high_components >= 3:
            instability *= 1.4
        
        return min(1.0, instability)
    
    def _classify_saturation_level(self, instability_score: float) -> OversaturationLevel:
        """Classify instability score into saturation level"""
        if instability_score >= self.thresholds['instability_emergency']:
            return OversaturationLevel.EMERGENCY
        elif instability_score >= self.thresholds['instability_critical']:
            return OversaturationLevel.CRITICAL
        elif instability_score >= self.thresholds['instability_elevated']:
            return OversaturationLevel.ELEVATED
        else:
            return OversaturationLevel.STABLE
    
    def _recommend_intervention(self, level: OversaturationLevel,
                              arousal_sat: float, valence_sat: float,
                              entropy_sat: float) -> InterventionType:
        """Recommend appropriate intervention"""
        if level == OversaturationLevel.EMERGENCY:
            return InterventionType.CIRCUIT_BREAKER
        elif level == OversaturationLevel.CRITICAL:
            if max(arousal_sat, valence_sat, entropy_sat) > 0.9:
                return InterventionType.CIRCUIT_BREAKER
            else:
                return InterventionType.CAPPING
        elif level == OversaturationLevel.ELEVATED:
            return InterventionType.DAMPENING
        else:
            return InterventionType.NONE
    
    def _calculate_dampening_factor(self, level: OversaturationLevel) -> float:
        """Calculate dampening factor"""
        if level == OversaturationLevel.CRITICAL:
            return self.intervention_params['dampening_critical']
        elif level == OversaturationLevel.ELEVATED:
            return self.intervention_params['dampening_base']
        else:
            return 1.0
    
    def _calculate_cap_values(self, level: OversaturationLevel,
                            mood_state: Dict[str, float]) -> Dict[str, float]:
        """Calculate capping values"""
        caps = {}
        if level in [OversaturationLevel.CRITICAL, OversaturationLevel.EMERGENCY]:
            caps['arousal'] = self.intervention_params['cap_arousal_max']
            caps['valence_positive'] = self.intervention_params['cap_valence_max']
            caps['valence_negative'] = -self.intervention_params['cap_valence_max']
            caps['entropy'] = self.intervention_params['cap_entropy_max']
        return caps
    
    def _update_state(self, reading: OversaturationReading):
        """Update handler state"""
        self.current_level = reading.saturation_level
        
        if reading.recommended_intervention == InterventionType.CIRCUIT_BREAKER:
            if not self.circuit_breaker_active:
                self.circuit_breaker_active = True
                self.circuit_breaker_timestamp = reading.timestamp
                print("[OversaturationHandler] 🚨 CIRCUIT BREAKER ACTIVATED")
    
    def _create_safe_reading(self, mood_state: Dict[str, float]) -> OversaturationReading:
        """Create safe default reading"""
        return OversaturationReading(
            timestamp=time.time(),
            arousal_level=mood_state.get('arousal', 0.5),
            valence_level=mood_state.get('valence', 0.0),
            entropy_level=mood_state.get('entropy', 0.5),
            momentum=0.0,
            instability_score=0.3,
            saturation_level=OversaturationLevel.STABLE,
            recommended_intervention=InterventionType.NONE
        )
    
    def apply_stabilization(self, mood_state: Dict[str, float],
                          reading: OversaturationReading) -> Dict[str, float]:
        """Apply stabilization interventions to mood state"""
        if reading.recommended_intervention == InterventionType.NONE:
            return mood_state.copy()
        
        stabilized = mood_state.copy()
        
        if (reading.recommended_intervention == InterventionType.CIRCUIT_BREAKER or
            self.circuit_breaker_active):
            stabilized = self._apply_circuit_breaker(stabilized)
        elif reading.recommended_intervention == InterventionType.CAPPING:
            stabilized = self._apply_capping(stabilized, reading.cap_values)
        elif reading.recommended_intervention == InterventionType.DAMPENING:
            stabilized = self._apply_dampening(stabilized, reading.dampening_factor)
        
        return stabilized
    
    def _apply_circuit_breaker(self, mood_state: Dict[str, float]) -> Dict[str, float]:
        """Apply emergency circuit breaker"""
        safe_baseline = {
            'arousal': 0.4,
            'valence': 0.1,  # Slight positive
            'entropy': 0.3,
            'dominance': 0.5
        }
        
        stabilized = {}
        for key, current_value in mood_state.items():
            if key in safe_baseline:
                target = safe_baseline[key]
                stabilized[key] = current_value * 0.3 + target * 0.7
            else:
                stabilized[key] = current_value
        
        return stabilized
    
    def _apply_capping(self, mood_state: Dict[str, float],
                      cap_values: Dict[str, float]) -> Dict[str, float]:
        """Apply value capping"""
        stabilized = mood_state.copy()
        
        if 'arousal' in cap_values:
            stabilized['arousal'] = min(stabilized.get('arousal', 0.5), cap_values['arousal'])
        
        if 'valence_positive' in cap_values and stabilized.get('valence', 0) > 0:
            stabilized['valence'] = min(stabilized['valence'], cap_values['valence_positive'])
        
        if 'valence_negative' in cap_values and stabilized.get('valence', 0) < 0:
            stabilized['valence'] = max(stabilized['valence'], cap_values['valence_negative'])
        
        if 'entropy' in cap_values:
            stabilized['entropy'] = min(stabilized.get('entropy', 0.5), cap_values['entropy'])
        
        return stabilized
    
    def _apply_dampening(self, mood_state: Dict[str, float],
                        dampening_factor: float) -> Dict[str, float]:
        """Apply dampening to reduce emotional intensity"""
        stabilized = mood_state.copy()
        
        if 'arousal' in stabilized:
            baseline_arousal = 0.5
            stabilized['arousal'] = (
                baseline_arousal + 
                (stabilized['arousal'] - baseline_arousal) * dampening_factor
            )
        
        if 'valence' in stabilized:
            baseline_valence = 0.1
            valence_delta = stabilized['valence'] - baseline_valence
            
            if abs(valence_delta) > self.constitutional_limits['authenticity_threshold']:
                dampened_delta = valence_delta * dampening_factor
                stabilized['valence'] = baseline_valence + dampened_delta
        
        if 'entropy' in stabilized:
            baseline_entropy = 0.4
            stabilized['entropy'] = (
                baseline_entropy +
                (stabilized['entropy'] - baseline_entropy) * dampening_factor
            )
        
        return stabilized

# Global handler instance
oversaturation_handler = EmotionalOversaturationHandler()

# Convenience functions
def assess_emotional_stability(mood_state: Dict[str, float]) -> OversaturationReading:
    """Assess current emotional stability"""
    return oversaturation_handler.assess_oversaturation(mood_state)

def stabilize_mood_state(mood_state: Dict[str, float]) -> Dict[str, float]:
    """Apply stabilization to mood state if needed"""
    reading = oversaturation_handler.assess_oversaturation(mood_state)
    return oversaturation_handler.apply_stabilization(mood_state, reading)

def is_circuit_breaker_active() -> bool:
    """Check if emotional circuit breaker is active"""
    return oversaturation_handler.circuit_breaker_active

print("[EmotionalOversaturation] 🛡️ DAWN emotional stability guardian initialized")
