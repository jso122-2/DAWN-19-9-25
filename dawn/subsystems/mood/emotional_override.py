"""
DAWN Emotional Override System - Emergency Emotional State Management

Emergency emotional override system for critical system states.
Provides graduated intervention levels based on system conditions.
"""

from typing import Dict, Optional, Any, List
from enum import Enum
from dataclasses import dataclass
import time

class OverrideLevel(Enum):
    """Emotional override intervention levels"""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class OverrideTrigger(Enum):
    """Triggers that can activate emotional overrides"""
    PULSE_SURGE = "pulse_surge"
    THERMAL_CRITICAL = "thermal_critical"
    ENTROPY_OVERFLOW = "entropy_overflow"
    SYSTEM_INSTABILITY = "system_instability"
    MANUAL_ACTIVATION = "manual_activation"

@dataclass
class EmotionalOverride:
    """Represents an active emotional override"""
    level: OverrideLevel
    trigger: OverrideTrigger
    timestamp: float
    duration: Optional[float] = None
    parameters: Dict[str, Any] = None
    
    def is_active(self) -> bool:
        """Check if override is still active"""
        if self.duration is None:
            return True  # Indefinite override
        return (time.time() - self.timestamp) < self.duration

class EmotionalOverrideSystem:
    """Core emotional override management system"""
    
    def __init__(self):
        self.active_overrides: List[EmotionalOverride] = []
        self.override_history = []
        
        # Override parameters by level
        self.override_configs = {
            OverrideLevel.MILD: {
                'arousal_dampening': 0.9,
                'valence_centering': 0.1,
                'entropy_reduction': 0.05,
                'duration': 30.0
            },
            OverrideLevel.MODERATE: {
                'arousal_dampening': 0.7,
                'valence_centering': 0.3,
                'entropy_reduction': 0.2,
                'duration': 60.0
            },
            OverrideLevel.CRITICAL: {
                'arousal_dampening': 0.5,
                'valence_centering': 0.5,
                'entropy_reduction': 0.4,
                'duration': 120.0
            },
            OverrideLevel.EMERGENCY: {
                'arousal_dampening': 0.2,
                'valence_centering': 0.8,
                'entropy_reduction': 0.7,
                'duration': None  # Manual release required
            }
        }
        
        # Constitutional preservation settings
        self.constitutional_limits = {
            'min_kindness': 0.6,
            'min_empathy': 0.5,
            'authenticity_preservation': 0.3
        }
        
        print("[EmotionalOverride] 🚨 Emergency emotional override system initialized")
    
    def assess_override_need(self, system_state: Dict[str, Any]) -> Optional[OverrideLevel]:
        """Assess if emotional override is needed based on system state"""
        # Check pulse zone
        pulse_zone = system_state.get('pulse_zone', 'calm')
        if pulse_zone == "🔴 surge":
            return OverrideLevel.CRITICAL
        elif pulse_zone == "🟡 active":
            return OverrideLevel.MILD
        
        # Check thermal state
        thermal_state = system_state.get('thermal_activity', 0.5)
        if thermal_state > 0.9:
            return OverrideLevel.MODERATE
        
        # Check entropy levels
        entropy_level = system_state.get('entropy', 0.5)
        if entropy_level > 0.9:
            return OverrideLevel.MODERATE
        elif entropy_level > 0.95:
            return OverrideLevel.CRITICAL
        
        # Check system instability
        instability = system_state.get('instability_score', 0.0)
        if instability > 0.8:
            return OverrideLevel.CRITICAL
        elif instability > 0.9:
            return OverrideLevel.EMERGENCY
        
        return None
    
    def activate_override(self, level: OverrideLevel, trigger: OverrideTrigger,
                         custom_params: Optional[Dict[str, Any]] = None) -> EmotionalOverride:
        """Activate an emotional override"""
        base_params = self.override_configs[level].copy()
        if custom_params:
            base_params.update(custom_params)
        
        override = EmotionalOverride(
            level=level,
            trigger=trigger,
            timestamp=time.time(),
            duration=base_params.get('duration'),
            parameters=base_params
        )
        
        self.active_overrides.append(override)
        self.override_history.append({
            'action': 'activate',
            'override': override,
            'timestamp': time.time()
        })
        
        print(f"[EmotionalOverride] 🚨 {level.value.upper()} override activated - trigger: {trigger.value}")
        return override
    
    def apply_overrides(self, mood_state: Dict[str, float]) -> Dict[str, float]:
        """Apply all active overrides to mood state"""
        self._cleanup_expired_overrides()
        
        if not self.active_overrides:
            return mood_state.copy()
        
        modified_mood = mood_state.copy()
        
        # Apply overrides in order of severity (highest first)
        active_sorted = sorted(self.active_overrides, 
                             key=lambda o: list(OverrideLevel).index(o.level),
                             reverse=True)
        
        for override in active_sorted:
            modified_mood = self._apply_single_override(modified_mood, override)
        
        return modified_mood
    
    def _apply_single_override(self, mood_state: Dict[str, float], 
                             override: EmotionalOverride) -> Dict[str, float]:
        """Apply a single override to mood state"""
        modified = mood_state.copy()
        params = override.parameters
        
        # Apply arousal dampening
        if 'arousal_dampening' in params:
            dampening = params['arousal_dampening']
            baseline_arousal = 0.4
            current_arousal = modified.get('arousal', 0.5)
            modified['arousal'] = (
                baseline_arousal + 
                (current_arousal - baseline_arousal) * dampening
            )
        
        # Apply valence centering
        if 'valence_centering' in params:
            centering = params['valence_centering']
            target_valence = self.constitutional_limits['min_kindness'] - 0.5
            current_valence = modified.get('valence', 0.0)
            modified['valence'] = (
                current_valence * (1 - centering) + 
                target_valence * centering
            )
        
        # Apply entropy reduction
        if 'entropy_reduction' in params:
            reduction = params['entropy_reduction']
            baseline_entropy = 0.3
            current_entropy = modified.get('entropy', 0.5)
            modified['entropy'] = (
                baseline_entropy + 
                (current_entropy - baseline_entropy) * (1 - reduction)
            )
        
        return modified
    
    def _cleanup_expired_overrides(self):
        """Remove expired overrides"""
        active_before = len(self.active_overrides)
        self.active_overrides = [o for o in self.active_overrides if o.is_active()]
        
        expired_count = active_before - len(self.active_overrides)
        if expired_count > 0:
            print(f"[EmotionalOverride] ⏰ {expired_count} overrides expired")
    
    def get_override_status(self) -> Dict[str, Any]:
        """Get current override system status"""
        self._cleanup_expired_overrides()
        
        return {
            'active_overrides': len(self.active_overrides),
            'override_details': [
                {
                    'level': o.level.value,
                    'trigger': o.trigger.value,
                    'active_time': time.time() - o.timestamp,
                }
                for o in self.active_overrides
            ],
            'highest_active_level': (
                max(self.active_overrides, key=lambda o: list(OverrideLevel).index(o.level)).level.value
                if self.active_overrides else 'none'
            )
        }

# Global override system instance
emotional_override_system = EmotionalOverrideSystem()

# Legacy function for backward compatibility
def apply_emotional_override(pulse_zone, active_bloom):
    """Legacy emotional override function"""
    if pulse_zone == "🔴 surge":
        if hasattr(active_bloom, 'override_flags'):
            active_bloom.override_flags["reroute"] = True
            active_bloom.override_flags["suppress_entropy"] = True
        return "⚠️ Surge override applied"
    elif pulse_zone == "🟡 active":
        if hasattr(active_bloom, 'override_flags'):
            active_bloom.override_flags["drift_softening"] = True
        return "🌗 Drift softened"
    return "🟢 No override"

# Convenience functions
def apply_emotional_overrides(mood_state: Dict[str, float]) -> Dict[str, float]:
    """Apply all active overrides to mood state"""
    return emotional_override_system.apply_overrides(mood_state)

def get_override_status() -> Dict[str, Any]:
    """Get current override status"""
    return emotional_override_system.get_override_status()

# Schema phase tagging
__schema_phase__ = "Emergency-Emotional-Control"
__dawn_signature__ = "🚨 DAWN Emotional Override Guardian"
