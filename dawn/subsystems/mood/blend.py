
"""
DAWN Mood Blending and Interpolation System
Smooth mood state transitions and multi-dimensional blending
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter

# Mood archetype hierarchy for blending priority
MOOD_ARCHETYPE_HIERARCHY = [
    "Crystalline", "Sharp Edge", "Contained Burn", 
    "Submerged", "Drifting", "Hollow Echo"
]

# Mood blend priority table (legacy support)
MOOD_HIERARCHY = [
    "joyful", "focused", "reflective", "anxious", "sad"
]

def blend_mood_vectors(mood_states: List[Dict[str, float]], 
                      weights: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Blend multiple mood vectors with optional weighting
    
    Args:
        mood_states: List of mood dictionaries with arousal, valence, entropy, etc.
        weights: Optional weights for each mood state (normalized internally)
        
    Returns:
        Blended mood state dictionary
    """
    if not mood_states:
        return {'arousal': 0.5, 'valence': 0.0, 'entropy': 0.5, 'dominance': 0.5}
    
    if len(mood_states) == 1:
        return mood_states[0].copy()
    
    # Normalize weights
    if weights is None:
        weights = [1.0] * len(mood_states)
    else:
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize to sum to 1
    
    # Extract common mood dimensions
    dimensions = ['arousal', 'valence', 'entropy', 'dominance']
    blended = {}
    
    for dim in dimensions:
        values = []
        valid_weights = []
        
        for i, mood in enumerate(mood_states):
            if dim in mood:
                values.append(mood[dim])
                valid_weights.append(weights[i])
        
        if values:
            # Weighted average
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / np.sum(valid_weights)
            blended[dim] = np.average(values, weights=valid_weights)
        else:
            # Default values
            defaults = {'arousal': 0.5, 'valence': 0.0, 'entropy': 0.5, 'dominance': 0.5}
            blended[dim] = defaults.get(dim, 0.5)
    
    return blended

def interpolate_mood_transition(start_mood: Dict[str, float], 
                               end_mood: Dict[str, float],
                               progress: float,
                               interpolation_type: str = 'linear') -> Dict[str, float]:
    """
    Interpolate between two mood states
    
    Args:
        start_mood: Starting mood state
        end_mood: Target mood state  
        progress: Interpolation progress [0.0, 1.0]
        interpolation_type: 'linear', 'smooth', 'easeout'
        
    Returns:
        Interpolated mood state
    """
    progress = np.clip(progress, 0.0, 1.0)
    
    # Apply interpolation curve
    if interpolation_type == 'smooth':
        # Smooth step function
        progress = progress * progress * (3.0 - 2.0 * progress)
    elif interpolation_type == 'easeout':
        # Ease-out cubic
        progress = 1 - (1 - progress) ** 3
    # 'linear' uses progress as-is
    
    # Interpolate each dimension
    interpolated = {}
    all_keys = set(start_mood.keys()) | set(end_mood.keys())
    
    for key in all_keys:
        start_val = start_mood.get(key, 0.5)
        end_val = end_mood.get(key, 0.5)
        interpolated[key] = start_val + (end_val - start_val) * progress
    
    return interpolated

def blend_mood_archetypes(archetype_scores: Dict[str, float]) -> str:
    """
    Blend mood archetype scores to determine dominant archetype
    
    Args:
        archetype_scores: Dictionary of archetype names to scores
        
    Returns:
        Dominant archetype name
    """
    if not archetype_scores:
        return "Crystalline"  # Default stable state
    
    # Find archetype with highest score
    max_score = max(archetype_scores.values())
    candidates = [arch for arch, score in archetype_scores.items() if score == max_score]
    
    # If tie, use hierarchy priority
    for arch in MOOD_ARCHETYPE_HIERARCHY:
        if arch in candidates:
            return arch
    
    # Fallback to first candidate
    return candidates[0] if candidates else "Crystalline"

def calculate_mood_distance(mood1: Dict[str, float], mood2: Dict[str, float]) -> float:
    """
    Calculate Euclidean distance between two mood states
    
    Args:
        mood1, mood2: Mood state dictionaries
        
    Returns:
        Distance between mood states
    """
    dimensions = ['arousal', 'valence', 'entropy', 'dominance']
    distance_squared = 0.0
    
    for dim in dimensions:
        val1 = mood1.get(dim, 0.5)
        val2 = mood2.get(dim, 0.5)
        distance_squared += (val1 - val2) ** 2
    
    return np.sqrt(distance_squared)

def smooth_mood_trajectory(mood_history: List[Dict[str, float]], 
                          smoothing_factor: float = 0.3) -> List[Dict[str, float]]:
    """
    Apply smoothing to a sequence of mood states
    
    Args:
        mood_history: List of mood states in temporal order
        smoothing_factor: Amount of smoothing [0.0, 1.0]
        
    Returns:
        Smoothed mood trajectory
    """
    if len(mood_history) <= 2:
        return mood_history.copy()
    
    smoothed = [mood_history[0].copy()]  # Keep first state unchanged
    
    for i in range(1, len(mood_history)):
        current = mood_history[i]
        previous_smoothed = smoothed[i-1]
        
        # Blend current state with smoothed previous state
        blended = blend_mood_vectors([previous_smoothed, current], 
                                   [smoothing_factor, 1.0 - smoothing_factor])
        smoothed.append(blended)
    
    return smoothed

def detect_mood_transitions(mood_history: List[Dict[str, float]], 
                          threshold: float = 0.3) -> List[int]:
    """
    Detect significant mood transitions in history
    
    Args:
        mood_history: List of mood states
        threshold: Minimum distance for transition detection
        
    Returns:
        List of indices where transitions occur
    """
    if len(mood_history) < 2:
        return []
    
    transitions = []
    
    for i in range(1, len(mood_history)):
        distance = calculate_mood_distance(mood_history[i-1], mood_history[i])
        if distance > threshold:
            transitions.append(i)
    
    return transitions

# Legacy function for backward compatibility
def blend_moods(source_blooms):
    """Legacy mood blending function"""
    moods = [bloom.get("mood", "undefined") for bloom in source_blooms]
    mood_counts = Counter(moods)

    if not mood_counts:
        return "undefined"

    # Find highest priority mood by presence in source set
    for mood in MOOD_HIERARCHY:
        if mood in mood_counts:
            return mood

    return moods[0] if moods else "undefined"  # fallback

# Schema phase tagging
__schema_phase__ = "Mood-Blending-Dynamics"
__dawn_signature__ = "🌈 DAWN Mood Interpolation"
