"""
A fluid, interconnected mood analysis system that treats emotions as frequencies rather than categories.
Emphasizes resonance and transformation over static classification.
"""

from typing import Dict, List, Tuple, Optional
import re
from collections import Counter

# Core mood archetypes with fluid, overlapping descriptions
MOOD_ARCHETYPES = {
    "Contained Burn": {
        "description": "Pressure building, potential for sharp release",
        "pressure_affinity": ["Sharp Edge"],
        "drift_tendency": ["Submerged", "Crystalline"]
    },
    "Sharp Edge": {
        "description": "Intense, cutting, immediate impact",
        "pressure_affinity": ["Crystalline"],
        "drift_tendency": ["Contained Burn", "Hollow Echo"]
    },
    "Submerged": {
        "description": "Deep, flowing, potentially overwhelming",
        "pressure_affinity": ["Drifting"],
        "drift_tendency": ["Contained Burn", "Hollow Echo"]
    },
    "Drifting": {
        "description": "Floating, uncertain, transitional",
        "pressure_affinity": ["Hollow Echo"],
        "drift_tendency": ["Submerged", "Crystalline"]
    },
    "Hollow Echo": {
        "description": "Empty, resonant, potentially transformative",
        "pressure_affinity": ["Crystalline"],
        "drift_tendency": ["Drifting", "Sharp Edge"]
    },
    "Crystalline": {
        "description": "Clear, structured, potentially fragile",
        "pressure_affinity": ["Sharp Edge"],
        "drift_tendency": ["Drifting", "Contained Burn"]
    }
}

# Emotional resonance mapping with overlapping categories
EMOTIONAL_RESONANCE = {
    "Contained Burn": {
        "words": ["pressure", "build", "contain", "drowning", "trapped", "burn"],
        "weight": 1.0
    },
    "Sharp Edge": {
        "words": ["cut", "sharp", "break", "shatter", "intense", "immediate"],
        "weight": 1.2
    },
    "Submerged": {
        "words": ["deep", "flow", "drowning", "under", "sink", "drown"],
        "weight": 0.9
    },
    "Drifting": {
        "words": ["float", "maybe", "perhaps", "uncertain", "flow", "drift"],
        "weight": 0.8
    },
    "Hollow Echo": {
        "words": ["empty", "echo", "hollow", "void", "nothing", "silence"],
        "weight": 1.1
    },
    "Crystalline": {
        "words": ["clear", "structure", "fragile", "break", "sharp", "pure"],
        "weight": 1.0
    }
}

# Intensity markers that can shift moods
INTENSITY_MARKERS = {
    "exclamation": {
        "symbol": "!",
        "mood_shift": {
            "Contained Burn": "Sharp Edge",
            "Drifting": "Sharp Edge",
            "Submerged": "Sharp Edge"
        }
    },
    "ellipsis": {
        "symbol": "...",
        "mood_shift": {
            "Drifting": "Hollow Echo",
            "Crystalline": "Drifting",
            "Sharp Edge": "Hollow Echo"
        }
    },
    "repetition": {
        "threshold": 3,
        "mood_shift": {
            "Submerged": "Contained Burn",
            "Drifting": "Hollow Echo",
            "Crystalline": "Sharp Edge"
        }
    }
}

def measure_linguistic_pressure(text: str) -> Dict[str, float]:
    """
    Analyzes the fluid dynamics of text through various linguistic features.
    """
    sentences = re.split(r'[.!?]+', text)
    words = text.split()
    
    # Calculate sentence length variance
    sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
    length_variance = max(sentence_lengths) - min(sentence_lengths) if sentence_lengths else 0
    
    # Calculate punctuation density
    punctuation_count = len(re.findall(r'[.!?]', text))
    punctuation_density = punctuation_count / len(words) if words else 0
    
    # Calculate capital letter pressure
    capital_count = len(re.findall(r'[A-Z]', text))
    capital_pressure = capital_count / len(text) if text else 0
    
    return {
        "length_variance": length_variance,
        "punctuation_density": punctuation_density,
        "capital_pressure": capital_pressure
    }

def detect_repetition_patterns(text: str) -> List[Tuple[str, int]]:
    """
    Captures the stuttering of consciousness through repeated words.
    """
    words = text.lower().split()
    word_counts = Counter(words)
    return [(word, count) for word, count in word_counts.items() if count >= INTENSITY_MARKERS["repetition"]["threshold"]]

def calculate_mood_resonance(text: str) -> Dict[str, float]:
    """
    Calculates how text resonates with different moods, allowing for blended states.
    """
    text = text.lower()
    mood_scores = {mood: 0.0 for mood in MOOD_ARCHETYPES}
    
    # Calculate base resonance from words
    for mood, data in EMOTIONAL_RESONANCE.items():
        for word in data["words"]:
            if word in text:
                mood_scores[mood] += data["weight"]
    
    # Normalize scores
    total_score = sum(mood_scores.values())
    if total_score > 0:
        mood_scores = {mood: score/total_score for mood, score in mood_scores.items()}
    
    return mood_scores

def apply_mood_specific_modulation(text: str, base_resonance: Dict[str, float]) -> Dict[str, float]:
    """
    Creates dynamic mood interactions based on linguistic features.
    """
    pressure = measure_linguistic_pressure(text)
    repetitions = detect_repetition_patterns(text)
    
    # Apply pressure-based modulation
    if pressure["capital_pressure"] > 0.3:
        for mood in base_resonance:
            if mood in INTENSITY_MARKERS["exclamation"]["mood_shift"]:
                target_mood = INTENSITY_MARKERS["exclamation"]["mood_shift"][mood]
                base_resonance[target_mood] += base_resonance[mood] * 0.2
    
    # Apply repetition-based modulation
    for word, count in repetitions:
        for mood in base_resonance:
            if mood in INTENSITY_MARKERS["repetition"]["mood_shift"]:
                target_mood = INTENSITY_MARKERS["repetition"]["mood_shift"][mood]
                base_resonance[target_mood] += base_resonance[mood] * (count / INTENSITY_MARKERS["repetition"]["threshold"])
    
    return base_resonance

def infer_mood(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Infers the dominant mood and its relationships with other moods.
    """
    base_resonance = calculate_mood_resonance(text)
    modulated_resonance = apply_mood_specific_modulation(text, base_resonance)
    
    # Find dominant mood
    dominant_mood = max(modulated_resonance.items(), key=lambda x: x[1])[0]
    
    return dominant_mood, modulated_resonance

def get_mood_metadata(mood: str) -> Optional[Dict[str, List[str]]]:
    """
    Provides properties that show mood relationships.
    """
    return MOOD_ARCHETYPES.get(mood)

# Test cases
if __name__ == "__main__":
    test_cases = [
        "I can't breathe in here anymore",
        "drowning drowning drowning",
        "maybe... i don't know... whatever..."
    ]
    
    for test in test_cases:
        dominant_mood, resonance = infer_mood(test)
        print(f"\nText: {test}")
        print(f"Dominant mood: {dominant_mood}")
        print("Mood resonance:")
        for mood, score in resonance.items():
            if score > 0.1:  # Only show significant resonances
                print(f"  {mood}: {score:.2f}")
