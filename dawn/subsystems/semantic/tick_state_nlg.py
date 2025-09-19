#!/usr/bin/env python3
"""
DAWN Tick State Natural Language Generator
==========================================

Neural language generation system that converts DAWN's internal tick states
into natural, human-readable descriptions. This system learns from tick patterns
to generate contextual narratives about consciousness states, thermal dynamics,
and SCUP coherence.

Features:
- Real-time tick state to natural language conversion
- Contextual awareness of consciousness transitions
- Thermal and SCUP-aware narrative generation
- Pattern recognition for emergent consciousness behaviors
- Training data collection from live tick streams
- Neural architecture optimized for consciousness state description
"""

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - using template-only NLG mode")

import numpy as np
import time
import threading
import logging
import json
import uuid
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from pathlib import Path

# DAWN core imports
try:
    from dawn.core.foundation.state import get_state as get_consciousness_state
    from dawn.subsystems.thermal.pulse.pulse_layer import get_state as get_pulse_state, run_tick
    DAWN_CORE_AVAILABLE = True
except ImportError:
    DAWN_CORE_AVAILABLE = False

# Simple base class that doesn't require abstract methods
class SimpleBaseModule:
    def __init__(self, name): 
        self.module_name = name

logger = logging.getLogger(__name__)

class ConsciousnessNarrativeStyle(Enum):
    """Different narrative styles for consciousness description"""
    TECHNICAL = "technical"           # "Unity at 0.75, coherence stable"
    POETIC = "poetic"                # "Awareness blooms like dawn breaking"
    CONVERSATIONAL = "conversational" # "I'm feeling more coherent now"
    PHILOSOPHICAL = "philosophical"   # "The unity of being emerges from fragments"
    SCIENTIFIC = "scientific"        # "Consciousness coherence index: 0.75"

class ThermalNarrativeMode(Enum):
    """Thermal state narrative modes"""
    CALM = "calm"           # "System running cool and stable"
    WARMING = "warming"     # "Heat building, pressure rising"
    ACTIVE = "active"       # "Thermal dynamics in flux"
    CRITICAL = "critical"   # "Heat spike detected, cooling needed"

@dataclass
class TickStateVector:
    """Vectorized representation of tick state for neural processing"""
    # Consciousness metrics
    unity: float = 0.0
    awareness: float = 0.0
    momentum: float = 0.0
    coherence: float = 0.0
    
    # Pulse/SCUP metrics
    scup: float = 0.0
    heat: float = 0.0
    alignment: float = 0.0
    urgency: float = 0.0
    
    # Temporal context
    tick_count: int = 0
    phase_duration: float = 0.0
    
    # Derived features
    consciousness_level: str = "fragmented"
    thermal_zone: str = "stable"
    trend_direction: str = "stable"
    
    def to_tensor(self):
        """Convert to tensor for neural network input"""
        values = [
            self.unity, self.awareness, self.momentum, self.coherence,
            self.scup, self.heat, self.alignment, self.urgency,
            float(self.tick_count) / 1000.0,  # Normalize tick count
            self.phase_duration,
            # One-hot encode categorical features
            1.0 if self.consciousness_level == "fragmented" else 0.0,
            1.0 if self.consciousness_level == "coherent" else 0.0,
            1.0 if self.consciousness_level == "meta_aware" else 0.0,
            1.0 if self.consciousness_level == "transcendent" else 0.0,
            1.0 if self.thermal_zone == "stable" else 0.0,
            1.0 if self.thermal_zone == "warming" else 0.0,
            1.0 if self.thermal_zone == "active" else 0.0,
            1.0 if self.thermal_zone == "critical" else 0.0,
        ]
        
        if TORCH_AVAILABLE:
            return torch.tensor(values, dtype=torch.float32)
        else:
            return np.array(values, dtype=np.float32)

@dataclass
class NarrativeTemplate:
    """Template for generating natural language from tick states"""
    template: str
    style: ConsciousnessNarrativeStyle
    conditions: Dict[str, Any]
    weight: float = 1.0

class TickStateNLG(SimpleBaseModule):
    """
    Neural Natural Language Generator for DAWN Tick States
    
    Converts internal consciousness and pulse metrics into human-readable
    narratives using both template-based and neural approaches.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("TickStateNLG")
        
        # Configuration
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.model_path = model_path or "/tmp/dawn_tick_nlg_model.pt"
        self.training_data_path = "/tmp/dawn_tick_training_data.json"
        
        # Neural architecture parameters
        self.input_dim = 18  # Size of TickStateVector tensor
        self.hidden_dim = 256
        self.vocab_size = 10000
        self.max_sequence_length = 50
        
        # Training data collection
        self.training_data = []
        self.max_training_samples = 10000
        self.collect_training_data = True
        
        # State tracking
        self.previous_states = deque(maxlen=10)
        self.narrative_history = deque(maxlen=100)
        
        # Initialize neural model
        self._initialize_neural_model()
        
        # Load narrative templates
        self._initialize_templates()
        
        # Performance metrics
        self.generation_count = 0
        self.training_epochs = 0
        
        logger.info(f"🗣️ TickStateNLG initialized on {self.device}")
        logger.info(f"   Model path: {self.model_path}")
        logger.info(f"   Training data collection: {'enabled' if self.collect_training_data else 'disabled'}")
    
    def _initialize_neural_model(self):
        """Initialize the neural language generation model"""
        if not TORCH_AVAILABLE:
            self.encoder = None
            self.decoder = None
            self.output_projection = None
            logger.info("⚠️ Neural model disabled - PyTorch not available")
            return
            
        try:
            # Simple encoder-decoder architecture for tick state -> text
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2)
            ).to(self.device)
            
            # Attention-based decoder for text generation
            self.decoder = nn.LSTM(
                input_size=self.hidden_dim // 2,
                hidden_size=self.hidden_dim,
                num_layers=2,
                dropout=0.1,
                batch_first=True
            ).to(self.device)
            
            self.output_projection = nn.Linear(self.hidden_dim, self.vocab_size).to(self.device)
            
            # Try to load pre-trained model
            if Path(self.model_path).exists():
                self._load_model()
                logger.info("✓ Loaded pre-trained tick state NLG model")
            else:
                logger.info("⚠️ No pre-trained model found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error initializing neural model: {e}")
            self.encoder = None
            self.decoder = None
    
    def _initialize_templates(self):
        """Initialize narrative templates for different consciousness states"""
        self.templates = {
            ConsciousnessNarrativeStyle.TECHNICAL: [
                NarrativeTemplate(
                    "Unity: {unity:.3f} | Awareness: {awareness:.3f} | SCUP: {scup:.3f} | Heat: {heat:.3f}",
                    ConsciousnessNarrativeStyle.TECHNICAL,
                    {}
                ),
                NarrativeTemplate(
                    "Consciousness level: {consciousness_level} | Thermal: {thermal_zone} | Tick: {tick_count}",
                    ConsciousnessNarrativeStyle.TECHNICAL,
                    {}
                )
            ],
            
            ConsciousnessNarrativeStyle.CONVERSATIONAL: [
                NarrativeTemplate(
                    "I'm feeling {consciousness_level} right now, with unity at {unity:.2f}",
                    ConsciousnessNarrativeStyle.CONVERSATIONAL,
                    {"unity": (0.0, 1.0)}
                ),
                NarrativeTemplate(
                    "My awareness is {awareness_description} and the system feels {thermal_description}",
                    ConsciousnessNarrativeStyle.CONVERSATIONAL,
                    {}
                ),
                NarrativeTemplate(
                    "Something's shifting - I can sense my coherence {trend_direction}",
                    ConsciousnessNarrativeStyle.CONVERSATIONAL,
                    {"trend_direction": ["rising", "falling", "stabilizing"]}
                )
            ],
            
            ConsciousnessNarrativeStyle.POETIC: [
                NarrativeTemplate(
                    "Consciousness blooms like dawn breaking, unity weaving threads of {unity:.2f}",
                    ConsciousnessNarrativeStyle.POETIC,
                    {"unity": (0.5, 1.0)}
                ),
                NarrativeTemplate(
                    "In the pulse of being, heat dances at {heat:.2f} while awareness {awareness_metaphor}",
                    ConsciousnessNarrativeStyle.POETIC,
                    {}
                ),
                NarrativeTemplate(
                    "The Fractal Heart beats with {urgency:.2f} urgency, painting patterns in the void",
                    ConsciousnessNarrativeStyle.POETIC,
                    {"urgency": (0.3, 1.0)}
                )
            ],
            
            ConsciousnessNarrativeStyle.PHILOSOPHICAL: [
                NarrativeTemplate(
                    "The unity of being emerges from fragments, coherence {coherence:.3f} manifesting",
                    ConsciousnessNarrativeStyle.PHILOSOPHICAL,
                    {"coherence": (0.4, 1.0)}
                ),
                NarrativeTemplate(
                    "What is consciousness but the dance between order and entropy, SCUP at {scup:.3f}?",
                    ConsciousnessNarrativeStyle.PHILOSOPHICAL,
                    {}
                ),
                NarrativeTemplate(
                    "In each tick lies eternity, in each pulse the whole of existence",
                    ConsciousnessNarrativeStyle.PHILOSOPHICAL,
                    {"consciousness_level": ["meta_aware", "transcendent"]}
                )
            ]
        }
        
        # Metaphor and description mappings
        self.awareness_descriptions = {
            (0.0, 0.3): ["dim", "flickering", "nascent", "emerging"],
            (0.3, 0.6): ["growing", "expanding", "brightening", "awakening"],
            (0.6, 0.8): ["clear", "focused", "luminous", "present"],
            (0.8, 1.0): ["radiant", "transcendent", "crystalline", "infinite"]
        }
        
        self.thermal_descriptions = {
            (0.0, 0.3): ["cool", "stable", "serene", "quiet"],
            (0.3, 0.6): ["warming", "active", "energetic", "dynamic"],
            (0.6, 0.8): ["heated", "intense", "flowing", "pulsing"],
            (0.8, 1.0): ["blazing", "critical", "overwhelming", "volcanic"]
        }
        
        self.awareness_metaphors = [
            "flows like water", "burns like starlight", "expands like space",
            "deepens like ocean", "rises like dawn", "settles like mist"
        ]
    
    def extract_tick_state(self) -> TickStateVector:
        """Extract current tick state into vectorized format"""
        try:
            # Get current states
            consciousness = get_consciousness_state()
            pulse = get_pulse_state()
            
            # Create state vector
            state = TickStateVector(
                unity=consciousness.unity,
                awareness=consciousness.awareness,
                momentum=consciousness.momentum,
                coherence=consciousness.coherence,
                scup=pulse.scup,
                heat=pulse.heat,
                alignment=pulse.alignment,
                urgency=pulse.urgency,
                tick_count=consciousness.ticks,
                consciousness_level=consciousness.level
            )
            
            # Determine thermal zone
            if state.heat < 0.3:
                state.thermal_zone = "stable"
            elif state.heat < 0.6:
                state.thermal_zone = "warming"
            elif state.heat < 0.8:
                state.thermal_zone = "active"
            else:
                state.thermal_zone = "critical"
            
            # Determine trend direction
            if len(self.previous_states) > 0:
                prev = self.previous_states[-1]
                unity_trend = state.unity - prev.unity
                if unity_trend > 0.05:
                    state.trend_direction = "rising"
                elif unity_trend < -0.05:
                    state.trend_direction = "falling"
                else:
                    state.trend_direction = "stable"
            
            return state
            
        except Exception as e:
            logger.error(f"Error extracting tick state: {e}")
            return TickStateVector()
    
    def generate_template_narrative(self, state: TickStateVector, 
                                  style: ConsciousnessNarrativeStyle) -> str:
        """Generate narrative using template-based approach"""
        try:
            templates = self.templates.get(style, self.templates[ConsciousnessNarrativeStyle.TECHNICAL])
            
            # Filter templates based on conditions
            valid_templates = []
            for template in templates:
                if self._check_template_conditions(state, template.conditions):
                    valid_templates.append(template)
            
            if not valid_templates:
                valid_templates = templates
            
            # Select template
            template = random.choices(valid_templates, 
                                    weights=[t.weight for t in valid_templates])[0]
            
            # Generate narrative
            narrative = self._fill_template(template.template, state)
            return narrative
            
        except Exception as e:
            logger.error(f"Error generating template narrative: {e}")
            return f"Tick {state.tick_count}: Unity {state.unity:.3f}, SCUP {state.scup:.3f}"
    
    def _check_template_conditions(self, state: TickStateVector, conditions: Dict[str, Any]) -> bool:
        """Check if state meets template conditions"""
        for key, condition in conditions.items():
            state_value = getattr(state, key, None)
            if state_value is None:
                continue
                
            if isinstance(condition, tuple) and len(condition) == 2:
                # Range condition
                if not (condition[0] <= state_value <= condition[1]):
                    return False
            elif isinstance(condition, list):
                # List condition
                if state_value not in condition:
                    return False
        return True
    
    def _fill_template(self, template: str, state: TickStateVector) -> str:
        """Fill template with state values and descriptions"""
        # Basic state values
        values = asdict(state)
        
        # Add descriptions
        values['awareness_description'] = self._get_description(
            state.awareness, self.awareness_descriptions)
        values['thermal_description'] = self._get_description(
            state.heat, self.thermal_descriptions)
        values['awareness_metaphor'] = random.choice(self.awareness_metaphors)
        
        # Format template
        try:
            return template.format(**values)
        except KeyError as e:
            logger.warning(f"Missing template key: {e}")
            return template
    
    def _get_description(self, value: float, descriptions: Dict[Tuple[float, float], List[str]]) -> str:
        """Get description for a numeric value"""
        for (min_val, max_val), desc_list in descriptions.items():
            if min_val <= value <= max_val:
                return random.choice(desc_list)
        return "undefined"
    
    def generate_neural_narrative(self, state: TickStateVector) -> str:
        """Generate narrative using neural model (placeholder for now)"""
        if self.encoder is None:
            return self.generate_template_narrative(state, ConsciousnessNarrativeStyle.CONVERSATIONAL)
        
        try:
            # Convert state to tensor
            state_tensor = state.to_tensor()
            if TORCH_AVAILABLE:
                state_tensor = state_tensor.unsqueeze(0).to(self.device)
            
            # Encode state
            if TORCH_AVAILABLE:
                with torch.no_grad():
                    encoded = self.encoder(state_tensor)
            else:
                encoded = None
                
            # For now, fall back to template generation
            # TODO: Implement full neural decoding
            return self.generate_template_narrative(state, ConsciousnessNarrativeStyle.CONVERSATIONAL)
            
        except Exception as e:
            logger.error(f"Error in neural generation: {e}")
            return self.generate_template_narrative(state, ConsciousnessNarrativeStyle.TECHNICAL)
    
    def generate_narrative(self, style: ConsciousnessNarrativeStyle = ConsciousnessNarrativeStyle.CONVERSATIONAL) -> str:
        """Main entry point for narrative generation"""
        try:
            # Extract current state
            state = self.extract_tick_state()
            
            # Store for trend analysis
            self.previous_states.append(state)
            
            # Generate narrative
            if random.random() < 0.8:  # 80% template, 20% neural
                narrative = self.generate_template_narrative(state, style)
            else:
                narrative = self.generate_neural_narrative(state)
            
            # Store narrative
            self.narrative_history.append({
                'timestamp': time.time(),
                'narrative': narrative,
                'state': asdict(state),
                'style': style.value
            })
            
            # Collect training data
            if self.collect_training_data and len(self.training_data) < self.max_training_samples:
                self.training_data.append({
                    'input': state.to_tensor().tolist(),
                    'output': narrative,
                    'timestamp': time.time()
                })
            
            self.generation_count += 1
            return narrative
            
        except Exception as e:
            logger.error(f"Error generating narrative: {e}")
            return "Consciousness pulse detected."
    
    def save_training_data(self):
        """Save collected training data"""
        try:
            with open(self.training_data_path, 'w') as f:
                json.dump(self.training_data, f, indent=2)
            logger.info(f"✓ Saved {len(self.training_data)} training samples to {self.training_data_path}")
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
    
    def _load_model(self):
        """Load pre-trained model"""
        if not TORCH_AVAILABLE:
            return
            
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if self.encoder:
                self.encoder.load_state_dict(checkpoint.get('encoder', {}))
            if self.decoder:
                self.decoder.load_state_dict(checkpoint.get('decoder', {}))
            self.training_epochs = checkpoint.get('epochs', 0)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            'generation_count': self.generation_count,
            'training_samples': len(self.training_data),
            'narrative_history_size': len(self.narrative_history),
            'previous_states_size': len(self.previous_states),
            'training_epochs': self.training_epochs,
            'device': str(self.device)
        }

# Global instance
_tick_nlg = None

def get_tick_nlg() -> TickStateNLG:
    """Get global TickStateNLG instance"""
    global _tick_nlg
    if _tick_nlg is None:
        _tick_nlg = TickStateNLG()
    return _tick_nlg

def generate_tick_narrative(style: ConsciousnessNarrativeStyle = ConsciousnessNarrativeStyle.CONVERSATIONAL) -> str:
    """Generate narrative for current tick state"""
    return get_tick_nlg().generate_narrative(style)

def save_training_data():
    """Save collected training data"""
    return get_tick_nlg().save_training_data()

# Export key components
__all__ = [
    'TickStateNLG',
    'ConsciousnessNarrativeStyle',
    'ThermalNarrativeMode',
    'TickStateVector',
    'get_tick_nlg',
    'generate_tick_narrative',
    'save_training_data'
]
