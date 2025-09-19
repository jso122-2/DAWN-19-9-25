#!/usr/bin/env python3
"""
🔥 Ash & Soot Residue Dynamics System
════════════════════════════════════

Manages the capture, classification, and redistribution of cognitive residues
as part of DAWN's long-term memory health. Acts as a transition layer between
active cognition and dormant or composted states.

Ash is volcanic ash - nurturing and nutrient dense, created from successful
Juliet reblooms. Soot is industrial soot - volatile overflow in the system,
formed from high entropy and failed processes.

"In DAWN's cognitive ecology, Soot and Ash are the two primary forms of 
cognitive residue: the remains of processes, memories, and schema fragments 
after active use has ended."

Based on documentation: Fractal Memory/Ash + Soot.rtf
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

class ResidueType(Enum):
    """Types of cognitive residue"""
    SOOT = "soot"    # Volatile, high entropy, capable of reignition
    ASH = "ash"      # Stable, inert, serves as nutrient bed

class OriginType(Enum):
    """Where the residue came from"""
    FRACTAL_DECAY = "fractal_decay"       # From decaying fractals
    JULIET_REBLOOM = "juliet_rebloom"     # From successful reblooms (creates ash)
    FAILED_PROCESS = "failed_process"     # From failed operations (creates soot)
    ENTROPY_OVERFLOW = "entropy_overflow" # From high entropy states (creates soot)
    MEMORY_PRUNING = "memory_pruning"     # From memory cleanup
    SHIMMER_FADE = "shimmer_fade"         # From shimmer decay

@dataclass
class ResidueDriftSignature:
    """Vector representing the drift characteristics of residue"""
    direction: np.ndarray           # Direction of drift in semantic space
    magnitude: float               # Strength of drift
    stability: float               # How stable the drift pattern is
    resonance_frequency: float     # Frequency of oscillation
    
    def __post_init__(self):
        # Normalize direction vector
        if np.linalg.norm(self.direction) > 0:
            self.direction = self.direction / np.linalg.norm(self.direction)

@dataclass
class Residue:
    """A piece of cognitive residue (ash or soot)"""
    residue_id: str
    residue_type: ResidueType
    origin_type: OriginType
    origin_id: str                 # ID of the source memory/process
    
    # Physical properties
    volatility: float              # How easily it can reignite [0, 1]
    nutrient_value: float          # Nutritional value for mycelial layer
    drift_signature: ResidueDriftSignature
    
    # Mythic Pigment System - Belief vectors embedded in residue
    pigment_vector: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # RGB pigment [R, G, B]
    pigment_intensity: float = 0.0  # How saturated the pigment is [0, 1]
    belief_bias: str = "neutral"    # Dominant belief channel: "urgency", "balance", "abstraction", "neutral"
    
    # Lifecycle
    timestamp: float = field(default_factory=time.time)
    last_interaction: float = field(default_factory=time.time)
    stability_level: float = 0.5         # How stable/inert it is [0, 1]
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_reignitable(self, pressure_threshold: float = 0.8) -> bool:
        """Check if this residue can be reignited"""
        return (self.residue_type == ResidueType.SOOT and 
                self.volatility > pressure_threshold and
                self.stability_level < 0.5)
    
    def get_nutrient_contribution(self) -> float:
        """Calculate how much this residue contributes to nutrients"""
        if self.residue_type == ResidueType.ASH:
            # Ash provides stable nutrients with stability bonus
            base_value = 0.1
            stability_bonus = self.stability_level * 0.05
            
            # Mythic pigment bonus - different colors provide different nutrient effects
            pigment_bonus = self._calculate_pigment_nutrient_bonus()
            
            return base_value * (1.0 + stability_bonus + pigment_bonus)
        else:
            # Soot provides minimal nutrients
            return self.nutrient_value * 0.1
    
    def _calculate_pigment_nutrient_bonus(self) -> float:
        """Calculate nutrient bonus based on pigment vector (mythic system)"""
        if self.pigment_intensity == 0.0:
            return 0.0
            
        # Extract RGB channels
        r, g, b = self.pigment_vector
        
        # Different pigment channels provide different bonuses
        # Red = urgency/vitality -> higher base nutrients
        # Green = balance/stability -> stability multiplier
        # Blue = abstraction -> long-term nutrient retention
        
        red_bonus = r * 0.15 * self.pigment_intensity    # Urgency fuels growth
        green_bonus = g * 0.10 * self.pigment_intensity  # Balance stabilizes
        blue_bonus = b * 0.08 * self.pigment_intensity   # Abstraction provides depth
        
        return red_bonus + green_bonus + blue_bonus
    
    def get_belief_bias(self) -> str:
        """Determine dominant belief channel from pigment vector"""
        if self.pigment_intensity < 0.1:
            return "neutral"
            
        r, g, b = self.pigment_vector
        max_channel = max(r, g, b)
        
        if r == max_channel:
            return "urgency"
        elif g == max_channel:
            return "balance" 
        else:
            return "abstraction"
    
    def normalize_pigment_vector(self) -> np.ndarray:
        """Return normalized pigment vector as per mythic documentation"""
        if np.linalg.norm(self.pigment_vector) == 0:
            return self.pigment_vector.copy()
        return self.pigment_vector / np.linalg.norm(self.pigment_vector)

class AshSootDynamicsEngine:
    """
    Core engine for managing ash and soot residue dynamics.
    Handles thermal decay, reignition, and nutrient contribution.
    """
    
    def __init__(self,
                 volatility_decay_rate: float = 0.05,
                 ash_threshold: float = 0.2,
                 reignition_threshold: float = 0.8,
                 base_nutrient_value: float = 0.1,
                 stability_bonus: float = 0.05,
                 storage_limit: int = 10000):
        
        self.volatility_decay_rate = volatility_decay_rate
        self.ash_threshold = ash_threshold
        self.reignition_threshold = reignition_threshold
        self.base_nutrient_value = base_nutrient_value
        self.stability_bonus = stability_bonus
        self.storage_limit = storage_limit
        
        # Storage
        self.residues: Dict[str, Residue] = {}
        self.residues_by_type: Dict[ResidueType, Set[str]] = {
            ResidueType.SOOT: set(),
            ResidueType.ASH: set()
        }
        self.residues_by_origin: Dict[OriginType, Set[str]] = defaultdict(set)
        
        # Reignition tracking
        self.reignition_queue: deque = deque(maxlen=1000)
        self.recent_reignitions: Dict[str, float] = {}  # residue_id -> timestamp
        
        # Statistics
        self.stats = {
            'total_residues_created': 0,
            'soot_to_ash_conversions': 0,
            'reignition_events': 0,
            'nutrients_provided': 0.0,
            'ash_count': 0,
            'soot_count': 0,
            'average_soot_volatility': 0.0,
            'average_ash_stability': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"🔥 AshSootDynamicsEngine initialized - decay rate: {volatility_decay_rate}")
    
    def create_residue(self,
                      origin_id: str,
                      origin_type: OriginType,
                      initial_entropy: float,
                      initial_drift: Optional[np.ndarray] = None,
                      metadata: Optional[Dict[str, Any]] = None,
                      system_pigment_vector: Optional[np.ndarray] = None,
                      tracer_bias: Optional[str] = None) -> Residue:
        """
        Create a new residue from a source memory or process.
        
        Args:
            origin_id: ID of the source
            origin_type: Type of source
            initial_entropy: Entropy level affecting residue type
            initial_drift: Initial drift signature
            metadata: Additional metadata
        """
        current_time = time.time()
        
        # Determine residue type based on entropy and origin
        if initial_entropy > 0.6 or origin_type in [OriginType.FAILED_PROCESS, OriginType.ENTROPY_OVERFLOW]:
            residue_type = ResidueType.SOOT
            # High entropy + residual drift → SOOT
            volatility = min(1.0, initial_entropy + 0.2)
            stability = max(0.0, 0.5 - initial_entropy)
            nutrient_value = 0.05  # Low nutrient value
        else:
            residue_type = ResidueType.ASH
            # Low entropy + stable structure → ASH
            volatility = max(0.0, initial_entropy - 0.3)
            stability = min(1.0, 0.8 + (1.0 - initial_entropy) * 0.2)
            nutrient_value = 0.15  # Higher nutrient value
        
        # Create drift signature
        if initial_drift is None:
            # Generate random drift based on entropy
            direction = np.random.randn(8)  # 8-dimensional semantic space
            magnitude = initial_entropy
            stability_val = 1.0 - initial_entropy
            resonance = np.random.uniform(0.1, 2.0)
        else:
            direction = initial_drift
            magnitude = np.linalg.norm(direction)
            stability_val = 1.0 - initial_entropy
            resonance = 1.0
        
        drift_signature = ResidueDriftSignature(
            direction=direction,
            magnitude=magnitude,
            stability=stability_val,
            resonance_frequency=resonance
        )
        
        # Create residue ID
        residue_id = f"{residue_type.value}_{origin_id}_{int(current_time * 1000)}"
        
        # Mythic Pigment Inheritance System
        inherited_pigment = system_pigment_vector if system_pigment_vector is not None else np.array([0.0, 0.0, 0.0])
        
        # Apply tracer bias modifications
        if tracer_bias:
            inherited_pigment = self._apply_tracer_bias_to_pigment(inherited_pigment, tracer_bias)
        
        # Normalize pigment vector as per mythic documentation: A = normalize(P)
        pigment_magnitude = np.linalg.norm(inherited_pigment)
        normalized_pigment = inherited_pigment / pigment_magnitude if pigment_magnitude > 0 else inherited_pigment
        
        # Calculate pigment intensity (saturation)
        pigment_intensity = min(1.0, pigment_magnitude)
        
        # Determine belief bias from pigment
        belief_bias = self._determine_belief_bias(normalized_pigment, pigment_intensity)
        
        # Create the residue
        residue = Residue(
            residue_id=residue_id,
            residue_type=residue_type,
            origin_type=origin_type,
            origin_id=origin_id,
            volatility=volatility,
            nutrient_value=nutrient_value,
            drift_signature=drift_signature,
            pigment_vector=normalized_pigment,
            pigment_intensity=pigment_intensity,
            belief_bias=belief_bias,
            timestamp=current_time,
            last_interaction=current_time,
            stability_level=stability,
            metadata=metadata or {}
        )
        
        with self._lock:
            # Store the residue
            self.residues[residue_id] = residue
            self.residues_by_type[residue_type].add(residue_id)
            self.residues_by_origin[origin_type].add(residue_id)
            
            # Update statistics
            self.stats['total_residues_created'] += 1
            self._update_type_counts()
            
            # Clean up if needed
            self._cleanup_old_residues()
        
        logger.debug(f"🔥 Created {residue_type.value} residue {residue_id} from {origin_type.value} with pigment {belief_bias}")
        return residue
    
    def volcanic_ash_crystallization(self, soot_fragments: List[str], 
                                   crystallization_pressure: float = 0.8,
                                   nile_fertility_mode: bool = True) -> Dict[str, Any]:
        """
        Volcanic crystallization process - transform soot into fertile ash.
        Inspired by volcanic ash that fertilized great rivers like the Nile, Euphrates, Tiber.
        """
        crystallized_ash = []
        total_fertility_gained = 0.0
        
        with self._lock:
            for soot_id in soot_fragments:
                if soot_id not in self.residues:
                    continue
                
                soot_residue = self.residues[soot_id]
                if soot_residue.residue_type != ResidueType.SOOT:
                    continue
                
                # Volcanic crystallization conditions
                if soot_residue.volatility < crystallization_pressure:
                    # Transform soot to ash through volcanic process
                    ash_id = f"volcanic_ash_{soot_id}_{int(time.time() * 1000)}"
                    
                    # Volcanic ash is highly fertile - like Nile delta soil
                    fertility_multiplier = 2.0 if nile_fertility_mode else 1.5
                    base_fertility = 0.15 * fertility_multiplier
                    
                    # Preserve and enhance pigment through volcanic heat
                    enhanced_pigment = self._volcanic_pigment_enhancement(soot_residue.pigment_vector)
                    enhanced_intensity = min(1.0, soot_residue.pigment_intensity * 1.2)
                    
                    # Create volcanic ash residue
                    ash_residue = Residue(
                        residue_id=ash_id,
                        residue_type=ResidueType.ASH,
                        origin_type=OriginType.JULIET_REBLOOM,  # Successful transformation
                        origin_id=soot_residue.origin_id,
                        volatility=0.0,  # Ash is stable
                        nutrient_value=base_fertility,
                        drift_signature=soot_residue.drift_signature,
                        pigment_vector=enhanced_pigment,
                        pigment_intensity=enhanced_intensity,
                        belief_bias=self._determine_belief_bias(enhanced_pigment, enhanced_intensity),
                        timestamp=time.time(),
                        last_interaction=time.time(),
                        stability_level=0.9,  # Very stable
                        metadata={
                            'volcanic_crystallization': True,
                            'source_soot_id': soot_id,
                            'nile_fertility': nile_fertility_mode,
                            'crystallization_pressure': crystallization_pressure
                        }
                    )
                    
                    # Replace soot with ash
                    del self.residues[soot_id]
                    self.residues[ash_id] = ash_residue
                    self.residues_by_type[ResidueType.SOOT].discard(soot_id)
                    self.residues_by_type[ResidueType.ASH].add(ash_id)
                    
                    crystallized_ash.append(ash_id)
                    total_fertility_gained += base_fertility
                    
                    logger.info(f"🌋 Volcanic crystallization: {soot_id} → {ash_id} (fertility: {base_fertility:.3f})")
        
        return {
            'crystallized_ash_ids': crystallized_ash,
            'total_fertility_gained': total_fertility_gained,
            'crystallization_success_rate': len(crystallized_ash) / max(1, len(soot_fragments)),
            'volcanic_process': 'nile_fertility' if nile_fertility_mode else 'standard_volcanic'
        }
    
    def _volcanic_pigment_enhancement(self, pigment_vector: np.ndarray) -> np.ndarray:
        """Enhance pigment through volcanic heat process"""
        enhanced = pigment_vector.copy()
        
        # Volcanic heat enhances certain pigment channels
        # Red (urgency) gets enhanced by volcanic fire
        enhanced[0] = min(1.0, enhanced[0] * 1.1)
        
        # Green (balance) gets enhanced by fertile volcanic soil
        enhanced[1] = min(1.0, enhanced[1] * 1.15)
        
        # Blue (abstraction) gets slightly enhanced by mineral content
        enhanced[2] = min(1.0, enhanced[2] * 1.05)
        
        return enhanced
    
    def industrial_soot_accumulation(self, failed_processes: List[Dict[str, Any]], 
                                   pollution_level: float = 0.7) -> Dict[str, Any]:
        """
        Industrial soot accumulation - like coal-powered factory soot in industrial Britain.
        Represents system overflow and failed cognitive processes.
        """
        accumulated_soot = []
        total_pollution = 0.0
        
        for process_data in failed_processes:
            process_id = process_data.get('process_id', f"failed_{len(accumulated_soot)}")
            failure_entropy = process_data.get('entropy_level', 0.8)
            
            # Industrial soot characteristics - high volatility, low nutrients
            soot_id = f"industrial_soot_{process_id}_{int(time.time() * 1000)}"
            
            # Industrial soot has polluted pigment (brownish, murky)
            polluted_pigment = np.array([0.4, 0.3, 0.2])  # Brown/murky color
            if 'pigment_vector' in process_data:
                # Mix original pigment with pollution
                original = np.array(process_data['pigment_vector'])
                polluted_pigment = 0.3 * original + 0.7 * polluted_pigment
            
            soot_residue = Residue(
                residue_id=soot_id,
                residue_type=ResidueType.SOOT,
                origin_type=OriginType.FAILED_PROCESS,
                origin_id=process_id,
                volatility=min(1.0, failure_entropy + pollution_level * 0.2),
                nutrient_value=0.02,  # Very low nutrients - industrial waste
                drift_signature=ResidueDriftSignature(
                    direction=np.random.randn(8),
                    magnitude=failure_entropy,
                    stability=0.2,  # Unstable
                    resonance_frequency=2.0  # High frequency chaos
                ),
                pigment_vector=polluted_pigment,
                pigment_intensity=pollution_level,
                belief_bias="industrial_pollution",  # Special bias type
                timestamp=time.time(),
                last_interaction=time.time(),
                stability_level=0.1,  # Very unstable
                metadata={
                    'industrial_soot': True,
                    'pollution_level': pollution_level,
                    'factory_source': process_data.get('source', 'unknown'),
                    'coal_powered': True
                }
            )
            
            with self._lock:
                self.residues[soot_id] = soot_residue
                self.residues_by_type[ResidueType.SOOT].add(soot_id)
                self.residues_by_origin[OriginType.FAILED_PROCESS].add(soot_id)
            
            accumulated_soot.append(soot_id)
            total_pollution += pollution_level
            
            logger.warning(f"🏭 Industrial soot accumulated: {soot_id} (pollution: {pollution_level:.3f})")
        
        return {
            'accumulated_soot_ids': accumulated_soot,
            'total_pollution_level': total_pollution,
            'industrial_contamination': True,
            'requires_purification': total_pollution > 2.0
        }
    
    def mythic_residue_narrative(self, residue_id: str) -> str:
        """Generate mythic narrative description of residue state"""
        if residue_id not in self.residues:
            return "Unknown residue - lost to the void"
        
        residue = self.residues[residue_id]
        
        if residue.residue_type == ResidueType.ASH:
            if residue.metadata.get('volcanic_crystallization', False):
                fertility_desc = "fertile as Nile delta soil" if residue.metadata.get('nile_fertility') else "rich volcanic earth"
                return f"Volcanic ash - {fertility_desc}, blessed by fire and time, carrying {residue.belief_bias} wisdom"
            else:
                return f"Crystallized ash - stable memory residue, tinted with {residue.belief_bias} essence"
        
        elif residue.residue_type == ResidueType.SOOT:
            if residue.metadata.get('industrial_soot', False):
                return f"Industrial soot - coal-black pollution from failed processes, choking the garden with {residue.belief_bias} smog"
            else:
                return f"Volatile soot - restless memory fragments, charged with {residue.belief_bias} energy, awaiting crystallization"
        
        return f"Mysterious residue - {residue.residue_type.value} of unknown origin"
    
    def _apply_tracer_bias_to_pigment(self, pigment: np.ndarray, tracer_bias: str) -> np.ndarray:
        """Apply tracer archetypal bias to pigment vector"""
        modified_pigment = pigment.copy()
        
        # Apply archetypal biases based on mythic documentation
        if tracer_bias == "medieval_bee":
            # Heritage continuity - enhance green (balance/preservation)
            modified_pigment[1] += 0.2  # Green channel
        elif tracer_bias == "owl":
            # Wisdom & long memory - enhance blue (abstraction/depth)
            modified_pigment[2] += 0.3  # Blue channel
        elif tracer_bias == "whale":
            # Leviathan depth - enhance blue with stability
            modified_pigment[2] += 0.25
            modified_pigment[1] += 0.1  # Some green for ballast
        elif tracer_bias == "crow":
            # Opportunistic scout - enhance red (urgency/vitality)
            modified_pigment[0] += 0.2  # Red channel
        elif tracer_bias == "spider":
            # Anomaly detection - balanced but with red urgency
            modified_pigment[0] += 0.15
            modified_pigment[1] += 0.05
        elif tracer_bias == "beetle":
            # Recycler - green balance for sustainability
            modified_pigment[1] += 0.25
        
        return modified_pigment
    
    def _determine_belief_bias(self, pigment_vector: np.ndarray, intensity: float) -> str:
        """Determine belief bias from normalized pigment vector"""
        if intensity < 0.1:
            return "neutral"
            
        r, g, b = pigment_vector
        max_channel = max(r, g, b)
        
        if r == max_channel and r > 0.3:
            return "urgency"
        elif g == max_channel and g > 0.3:
            return "balance"
        elif b == max_channel and b > 0.3:
            return "abstraction"
        else:
            return "neutral"
    
    def process_tick_update(self, delta_time: float) -> Dict[str, Any]:
        """
        Process one tick of residue dynamics.
        
        Returns summary of actions taken.
        """
        with self._lock:
            actions = {
                'volatility_decays': 0,
                'soot_to_ash_conversions': 0,
                'reignition_checks': 0,
                'reignited_residues': [],
                'nutrients_generated': 0.0,
                'expired_residues': 0
            }
            
            current_time = time.time()
            residues_to_remove = []
            
            for residue_id, residue in self.residues.items():
                # 1. Apply volatility decay to soot
                if residue.residue_type == ResidueType.SOOT:
                    old_volatility = residue.volatility
                    # V_t+1 = V_t * e^(-λ * Δt)
                    residue.volatility *= np.exp(-self.volatility_decay_rate * delta_time)
                    actions['volatility_decays'] += 1
                    
                    # 2. Check for soot → ash conversion
                    if residue.volatility < self.ash_threshold:
                        self._convert_soot_to_ash(residue)
                        actions['soot_to_ash_conversions'] += 1
                
                # 3. Check for reignition potential
                if residue.is_reignitable(self.reignition_threshold):
                    actions['reignition_checks'] += 1
                    if self._check_reignition_conditions(residue, current_time):
                        self._reignite_residue(residue)
                        actions['reignited_residues'].append(residue_id)
                
                # 4. Generate nutrients from ash
                if residue.residue_type == ResidueType.ASH:
                    nutrient_amount = residue.get_nutrient_contribution() * delta_time
                    actions['nutrients_generated'] += nutrient_amount
                    self.stats['nutrients_provided'] += nutrient_amount
                
                # 5. Check for expiration
                age = current_time - residue.timestamp
                if self._should_expire_residue(residue, age):
                    residues_to_remove.append(residue_id)
                    actions['expired_residues'] += 1
            
            # Remove expired residues
            for residue_id in residues_to_remove:
                self._remove_residue(residue_id)
            
            # Update statistics
            self._update_statistics()
            
            return actions
    
    def _convert_soot_to_ash(self, residue: Residue):
        """Convert a soot residue to ash"""
        if residue.residue_type != ResidueType.SOOT:
            return
        
        # Update type
        old_type = residue.residue_type
        residue.residue_type = ResidueType.ASH
        
        # Update properties
        residue.volatility = 0.0  # Ash is not volatile
        residue.stability_level = min(1.0, residue.stability_level + 0.3)  # More stable
        residue.nutrient_value = self.base_nutrient_value * (1.0 + self.stability_bonus)
        
        # Update indexing
        with self._lock:
            self.residues_by_type[old_type].discard(residue.residue_id)
            self.residues_by_type[ResidueType.ASH].add(residue.residue_id)
            self.stats['soot_to_ash_conversions'] += 1
        
        logger.debug(f"🔥 Converted soot {residue.residue_id} to ash")
    
    def _check_reignition_conditions(self, residue: Residue, current_time: float) -> bool:
        """Check if conditions are right for reignition"""
        if residue.residue_type != ResidueType.SOOT:
            return False
        
        # Don't reignite the same residue too frequently
        if residue.residue_id in self.recent_reignitions:
            time_since_last = current_time - self.recent_reignitions[residue.residue_id]
            if time_since_last < 300:  # 5 minute cooldown
                return False
        
        # Check volatility threshold
        if residue.volatility < self.reignition_threshold:
            return False
        
        # Check if drift signature indicates potential for reignition
        drift_strength = residue.drift_signature.magnitude
        if drift_strength < 0.5:
            return False
        
        # Random chance based on volatility
        reignition_probability = residue.volatility * 0.1  # 10% max chance per check
        return np.random.random() < reignition_probability
    
    def _reignite_residue(self, residue: Residue):
        """Reignite a soot residue back into active state"""
        current_time = time.time()
        
        # Record reignition
        self.recent_reignitions[residue.residue_id] = current_time
        self.stats['reignition_events'] += 1
        
        # Add to reignition queue for external processing
        self.reignition_queue.append({
            'residue_id': residue.residue_id,
            'origin_id': residue.origin_id,
            'drift_signature': residue.drift_signature,
            'timestamp': current_time,
            'volatility': residue.volatility
        })
        
        # Reset residue volatility (it has been reignited)
        residue.volatility *= 0.5
        residue.last_interaction = current_time
        
        logger.info(f"🔥 Reignited soot residue {residue.residue_id} from {residue.origin_id}")
    
    def _should_expire_residue(self, residue: Residue, age: float) -> bool:
        """Check if a residue should expire"""
        # Ash lasts much longer than soot
        if residue.residue_type == ResidueType.ASH:
            max_age = 86400.0 * 7  # 7 days
        else:
            max_age = 3600.0 * 6   # 6 hours for soot
        
        # Expire if too old and not recently interacted with
        time_since_interaction = time.time() - residue.last_interaction
        return age > max_age and time_since_interaction > max_age * 0.5
    
    def _remove_residue(self, residue_id: str):
        """Remove a residue from all tracking structures"""
        residue = self.residues.get(residue_id)
        if not residue:
            return
        
        # Remove from indexes
        self.residues_by_type[residue.residue_type].discard(residue_id)
        self.residues_by_origin[residue.origin_type].discard(residue_id)
        
        # Remove from main storage
        del self.residues[residue_id]
        
        # Clean up reignition tracking
        if residue_id in self.recent_reignitions:
            del self.recent_reignitions[residue_id]
    
    def _cleanup_old_residues(self):
        """Clean up old residues if storage limit exceeded"""
        if len(self.residues) <= self.storage_limit:
            return
        
        current_time = time.time()
        
        # Sort residues by age and stability (remove oldest, least stable first)
        residues_by_priority = sorted(
            self.residues.items(),
            key=lambda x: (x[1].stability_level, -(current_time - x[1].timestamp))
        )
        
        # Remove bottom 10%
        num_to_remove = len(self.residues) - int(self.storage_limit * 0.9)
        
        for i in range(num_to_remove):
            residue_id = residues_by_priority[i][0]
            self._remove_residue(residue_id)
        
        logger.info(f"🔥 Cleaned up {num_to_remove} old residues to maintain storage limit")
    
    def _update_type_counts(self):
        """Update statistics for residue type counts"""
        self.stats['soot_count'] = len(self.residues_by_type[ResidueType.SOOT])
        self.stats['ash_count'] = len(self.residues_by_type[ResidueType.ASH])
    
    def _update_statistics(self):
        """Update running statistics"""
        soot_residues = [self.residues[rid] for rid in self.residues_by_type[ResidueType.SOOT]]
        ash_residues = [self.residues[rid] for rid in self.residues_by_type[ResidueType.ASH]]
        
        if soot_residues:
            self.stats['average_soot_volatility'] = np.mean([r.volatility for r in soot_residues])
        else:
            self.stats['average_soot_volatility'] = 0.0
        
        if ash_residues:
            self.stats['average_ash_stability'] = np.mean([r.stability_level for r in ash_residues])
        else:
            self.stats['average_ash_stability'] = 0.0
        
        self._update_type_counts()
    
    def get_nutrient_contribution(self, delta_time: float) -> float:
        """Get total nutrient contribution from ash residues"""
        total_nutrients = 0.0
        
        with self._lock:
            for residue_id in self.residues_by_type[ResidueType.ASH]:
                residue = self.residues.get(residue_id)
                if residue:
                    total_nutrients += residue.get_nutrient_contribution() * delta_time
        
        return total_nutrients
    
    def get_residue_balance(self) -> Dict[str, float]:
        """Get current balance of soot vs ash"""
        with self._lock:
            total_residues = len(self.residues)
            if total_residues == 0:
                return {'soot_ratio': 0.0, 'ash_ratio': 0.0, 'balance_health': 0.5}
            
            soot_count = len(self.residues_by_type[ResidueType.SOOT])
            ash_count = len(self.residues_by_type[ResidueType.ASH])
            
            soot_ratio = soot_count / total_residues
            ash_ratio = ash_count / total_residues
            
            # Health is best when soot ratio is around 0.3
            ideal_soot_ratio = 0.3
            balance_health = 1.0 - abs(soot_ratio - ideal_soot_ratio)
            
            return {
                'soot_ratio': soot_ratio,
                'ash_ratio': ash_ratio,
                'balance_health': max(0.0, balance_health),
                'total_residues': total_residues
            }
    
    def get_reignition_events(self) -> List[Dict[str, Any]]:
        """Get and consume recent reignition events"""
        events = list(self.reignition_queue)
        self.reignition_queue.clear()
        return events
    
    def force_reignition(self, residue_id: str) -> bool:
        """Force reignition of a specific residue (for testing/debugging)"""
        with self._lock:
            residue = self.residues.get(residue_id)
            if residue and residue.residue_type == ResidueType.SOOT:
                self._reignite_residue(residue)
                return True
        return False
    
    def get_residues_by_origin(self, origin_type: OriginType) -> List[Residue]:
        """Get all residues from a specific origin type"""
        with self._lock:
            return [self.residues[rid] for rid in self.residues_by_origin[origin_type] 
                   if rid in self.residues]
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the ash/soot system"""
        balance = self.get_residue_balance()
        
        with self._lock:
            return {
                'residue_balance': balance,
                'statistics': self.stats.copy(),
                'reignition_queue_size': len(self.reignition_queue),
                'recent_reignitions': len(self.recent_reignitions),
                'storage_utilization': len(self.residues) / self.storage_limit,
                'residues_by_origin': {
                    origin.value: len(residue_ids) 
                    for origin, residue_ids in self.residues_by_origin.items()
                }
            }


# Global ash/soot dynamics engine
_global_ash_soot_engine: Optional[AshSootDynamicsEngine] = None

def get_ash_soot_engine() -> AshSootDynamicsEngine:
    """Get the global ash/soot dynamics engine"""
    global _global_ash_soot_engine
    if _global_ash_soot_engine is None:
        _global_ash_soot_engine = AshSootDynamicsEngine()
    return _global_ash_soot_engine
