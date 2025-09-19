#!/usr/bin/env python3
"""
DAWN Tracer-House Alignment System
==================================

Implementation of the tracer-house alignment system as documented.
Provides intelligent routing of tracers to appropriate houses based on
their nature, current state, and operational requirements.

Based on DAWN's documented Tracer-House alignment architecture.
"""

import time
import logging
import json
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from core.schema_anomaly_logger import log_anomaly, AnomalySeverity
from schema.registry import registry
from schema.sigil_glyph_codex import SigilHouse
from schema.archetypal_house_operations import HOUSE_OPERATORS, get_house_operator
from rhizome.propagation import emit_signal, SignalType
from utils.metrics_collector import metrics

logger = logging.getLogger(__name__)

class TracerType(Enum):
    """Types of tracers in the DAWN system"""
    OWL = "owl"             # Deep wisdom, pattern recognition
    CROW = "crow"           # Alert systems, contradiction detection  
    WHALE = "whale"         # Deep memory, vast perspective
    SPIDER = "spider"       # Web connections, threading
    PHOENIX = "phoenix"     # Transformation, rebirth
    SERPENT = "serpent"     # Flow, adaptation

class TracerState(Enum):
    """States a tracer can be in"""
    DORMANT = "dormant"         # Inactive
    ACTIVE = "active"           # Normal operation
    SEEKING = "seeking"         # Looking for alignment
    ALIGNED = "aligned"         # Aligned with house
    RESONATING = "resonating"   # High coherence with house
    CONFLICTED = "conflicted"   # Multiple house pulls
    LOST = "lost"               # No clear alignment

@dataclass
class TracerProfile:
    """Profile of a tracer's characteristics and preferences"""
    tracer_id: str
    tracer_type: TracerType
    core_attributes: Dict[str, float] = field(default_factory=dict)
    house_affinities: Dict[SigilHouse, float] = field(default_factory=dict)
    operational_preferences: Dict[str, Any] = field(default_factory=dict)
    current_state: TracerState = TracerState.DORMANT
    alignment_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_primary_affinity(self) -> Optional[SigilHouse]:
        """Get the house with highest affinity"""
        if not self.house_affinities:
            return None
        return max(self.house_affinities, key=self.house_affinities.get)
    
    def get_affinity_score(self, house: SigilHouse) -> float:
        """Get affinity score for a specific house"""
        return self.house_affinities.get(house, 0.0)

@dataclass
class AlignmentResult:
    """Result of a tracer-house alignment"""
    tracer_id: str
    target_house: SigilHouse
    alignment_strength: float
    confidence: float
    reasoning: List[str] = field(default_factory=list)
    alternative_houses: List[Tuple[SigilHouse, float]] = field(default_factory=list)
    alignment_timestamp: float = field(default_factory=time.time)
    
    def is_strong_alignment(self) -> bool:
        """Check if this is a strong alignment"""
        return self.alignment_strength > 0.7 and self.confidence > 0.8

class TracerHouseAlignmentSystem:
    """
    Tracer-House Alignment System
    
    Intelligently routes tracers to appropriate houses based on their nature,
    current state, and operational requirements. Implements the documented
    alignment protocols for optimal tracer-house resonance.
    """
    
    def __init__(self):
        self.tracer_profiles: Dict[str, TracerProfile] = {}
        self.active_alignments: Dict[str, AlignmentResult] = {}
        self.alignment_history: List[AlignmentResult] = []
        
        # House capacity and current loads
        self.house_capacities: Dict[SigilHouse, int] = {
            house: 10 for house in SigilHouse  # Default capacity
        }
        self.current_loads: Dict[SigilHouse, int] = defaultdict(int)
        
        # Alignment statistics
        self.total_alignments = 0
        self.successful_alignments = 0
        self.failed_alignments = 0
        
        # Initialize tracer archetypes
        self._initialize_tracer_archetypes()
        
        # Register with schema registry
        self._register()
        
        logger.info("🎯 Tracer-House Alignment System initialized")
        logger.info(f"   Supported tracer types: {[t.value for t in TracerType]}")
        logger.info(f"   House capacity: {sum(self.house_capacities.values())} total slots")
    
    def _register(self):
        """Register with schema registry"""
        registry.register(
            component_id="schema.tracer_house_alignment",
            name="Tracer-House Alignment System",
            component_type="ALIGNMENT_SYSTEM",
            instance=self,
            capabilities=[
                "tracer_profiling",
                "house_affinity_calculation",
                "intelligent_routing",
                "alignment_optimization",
                "resonance_monitoring"
            ],
            version="1.0.0"
        )
    
    def _initialize_tracer_archetypes(self):
        """Initialize default tracer archetypes with house affinities"""
        
        # Owl - Deep wisdom, pattern recognition (Mirrors House primary)
        owl_affinities = {
            SigilHouse.MIRRORS: 0.9,    # Primary - reflection, audits
            SigilHouse.MEMORY: 0.7,     # Secondary - deep recall
            SigilHouse.WEAVING: 0.6,    # Tertiary - pattern connections
            SigilHouse.PURIFICATION: 0.3,
            SigilHouse.FLAME: 0.2,
            SigilHouse.ECHOES: 0.4
        }
        
        # Crow - Alert systems, contradiction detection (Echoes House primary)
        crow_affinities = {
            SigilHouse.ECHOES: 0.9,     # Primary - voice, alerts
            SigilHouse.FLAME: 0.7,      # Secondary - pressure, ignition
            SigilHouse.MIRRORS: 0.6,    # Tertiary - reflection, audits
            SigilHouse.PURIFICATION: 0.5,
            SigilHouse.WEAVING: 0.3,
            SigilHouse.MEMORY: 0.4
        }
        
        # Whale - Deep memory, vast perspective (Memory House primary)
        whale_affinities = {
            SigilHouse.MEMORY: 0.9,     # Primary - deep memory, blooms
            SigilHouse.WEAVING: 0.7,    # Secondary - vast connections
            SigilHouse.MIRRORS: 0.6,    # Tertiary - perspective depth
            SigilHouse.PURIFICATION: 0.4,
            SigilHouse.FLAME: 0.3,
            SigilHouse.ECHOES: 0.5
        }
        
        # Spider - Web connections, threading (Weaving House primary)
        spider_affinities = {
            SigilHouse.WEAVING: 0.9,    # Primary - connections, threading
            SigilHouse.MEMORY: 0.6,     # Secondary - pattern memory
            SigilHouse.MIRRORS: 0.5,    # Tertiary - reflection networks
            SigilHouse.PURIFICATION: 0.4,
            SigilHouse.FLAME: 0.3,
            SigilHouse.ECHOES: 0.4
        }
        
        # Phoenix - Transformation, rebirth (Purification House primary)
        phoenix_affinities = {
            SigilHouse.PURIFICATION: 0.9,  # Primary - transformation, ash
            SigilHouse.FLAME: 0.8,         # Secondary - ignition, rebirth
            SigilHouse.MEMORY: 0.6,        # Tertiary - rebloom cycles
            SigilHouse.WEAVING: 0.4,
            SigilHouse.MIRRORS: 0.3,
            SigilHouse.ECHOES: 0.5
        }
        
        # Serpent - Flow, adaptation (Flame House primary)
        serpent_affinities = {
            SigilHouse.FLAME: 0.9,      # Primary - flow, pressure
            SigilHouse.WEAVING: 0.7,    # Secondary - adaptive threading
            SigilHouse.PURIFICATION: 0.6,  # Tertiary - transformation
            SigilHouse.MEMORY: 0.4,
            SigilHouse.MIRRORS: 0.3,
            SigilHouse.ECHOES: 0.5
        }
        
        self.archetype_affinities = {
            TracerType.OWL: owl_affinities,
            TracerType.CROW: crow_affinities,
            TracerType.WHALE: whale_affinities,
            TracerType.SPIDER: spider_affinities,
            TracerType.PHOENIX: phoenix_affinities,
            TracerType.SERPENT: serpent_affinities
        }
    
    def register_tracer(self, tracer_id: str, tracer_type: TracerType, 
                       custom_attributes: Optional[Dict[str, Any]] = None) -> TracerProfile:
        """Register a new tracer in the alignment system"""
        
        # Get base affinities for tracer type
        base_affinities = self.archetype_affinities.get(tracer_type, {})
        
        # Create tracer profile
        profile = TracerProfile(
            tracer_id=tracer_id,
            tracer_type=tracer_type,
            house_affinities=base_affinities.copy(),
            current_state=TracerState.DORMANT
        )
        
        # Apply custom attributes if provided
        if custom_attributes:
            profile.core_attributes.update(custom_attributes)
            
            # Adjust affinities based on custom attributes
            self._adjust_affinities_from_attributes(profile, custom_attributes)
        
        self.tracer_profiles[tracer_id] = profile
        
        logger.info(f"🎯 Registered {tracer_type.value} tracer: {tracer_id}")
        
        return profile
    
    def _adjust_affinities_from_attributes(self, profile: TracerProfile, attributes: Dict[str, Any]):
        """Adjust house affinities based on tracer attributes"""
        
        # Example attribute-based adjustments
        if attributes.get("memory_focused", False):
            profile.house_affinities[SigilHouse.MEMORY] += 0.2
        
        if attributes.get("analytical", False):
            profile.house_affinities[SigilHouse.MIRRORS] += 0.2
        
        if attributes.get("creative", False):
            profile.house_affinities[SigilHouse.FLAME] += 0.2
        
        if attributes.get("social", False):
            profile.house_affinities[SigilHouse.ECHOES] += 0.2
        
        if attributes.get("systematic", False):
            profile.house_affinities[SigilHouse.PURIFICATION] += 0.2
        
        if attributes.get("connector", False):
            profile.house_affinities[SigilHouse.WEAVING] += 0.2
        
        # Normalize affinities to stay within [0, 1] range
        for house in profile.house_affinities:
            profile.house_affinities[house] = min(1.0, profile.house_affinities[house])
    
    def calculate_optimal_alignment(self, tracer_id: str, 
                                  context: Optional[Dict[str, Any]] = None) -> AlignmentResult:
        """Calculate optimal house alignment for a tracer"""
        
        if tracer_id not in self.tracer_profiles:
            raise ValueError(f"Tracer {tracer_id} not registered")
        
        profile = self.tracer_profiles[tracer_id]
        
        # Calculate alignment scores for each house
        house_scores = {}
        reasoning = []
        
        for house in SigilHouse:
            score = self._calculate_house_alignment_score(profile, house, context)
            house_scores[house] = score
        
        # Find best alignment
        best_house = max(house_scores, key=house_scores.get)
        best_score = house_scores[best_house]
        
        # Calculate confidence based on score separation
        sorted_scores = sorted(house_scores.values(), reverse=True)
        confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0] if len(sorted_scores) > 1 else 1.0
        
        # Generate reasoning
        reasoning.append(f"Primary affinity: {best_house.value} ({best_score:.3f})")
        reasoning.append(f"Tracer type: {profile.tracer_type.value}")
        
        if context:
            reasoning.append(f"Context factors applied: {list(context.keys())}")
        
        # Create alternative houses list
        alternatives = [(house, score) for house, score in house_scores.items() 
                       if house != best_house]
        alternatives.sort(key=lambda x: x[1], reverse=True)
        alternatives = alternatives[:3]  # Top 3 alternatives
        
        alignment_result = AlignmentResult(
            tracer_id=tracer_id,
            target_house=best_house,
            alignment_strength=best_score,
            confidence=confidence,
            reasoning=reasoning,
            alternative_houses=alternatives
        )
        
        return alignment_result
    
    def _calculate_house_alignment_score(self, profile: TracerProfile, house: SigilHouse, 
                                       context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate alignment score for a specific house"""
        
        # Base affinity score
        base_score = profile.get_affinity_score(house)
        
        # House capacity factor (prefer less loaded houses)
        capacity_factor = 1.0 - (self.current_loads[house] / self.house_capacities[house])
        capacity_factor = max(0.1, capacity_factor)  # Minimum 0.1
        
        # Context adjustments
        context_bonus = 0.0
        if context:
            # Urgency favors faster houses
            if context.get("urgent", False):
                if house in [SigilHouse.FLAME, SigilHouse.ECHOES]:
                    context_bonus += 0.2
            
            # Deep analysis favors reflection houses
            if context.get("deep_analysis", False):
                if house in [SigilHouse.MIRRORS, SigilHouse.MEMORY]:
                    context_bonus += 0.2
            
            # Transformation needs favor purification
            if context.get("transformation", False):
                if house == SigilHouse.PURIFICATION:
                    context_bonus += 0.3
            
            # Connection tasks favor weaving
            if context.get("connection_task", False):
                if house == SigilHouse.WEAVING:
                    context_bonus += 0.3
        
        # House operator availability bonus
        operator = get_house_operator(house)
        availability_bonus = 0.1 if operator else -0.5
        
        # Historical success bonus
        historical_bonus = self._get_historical_success_bonus(profile.tracer_id, house)
        
        # Combine all factors
        total_score = (base_score * 0.6 +          # 60% base affinity
                      capacity_factor * 0.15 +     # 15% capacity
                      context_bonus +              # Variable context
                      availability_bonus +         # Operator availability
                      historical_bonus * 0.1)      # 10% historical success
        
        return min(1.0, max(0.0, total_score))
    
    def _get_historical_success_bonus(self, tracer_id: str, house: SigilHouse) -> float:
        """Get historical success bonus for tracer-house combination"""
        
        # Look at recent alignment history
        recent_alignments = [a for a in self.alignment_history[-50:] 
                           if a.tracer_id == tracer_id and a.target_house == house]
        
        if not recent_alignments:
            return 0.0
        
        # Calculate success rate
        strong_alignments = [a for a in recent_alignments if a.is_strong_alignment()]
        success_rate = len(strong_alignments) / len(recent_alignments)
        
        # Convert to bonus (-0.2 to +0.2 range)
        return (success_rate - 0.5) * 0.4
    
    def execute_alignment(self, alignment_result: AlignmentResult) -> Dict[str, Any]:
        """Execute a tracer-house alignment"""
        
        tracer_id = alignment_result.tracer_id
        target_house = alignment_result.target_house
        
        # Check if tracer exists
        if tracer_id not in self.tracer_profiles:
            return {
                "success": False,
                "error": f"Tracer {tracer_id} not registered",
                "alignment_id": None
            }
        
        # Check house capacity
        if self.current_loads[target_house] >= self.house_capacities[target_house]:
            return {
                "success": False,
                "error": f"House {target_house.value} at capacity",
                "current_load": self.current_loads[target_house],
                "capacity": self.house_capacities[target_house],
                "alignment_id": None
            }
        
        # Update tracer state
        profile = self.tracer_profiles[tracer_id]
        profile.current_state = TracerState.ALIGNED
        profile.alignment_history.append({
            "timestamp": time.time(),
            "house": target_house.value,
            "alignment_strength": alignment_result.alignment_strength,
            "confidence": alignment_result.confidence
        })
        
        # Update house load
        self.current_loads[target_house] += 1
        
        # Store active alignment
        self.active_alignments[tracer_id] = alignment_result
        
        # Update statistics
        self.total_alignments += 1
        if alignment_result.is_strong_alignment():
            self.successful_alignments += 1
        
        # Add to history
        self.alignment_history.append(alignment_result)
        
        # Emit alignment signal
        emit_signal(
            SignalType.CONSCIOUSNESS,
            "tracer_house_alignment",
            {
                "event": "tracer_aligned",
                "tracer_id": tracer_id,
                "tracer_type": profile.tracer_type.value,
                "target_house": target_house.value,
                "alignment_strength": alignment_result.alignment_strength,
                "confidence": alignment_result.confidence
            }
        )
        
        logger.info(f"🎯 Aligned {profile.tracer_type.value} tracer {tracer_id} to {target_house.value} house")
        
        return {
            "success": True,
            "alignment_id": tracer_id,
            "target_house": target_house.value,
            "alignment_strength": alignment_result.alignment_strength,
            "confidence": alignment_result.confidence,
            "house_load": self.current_loads[target_house],
            "house_capacity": self.house_capacities[target_house]
        }
    
    def release_alignment(self, tracer_id: str) -> Dict[str, Any]:
        """Release a tracer from its current house alignment"""
        
        if tracer_id not in self.active_alignments:
            return {
                "success": False,
                "error": f"Tracer {tracer_id} not currently aligned"
            }
        
        alignment = self.active_alignments[tracer_id]
        target_house = alignment.target_house
        
        # Update tracer state
        if tracer_id in self.tracer_profiles:
            self.tracer_profiles[tracer_id].current_state = TracerState.ACTIVE
        
        # Update house load
        self.current_loads[target_house] = max(0, self.current_loads[target_house] - 1)
        
        # Remove from active alignments
        del self.active_alignments[tracer_id]
        
        logger.info(f"🎯 Released tracer {tracer_id} from {target_house.value} house")
        
        return {
            "success": True,
            "tracer_id": tracer_id,
            "released_from": target_house.value,
            "house_load": self.current_loads[target_house]
        }
    
    def get_alignment_status(self, tracer_id: Optional[str] = None) -> Dict[str, Any]:
        """Get alignment status for a specific tracer or all tracers"""
        
        if tracer_id:
            if tracer_id not in self.tracer_profiles:
                return {"error": f"Tracer {tracer_id} not found"}
            
            profile = self.tracer_profiles[tracer_id]
            active_alignment = self.active_alignments.get(tracer_id)
            
            return {
                "tracer_id": tracer_id,
                "tracer_type": profile.tracer_type.value,
                "current_state": profile.current_state.value,
                "active_alignment": {
                    "house": active_alignment.target_house.value,
                    "strength": active_alignment.alignment_strength,
                    "confidence": active_alignment.confidence
                } if active_alignment else None,
                "house_affinities": {house.value: score for house, score in profile.house_affinities.items()},
                "alignment_history_count": len(profile.alignment_history)
            }
        
        else:
            # Return system-wide status
            house_loads = {house.value: load for house, load in self.current_loads.items()}
            house_capacities = {house.value: cap for house, cap in self.house_capacities.items()}
            
            tracer_states = defaultdict(int)
            for profile in self.tracer_profiles.values():
                tracer_states[profile.current_state.value] += 1
            
            return {
                "total_tracers": len(self.tracer_profiles),
                "active_alignments": len(self.active_alignments),
                "house_loads": house_loads,
                "house_capacities": house_capacities,
                "tracer_states": dict(tracer_states),
                "alignment_statistics": {
                    "total_alignments": self.total_alignments,
                    "successful_alignments": self.successful_alignments,
                    "success_rate": (self.successful_alignments / self.total_alignments * 100) if self.total_alignments > 0 else 0
                }
            }
    
    def optimize_alignments(self) -> Dict[str, Any]:
        """Optimize current alignments for better distribution and performance"""
        
        optimization_results = {
            "tracers_realigned": 0,
            "load_balancing_improvements": {},
            "performance_improvements": []
        }
        
        # Identify overloaded houses
        overloaded_houses = [house for house, load in self.current_loads.items() 
                           if load > self.house_capacities[house] * 0.8]
        
        # Identify underutilized houses
        underutilized_houses = [house for house, load in self.current_loads.items() 
                              if load < self.house_capacities[house] * 0.3]
        
        # Realign tracers from overloaded to underutilized houses
        for overloaded_house in overloaded_houses:
            # Find tracers in this house with good alternative alignments
            candidates_for_realignment = []
            
            for tracer_id, alignment in self.active_alignments.items():
                if alignment.target_house == overloaded_house:
                    # Check if tracer has good alternatives
                    for alt_house, alt_score in alignment.alternative_houses:
                        if alt_house in underutilized_houses and alt_score > 0.6:
                            candidates_for_realignment.append((tracer_id, alt_house, alt_score))
            
            # Realign best candidates
            candidates_for_realignment.sort(key=lambda x: x[2], reverse=True)
            
            for tracer_id, new_house, score in candidates_for_realignment[:2]:  # Max 2 per house
                # Release from current house
                self.release_alignment(tracer_id)
                
                # Create new alignment
                profile = self.tracer_profiles[tracer_id]
                new_alignment = AlignmentResult(
                    tracer_id=tracer_id,
                    target_house=new_house,
                    alignment_strength=score,
                    confidence=0.8,
                    reasoning=[f"Optimized from {overloaded_house.value} to {new_house.value}"]
                )
                
                # Execute new alignment
                result = self.execute_alignment(new_alignment)
                if result["success"]:
                    optimization_results["tracers_realigned"] += 1
        
        # Update load balancing improvements
        for house in SigilHouse:
            optimization_results["load_balancing_improvements"][house.value] = {
                "before": self.current_loads[house],
                "capacity": self.house_capacities[house],
                "utilization": self.current_loads[house] / self.house_capacities[house]
            }
        
        logger.info(f"🎯 Alignment optimization completed - {optimization_results['tracers_realigned']} tracers realigned")
        
        return optimization_results

# Global alignment system instance
tracer_house_alignment = TracerHouseAlignmentSystem()

# Export key functions for easy access
def register_tracer(tracer_id: str, tracer_type: TracerType, 
                   custom_attributes: Optional[Dict[str, Any]] = None) -> TracerProfile:
    """Register a tracer in the alignment system"""
    return tracer_house_alignment.register_tracer(tracer_id, tracer_type, custom_attributes)

def align_tracer(tracer_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Calculate and execute optimal alignment for a tracer"""
    alignment_result = tracer_house_alignment.calculate_optimal_alignment(tracer_id, context)
    return tracer_house_alignment.execute_alignment(alignment_result)

def release_tracer(tracer_id: str) -> Dict[str, Any]:
    """Release a tracer from its current alignment"""
    return tracer_house_alignment.release_alignment(tracer_id)

def get_alignment_status(tracer_id: Optional[str] = None) -> Dict[str, Any]:
    """Get alignment status"""
    return tracer_house_alignment.get_alignment_status(tracer_id)
