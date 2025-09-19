#!/usr/bin/env python3
"""
🧠 Logical Reasoning Module - DAWN Cognitive Capability
═══════════════════════════════════════════════════════════

Implements logical reasoning capabilities for DAWN's cognitive framework.
Provides structured thinking, inference, and logical analysis.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class ReasoningType(Enum):
    """Types of logical reasoning"""
    DEDUCTIVE = "deductive"      # General to specific
    INDUCTIVE = "inductive"      # Specific to general
    ABDUCTIVE = "abductive"      # Best explanation
    ANALOGICAL = "analogical"    # Similarity-based
    CAUSAL = "causal"           # Cause-effect relationships

class LogicalOperator(Enum):
    """Logical operators for reasoning"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IF_THEN = "if_then"
    IFF = "if_and_only_if"

@dataclass
class Premise:
    """A logical premise for reasoning"""
    statement: str
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Conclusion:
    """A logical conclusion from reasoning"""
    statement: str
    confidence: float
    reasoning_type: ReasoningType
    premises: List[Premise]
    reasoning_chain: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ReasoningContext:
    """Context for reasoning operations"""
    domain: str = "general"
    constraints: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    prior_knowledge: Dict[str, Any] = field(default_factory=dict)

class LogicalReasoning:
    """
    Core logical reasoning capability for DAWN.
    Implements various forms of logical inference and structured thinking.
    """
    
    def __init__(self):
        self.reasoning_history: List[Conclusion] = []
        self.knowledge_base: Dict[str, Any] = {}
        self.confidence_threshold: float = 0.5
        self.max_reasoning_steps: int = 100
        
        logger.info("🧠 Logical Reasoning module initialized")
    
    def deductive_reasoning(
        self, 
        premises: List[Premise], 
        context: Optional[ReasoningContext] = None
    ) -> Optional[Conclusion]:
        """
        Perform deductive reasoning from premises to conclusion.
        
        Args:
            premises: List of premises for reasoning
            context: Optional reasoning context
            
        Returns:
            Conclusion if valid reasoning path found, None otherwise
        """
        if len(premises) < 2:
            logger.warning("Deductive reasoning requires at least 2 premises")
            return None
        
        context = context or ReasoningContext()
        reasoning_chain = []
        
        # Simple deductive pattern: If A and (A -> B) then B
        for i, premise1 in enumerate(premises):
            for j, premise2 in enumerate(premises[i+1:], i+1):
                if self._is_implication_pattern(premise1, premise2):
                    conclusion_text, confidence = self._apply_modus_ponens(premise1, premise2)
                    reasoning_chain.append(f"Applied modus ponens to premises {i} and {j}")
                    
                    conclusion = Conclusion(
                        statement=conclusion_text,
                        confidence=min(premise1.confidence, premise2.confidence) * confidence,
                        reasoning_type=ReasoningType.DEDUCTIVE,
                        premises=[premise1, premise2],
                        reasoning_chain=reasoning_chain
                    )
                    
                    self.reasoning_history.append(conclusion)
                    logger.info(f"🧠 Deductive conclusion: {conclusion_text} (confidence: {conclusion.confidence:.2f})")
                    return conclusion
        
        logger.info("🧠 No valid deductive reasoning path found")
        return None
    
    def inductive_reasoning(
        self,
        observations: List[Premise],
        context: Optional[ReasoningContext] = None
    ) -> Optional[Conclusion]:
        """
        Perform inductive reasoning from specific observations to general patterns.
        
        Args:
            observations: List of specific observations
            context: Optional reasoning context
            
        Returns:
            General conclusion if pattern found, None otherwise
        """
        if len(observations) < 3:
            logger.warning("Inductive reasoning requires at least 3 observations")
            return None
        
        context = context or ReasoningContext()
        
        # Look for patterns in observations
        pattern = self._identify_pattern(observations)
        if pattern:
            confidence = self._calculate_inductive_confidence(observations, pattern)
            reasoning_chain = [f"Observed pattern in {len(observations)} cases: {pattern}"]
            
            conclusion = Conclusion(
                statement=f"General pattern: {pattern}",
                confidence=confidence,
                reasoning_type=ReasoningType.INDUCTIVE,
                premises=observations,
                reasoning_chain=reasoning_chain
            )
            
            self.reasoning_history.append(conclusion)
            logger.info(f"🧠 Inductive conclusion: {pattern} (confidence: {confidence:.2f})")
            return conclusion
        
        logger.info("🧠 No inductive pattern identified")
        return None
    
    def abductive_reasoning(
        self,
        observation: Premise,
        possible_explanations: List[str],
        context: Optional[ReasoningContext] = None
    ) -> Optional[Conclusion]:
        """
        Perform abductive reasoning to find best explanation for observation.
        
        Args:
            observation: The observation to explain
            possible_explanations: List of candidate explanations
            context: Optional reasoning context
            
        Returns:
            Best explanation conclusion if found, None otherwise
        """
        if not possible_explanations:
            logger.warning("Abductive reasoning requires possible explanations")
            return None
        
        context = context or ReasoningContext()
        
        # Score explanations based on simplicity, plausibility, and explanatory power
        scored_explanations = []
        for explanation in possible_explanations:
            score = self._score_explanation(observation, explanation, context)
            scored_explanations.append((explanation, score))
        
        # Select best explanation
        best_explanation, best_score = max(scored_explanations, key=lambda x: x[1])
        
        reasoning_chain = [
            f"Evaluated {len(possible_explanations)} explanations",
            f"Best explanation score: {best_score:.2f}",
            f"Selected: {best_explanation}"
        ]
        
        conclusion = Conclusion(
            statement=f"Best explanation: {best_explanation}",
            confidence=best_score,
            reasoning_type=ReasoningType.ABDUCTIVE,
            premises=[observation],
            reasoning_chain=reasoning_chain
        )
        
        self.reasoning_history.append(conclusion)
        logger.info(f"🧠 Abductive conclusion: {best_explanation} (confidence: {best_score:.2f})")
        return conclusion
    
    def analogical_reasoning(
        self,
        source_case: Dict[str, Any],
        target_case: Dict[str, Any],
        context: Optional[ReasoningContext] = None
    ) -> Optional[Conclusion]:
        """
        Perform analogical reasoning between source and target cases.
        
        Args:
            source_case: Source case with known properties
            target_case: Target case to reason about
            context: Optional reasoning context
            
        Returns:
            Analogical conclusion if valid similarity found, None otherwise
        """
        context = context or ReasoningContext()
        
        # Calculate structural similarity
        similarity_score = self._calculate_similarity(source_case, target_case)
        
        if similarity_score > self.confidence_threshold:
            # Transfer properties from source to target
            transferred_properties = self._transfer_properties(source_case, target_case, similarity_score)
            
            reasoning_chain = [
                f"Identified similarity between cases (score: {similarity_score:.2f})",
                f"Transferred properties: {list(transferred_properties.keys())}"
            ]
            
            conclusion_text = f"By analogy: {transferred_properties}"
            
            conclusion = Conclusion(
                statement=conclusion_text,
                confidence=similarity_score,
                reasoning_type=ReasoningType.ANALOGICAL,
                premises=[],  # No premises in traditional sense
                reasoning_chain=reasoning_chain
            )
            
            self.reasoning_history.append(conclusion)
            logger.info(f"🧠 Analogical conclusion: {conclusion_text} (confidence: {similarity_score:.2f})")
            return conclusion
        
        logger.info(f"🧠 Insufficient similarity for analogical reasoning (score: {similarity_score:.2f})")
        return None
    
    def causal_reasoning(
        self,
        events: List[Dict[str, Any]],
        context: Optional[ReasoningContext] = None
    ) -> Optional[Conclusion]:
        """
        Perform causal reasoning to identify cause-effect relationships.
        
        Args:
            events: List of events with temporal and causal information
            context: Optional reasoning context
            
        Returns:
            Causal conclusion if relationship found, None otherwise
        """
        if len(events) < 2:
            logger.warning("Causal reasoning requires at least 2 events")
            return None
        
        context = context or ReasoningContext()
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda x: x.get('timestamp', 0))
        
        # Look for causal patterns
        causal_links = self._identify_causal_links(sorted_events)
        
        if causal_links:
            strongest_link = max(causal_links, key=lambda x: x['strength'])
            
            reasoning_chain = [
                f"Analyzed {len(events)} events for causal relationships",
                f"Found {len(causal_links)} potential causal links",
                f"Strongest link: {strongest_link['cause']} → {strongest_link['effect']}"
            ]
            
            conclusion_text = f"Causal relationship: {strongest_link['cause']} causes {strongest_link['effect']}"
            
            conclusion = Conclusion(
                statement=conclusion_text,
                confidence=strongest_link['strength'],
                reasoning_type=ReasoningType.CAUSAL,
                premises=[],  # Events rather than premises
                reasoning_chain=reasoning_chain
            )
            
            self.reasoning_history.append(conclusion)
            logger.info(f"🧠 Causal conclusion: {conclusion_text} (confidence: {strongest_link['strength']:.2f})")
            return conclusion
        
        logger.info("🧠 No causal relationships identified")
        return None
    
    def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get summary of reasoning activities"""
        if not self.reasoning_history:
            return {"total_reasoning_operations": 0}
        
        type_counts = {}
        for conclusion in self.reasoning_history:
            reasoning_type = conclusion.reasoning_type.value
            type_counts[reasoning_type] = type_counts.get(reasoning_type, 0) + 1
        
        avg_confidence = sum(c.confidence for c in self.reasoning_history) / len(self.reasoning_history)
        
        return {
            "total_reasoning_operations": len(self.reasoning_history),
            "reasoning_types": type_counts,
            "average_confidence": avg_confidence,
            "recent_conclusions": [c.statement for c in self.reasoning_history[-5:]],
            "knowledge_base_size": len(self.knowledge_base)
        }
    
    # Helper methods
    def _is_implication_pattern(self, premise1: Premise, premise2: Premise) -> bool:
        """Check if two premises form an implication pattern"""
        # Simple pattern matching for "if A then B" structures
        return ("if" in premise1.statement.lower() and "then" in premise1.statement.lower()) or \
               ("if" in premise2.statement.lower() and "then" in premise2.statement.lower())
    
    def _apply_modus_ponens(self, premise1: Premise, premise2: Premise) -> Tuple[str, float]:
        """Apply modus ponens logical rule"""
        # Simplified implementation
        conclusion_text = f"Conclusion from {premise1.statement} and {premise2.statement}"
        confidence = 0.8  # Base confidence for modus ponens
        return conclusion_text, confidence
    
    def _identify_pattern(self, observations: List[Premise]) -> Optional[str]:
        """Identify patterns in observations"""
        # Simplified pattern detection
        statements = [obs.statement.lower() for obs in observations]
        
        # Look for common words/phrases
        common_words = set()
        for statement in statements:
            words = set(statement.split())
            if not common_words:
                common_words = words
            else:
                common_words &= words
        
        if common_words:
            return f"Common elements: {', '.join(list(common_words)[:3])}"
        
        return None
    
    def _calculate_inductive_confidence(self, observations: List[Premise], pattern: str) -> float:
        """Calculate confidence for inductive reasoning"""
        # Base confidence increases with number of observations
        base_confidence = min(0.9, 0.3 + (len(observations) * 0.1))
        
        # Adjust based on observation confidence
        avg_obs_confidence = sum(obs.confidence for obs in observations) / len(observations)
        
        return base_confidence * avg_obs_confidence
    
    def _score_explanation(self, observation: Premise, explanation: str, context: ReasoningContext) -> float:
        """Score an explanation for abductive reasoning"""
        # Simplified scoring based on length (simplicity) and keyword matching
        simplicity_score = max(0.1, 1.0 - (len(explanation.split()) / 20))
        
        # Check if explanation relates to observation
        obs_words = set(observation.statement.lower().split())
        exp_words = set(explanation.lower().split())
        relevance_score = len(obs_words & exp_words) / max(len(obs_words), 1)
        
        return (simplicity_score + relevance_score) / 2
    
    def _calculate_similarity(self, source: Dict[str, Any], target: Dict[str, Any]) -> float:
        """Calculate structural similarity between cases"""
        common_keys = set(source.keys()) & set(target.keys())
        total_keys = set(source.keys()) | set(target.keys())
        
        if not total_keys:
            return 0.0
        
        return len(common_keys) / len(total_keys)
    
    def _transfer_properties(self, source: Dict[str, Any], target: Dict[str, Any], similarity: float) -> Dict[str, Any]:
        """Transfer properties from source to target based on similarity"""
        transferred = {}
        
        for key, value in source.items():
            if key not in target:
                # Transfer property with confidence based on similarity
                transferred[key] = {"value": value, "confidence": similarity}
        
        return transferred
    
    def _identify_causal_links(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential causal links between events"""
        links = []
        
        for i in range(len(events) - 1):
            for j in range(i + 1, len(events)):
                cause_event = events[i]
                effect_event = events[j]
                
                # Simple temporal and semantic relationship check
                temporal_proximity = self._calculate_temporal_proximity(cause_event, effect_event)
                semantic_relatedness = self._calculate_semantic_relatedness(cause_event, effect_event)
                
                strength = (temporal_proximity + semantic_relatedness) / 2
                
                if strength > self.confidence_threshold:
                    links.append({
                        "cause": cause_event.get("description", f"Event {i}"),
                        "effect": effect_event.get("description", f"Event {j}"),
                        "strength": strength
                    })
        
        return links
    
    def _calculate_temporal_proximity(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> float:
        """Calculate temporal proximity between events"""
        # Simplified: closer in time = higher proximity
        time1 = event1.get('timestamp', 0)
        time2 = event2.get('timestamp', 0)
        
        time_diff = abs(time2 - time1)
        
        # Normalize to 0-1 range (assuming max meaningful difference is 1000 units)
        return max(0.0, 1.0 - (time_diff / 1000))
    
    def _calculate_semantic_relatedness(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> float:
        """Calculate semantic relatedness between events"""
        desc1 = event1.get('description', '').lower().split()
        desc2 = event2.get('description', '').lower().split()
        
        if not desc1 or not desc2:
            return 0.0
        
        common_words = set(desc1) & set(desc2)
        total_words = set(desc1) | set(desc2)
        
        return len(common_words) / len(total_words) if total_words else 0.0


# Global reasoning instance for DAWN
_reasoning_engine = None

def get_logical_reasoning() -> LogicalReasoning:
    """Get the global logical reasoning instance"""
    global _reasoning_engine
    if _reasoning_engine is None:
        _reasoning_engine = LogicalReasoning()
    return _reasoning_engine


# Example usage and testing
if __name__ == "__main__":
    print("🧠 Testing DAWN Logical Reasoning Module")
    print("=" * 50)
    
    reasoning = LogicalReasoning()
    
    # Test deductive reasoning
    premises = [
        Premise("All humans are mortal", confidence=1.0),
        Premise("Socrates is a human", confidence=1.0)
    ]
    
    deductive_result = reasoning.deductive_reasoning(premises)
    if deductive_result:
        print(f"Deductive: {deductive_result.statement}")
    
    # Test inductive reasoning
    observations = [
        Premise("Swan 1 is white", confidence=0.9),
        Premise("Swan 2 is white", confidence=0.9),
        Premise("Swan 3 is white", confidence=0.9),
        Premise("Swan 4 is white", confidence=0.9)
    ]
    
    inductive_result = reasoning.inductive_reasoning(observations)
    if inductive_result:
        print(f"Inductive: {inductive_result.statement}")
    
    # Test abductive reasoning
    observation = Premise("The ground is wet", confidence=1.0)
    explanations = ["It rained", "Sprinkler was on", "Someone spilled water"]
    
    abductive_result = reasoning.abductive_reasoning(observation, explanations)
    if abductive_result:
        print(f"Abductive: {abductive_result.statement}")
    
    # Get summary
    summary = reasoning.get_reasoning_summary()
    print(f"\nReasoning Summary: {summary}")
