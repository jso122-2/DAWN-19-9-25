#!/usr/bin/env python3
"""
DAWN Consensus Engine - Multi-Module Decision Making System
===========================================================

Coordinate decisions across multiple DAWN subsystems for unified consciousness.
Provides weighted voting, consensus building, and emergency override mechanisms
to achieve coherent decision-making across fragmented subsystems.
"""

import time
import uuid
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of decisions that require consensus."""
    SIGIL_ACTIVATION = "sigil_activation"
    MEMORY_REBLOOM = "memory_rebloom"
    RECURSIVE_DEPTH_CHANGE = "recursive_depth_change"
    SYSTEM_STATE_TRANSITION = "system_state_transition"
    EMERGENCY_OVERRIDE = "emergency_override"
    GENERAL_COORDINATION = "general_coordination"

class DecisionStatus(Enum):
    """Status of a decision request."""
    PENDING = "pending"
    GATHERING_OPINIONS = "gathering_opinions"
    CALCULATING_CONSENSUS = "calculating_consensus"
    CONSENSUS_REACHED = "consensus_reached"
    CONSENSUS_FAILED = "consensus_failed"
    EXECUTED = "executed"
    CANCELLED = "cancelled"

class VoteConfidence(Enum):
    """Confidence levels for module votes."""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 1.0

@dataclass
class ModuleOpinion:
    """Represents a module's opinion on a decision."""
    module_name: str
    decision_id: str
    vote: Union[bool, float, str]  # Support different vote types
    confidence: VoteConfidence
    reasoning: str
    data_quality: float
    response_time: float
    timestamp: datetime
    contextual_factors: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecisionRequest:
    """Represents a decision that needs consensus."""
    decision_id: str
    decision_type: DecisionType
    requesting_module: str
    context_data: Dict[str, Any]
    required_modules: List[str]
    optional_modules: List[str]
    timeout_seconds: float
    emergency: bool
    created_at: datetime
    description: str
    expected_outcome_type: type = bool  # Expected type of consensus result

@dataclass
class ConsensusResult:
    """Result of a consensus calculation."""
    decision_id: str
    consensus_achieved: bool
    final_decision: Any
    confidence_score: float
    participating_modules: List[str]
    abstaining_modules: List[str]
    consensus_strength: float
    reasoning_summary: str
    execution_recommendation: str
    calculated_at: datetime
    vote_breakdown: Dict[str, ModuleOpinion] = field(default_factory=dict)

class ConsensusEngine:
    """
    Multi-module decision making system for unified consciousness.
    
    Coordinates decisions across DAWN subsystems using weighted voting,
    consensus building, and emergency override mechanisms.
    """
    
    def __init__(self, consciousness_bus=None, decision_timeout: float = 5.0):
        """Initialize the consensus engine."""
        self.engine_id = str(uuid.uuid4())
        
        # Use DAWN singleton if consciousness_bus not provided
        if consciousness_bus is None:
            try:
                from dawn.core.singleton import get_dawn
                dawn_system = get_dawn()
                self.consciousness_bus = dawn_system.consciousness_bus
                self.telemetry_system = dawn_system.telemetry_system
                logger.info("🌅 Consensus engine using DAWN singleton")
            except ImportError:
                logger.warning("DAWN singleton not available, consciousness_bus required")
                raise ValueError("consciousness_bus required when DAWN singleton not available")
        else:
            self.consciousness_bus = consciousness_bus
            self.telemetry_system = None
            
        self.creation_time = datetime.now()
        
        # Decision coordination
        self.active_decisions = {}  # Dict[str, DecisionRequest]
        self.decision_history = deque(maxlen=1000)
        self.module_opinions = defaultdict(dict)  # Dict[str, Dict[str, ModuleOpinion]]
        
        # Module expertise weights for different decision types
        self.decision_weights = {
            'entropy_analyzer': {
                DecisionType.SIGIL_ACTIVATION: 0.8,
                DecisionType.SYSTEM_STATE_TRANSITION: 0.7,
                DecisionType.GENERAL_COORDINATION: 0.3
            },
            'owl_bridge': {
                DecisionType.RECURSIVE_DEPTH_CHANGE: 0.9,
                DecisionType.SYSTEM_STATE_TRANSITION: 0.8,
                DecisionType.GENERAL_COORDINATION: 0.6
            },
            'symbolic_anatomy': {
                DecisionType.SIGIL_ACTIVATION: 0.7,
                DecisionType.MEMORY_REBLOOM: 0.6,
                DecisionType.SYSTEM_STATE_TRANSITION: 0.5,
                DecisionType.GENERAL_COORDINATION: 0.4
            },
            'memory_router': {
                DecisionType.MEMORY_REBLOOM: 0.8,
                DecisionType.SYSTEM_STATE_TRANSITION: 0.6,
                DecisionType.GENERAL_COORDINATION: 0.5
            },
            'recursive_bubble': {
                DecisionType.RECURSIVE_DEPTH_CHANGE: 0.8,
                DecisionType.SYSTEM_STATE_TRANSITION: 0.5,
                DecisionType.GENERAL_COORDINATION: 0.4
            },
            'visual_consciousness': {
                DecisionType.SYSTEM_STATE_TRANSITION: 0.4,
                DecisionType.GENERAL_COORDINATION: 0.3
            },
            'artistic_expression': {
                DecisionType.SYSTEM_STATE_TRANSITION: 0.3,
                DecisionType.GENERAL_COORDINATION: 0.3
            }
        }
        
        # Configuration
        self.default_timeout = decision_timeout
        self.consensus_threshold = 0.6  # Minimum consensus strength required
        self.emergency_timeout = 1.0  # Fast timeout for emergency decisions
        
        # Lifecycle control
        self._stopped = False
        
        # Performance tracking
        self.metrics = {
            'decisions_processed': 0,
            'consensus_success_rate': 0.0,
            'average_decision_time': 0.0,
            'module_participation_rates': defaultdict(float),
            'emergency_overrides': 0
        }
        
        # Threading
        self.decision_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="consensus_engine"
        )
        
        # Subscribe to consciousness bus for opinion gathering
        self._setup_consciousness_subscriptions()
        
        logger.info(f"🤝 Consensus Engine initialized: {self.engine_id}")
    
    def stop(self) -> None:
        """Stop the consensus engine and mark as stopped."""
        self._stopped = True
        logger.info(f"🤝 Consensus Engine stopped: {self.engine_id}")
    
    def _setup_consciousness_subscriptions(self) -> None:
        """Set up subscriptions to consciousness bus for decision coordination."""
        if self.consciousness_bus:
            # Subscribe to decision responses
            self.consciousness_bus.subscribe_to_state(
                'consensus_engine',
                self._handle_decision_response,
                ['decision_response', 'emergency_override']
            )
            
            # Register consensus engine with the bus
            self.consciousness_bus.register_module(
                'consensus_engine',
                ['decision_coordination', 'consensus_building', 'emergency_override'],
                {
                    'active_decisions': 'list',
                    'consensus_success_rate': 'float',
                    'decision_load': 'int'
                }
            )
    
    def request_decision(self, decision_type: DecisionType, context_data: Dict[str, Any],
                        requesting_module: str = 'unknown', required_modules: List[str] = None,
                        optional_modules: List[str] = None, timeout_seconds: float = None,
                        emergency: bool = False, description: str = "") -> str:
        """
        Request coordinated decision from relevant modules.
        
        Args:
            decision_type: Type of decision being requested
            context_data: Context information for the decision
            requesting_module: Module requesting the decision
            required_modules: Modules that must participate
            optional_modules: Modules that may participate
            timeout_seconds: Custom timeout (uses default if None)
            emergency: Whether this is an emergency decision
            description: Human-readable description of the decision
            
        Returns:
            Decision ID for tracking the request
        """
        decision_id = str(uuid.uuid4())
        
        # Determine participating modules based on decision type
        if required_modules is None:
            required_modules = self._get_required_modules_for_decision(decision_type)
        if optional_modules is None:
            optional_modules = self._get_optional_modules_for_decision(decision_type)
        
        # Set timeout
        if timeout_seconds is None:
            timeout_seconds = self.emergency_timeout if emergency else self.default_timeout
        
        # Create decision request
        decision_request = DecisionRequest(
            decision_id=decision_id,
            decision_type=decision_type,
            requesting_module=requesting_module,
            context_data=context_data,
            required_modules=required_modules,
            optional_modules=optional_modules,
            timeout_seconds=timeout_seconds,
            emergency=emergency,
            created_at=datetime.now(),
            description=description
        )
        
        with self.decision_lock:
            self.active_decisions[decision_id] = decision_request
        
        # Start decision process
        self.executor.submit(self._process_decision, decision_request)
        
        logger.info(f"🤝 Decision requested: {decision_type.value} by {requesting_module}")
        return decision_id
    
    def _process_decision(self, decision_request: DecisionRequest) -> ConsensusResult:
        """Process a decision request through the full consensus pipeline."""
        try:
            # Phase 1: Gather module opinions
            opinions = self.gather_module_opinions(decision_request)
            
            # Phase 2: Calculate consensus
            consensus_result = self.calculate_consensus(
                decision_request.decision_id,
                opinions,
                self._get_decision_weights(decision_request.decision_type)
            )
            
            # Phase 3: Execute unified decision if consensus reached
            if consensus_result.consensus_achieved:
                execution_result = self.execute_unified_decision(consensus_result)
                consensus_result.execution_recommendation = f"Executed: {execution_result}"
            else:
                consensus_result.execution_recommendation = "Consensus failed - no action taken"
            
            # Update metrics
            self._update_decision_metrics(consensus_result)
            
            # Store in history
            self.decision_history.append(consensus_result)
            
            # Clean up active decision
            with self.decision_lock:
                self.active_decisions.pop(decision_request.decision_id, None)
            
            return consensus_result
            
        except Exception as e:
            logger.error(f"Error processing decision {decision_request.decision_id}: {e}")
            
            # Create failed consensus result
            failed_result = ConsensusResult(
                decision_id=decision_request.decision_id,
                consensus_achieved=False,
                final_decision=None,
                confidence_score=0.0,
                participating_modules=[],
                abstaining_modules=decision_request.required_modules,
                consensus_strength=0.0,
                reasoning_summary=f"Decision processing failed: {e}",
                execution_recommendation="Error - no action taken",
                calculated_at=datetime.now()
            )
            
            self.decision_history.append(failed_result)
            
            with self.decision_lock:
                self.active_decisions.pop(decision_request.decision_id, None)
            
            return failed_result
    
    def gather_module_opinions(self, decision_request: DecisionRequest) -> Dict[str, ModuleOpinion]:
        """
        Collect input from all relevant subsystems.
        
        Args:
            decision_request: The decision requiring opinions
            
        Returns:
            Dictionary mapping module names to their opinions
        """
        opinions = {}
        all_modules = decision_request.required_modules + decision_request.optional_modules
        
        # Broadcast decision request via consciousness bus
        if self.consciousness_bus:
            event_data = {
                'decision_id': decision_request.decision_id,
                'decision_type': decision_request.decision_type.value,
                'context_data': decision_request.context_data,
                'timeout_seconds': decision_request.timeout_seconds,
                'emergency': decision_request.emergency,
                'description': decision_request.description
            }
            
            self.consciousness_bus.broadcast_event(
                'decision_request',
                event_data,
                'consensus_engine',
                target_modules=all_modules
            )
        
        # Wait for responses with timeout
        start_time = time.time()
        timeout = decision_request.timeout_seconds
        
        while time.time() - start_time < timeout:
            # Check if we have enough responses
            required_responses = set(decision_request.required_modules)
            received_responses = set(opinions.keys())
            
            if required_responses.issubset(received_responses):
                break
            
            # Check for new opinions
            with self.decision_lock:
                decision_opinions = self.module_opinions.get(decision_request.decision_id, {})
                for module_name, opinion in decision_opinions.items():
                    if module_name not in opinions:
                        opinions[module_name] = opinion
            
            time.sleep(0.1)  # Brief pause before next check
        
        # Handle missing required modules
        missing_required = set(decision_request.required_modules) - set(opinions.keys())
        for module_name in missing_required:
            # Create default abstention opinion
            opinions[module_name] = ModuleOpinion(
                module_name=module_name,
                decision_id=decision_request.decision_id,
                vote=None,  # Abstention
                confidence=VoteConfidence.VERY_LOW,
                reasoning="Module did not respond within timeout",
                data_quality=0.0,
                response_time=timeout,
                timestamp=datetime.now()
            )
        
        logger.info(f"🤝 Gathered {len(opinions)} opinions for decision {decision_request.decision_id}")
        return opinions
    
    def calculate_consensus(self, decision_id: str, opinions: Dict[str, ModuleOpinion],
                          weights: Dict[str, float]) -> ConsensusResult:
        """
        Calculate unified decision from module inputs.
        
        Args:
            decision_id: ID of the decision
            opinions: Module opinions collected
            weights: Weighted influence of each module
            
        Returns:
            ConsensusResult with the calculated consensus
        """
        if not opinions:
            return ConsensusResult(
                decision_id=decision_id,
                consensus_achieved=False,
                final_decision=None,
                confidence_score=0.0,
                participating_modules=[],
                abstaining_modules=[],
                consensus_strength=0.0,
                reasoning_summary="No opinions received",
                execution_recommendation="Cannot proceed without input",
                calculated_at=datetime.now()
            )
        
        # Separate participating and abstaining modules
        participating_modules = []
        abstaining_modules = []
        valid_opinions = {}
        
        for module_name, opinion in opinions.items():
            if opinion.vote is not None:
                participating_modules.append(module_name)
                valid_opinions[module_name] = opinion
            else:
                abstaining_modules.append(module_name)
        
        if not valid_opinions:
            return ConsensusResult(
                decision_id=decision_id,
                consensus_achieved=False,
                final_decision=None,
                confidence_score=0.0,
                participating_modules=[],
                abstaining_modules=list(opinions.keys()),
                consensus_strength=0.0,
                reasoning_summary="All modules abstained",
                execution_recommendation="Cannot proceed - no valid votes",
                calculated_at=datetime.now()
            )
        
        # Calculate weighted consensus for different vote types
        if all(isinstance(op.vote, bool) for op in valid_opinions.values()):
            # Boolean consensus
            consensus_result = self._calculate_boolean_consensus(
                decision_id, valid_opinions, weights, participating_modules, abstaining_modules
            )
        elif all(isinstance(op.vote, (int, float)) for op in valid_opinions.values()):
            # Numeric consensus
            consensus_result = self._calculate_numeric_consensus(
                decision_id, valid_opinions, weights, participating_modules, abstaining_modules
            )
        else:
            # Mixed or string consensus - use majority voting
            consensus_result = self._calculate_majority_consensus(
                decision_id, valid_opinions, weights, participating_modules, abstaining_modules
            )
        
        # Add vote breakdown
        consensus_result.vote_breakdown = opinions
        
        logger.info(f"🤝 Consensus calculated for {decision_id}: "
                   f"{'✅ Achieved' if consensus_result.consensus_achieved else '❌ Failed'} "
                   f"(strength: {consensus_result.consensus_strength:.3f})")
        
        return consensus_result
    
    def _calculate_boolean_consensus(self, decision_id: str, opinions: Dict[str, ModuleOpinion],
                                   weights: Dict[str, float], participating: List[str],
                                   abstaining: List[str]) -> ConsensusResult:
        """Calculate consensus for boolean (True/False) decisions."""
        weighted_yes = 0.0
        weighted_no = 0.0
        total_weight = 0.0
        confidence_sum = 0.0
        
        reasoning_parts = []
        
        for module_name, opinion in opinions.items():
            module_weight = weights.get(module_name, 0.3)  # Default weight
            confidence_multiplier = opinion.confidence.value
            adjusted_weight = module_weight * confidence_multiplier
            
            if opinion.vote:
                weighted_yes += adjusted_weight
                reasoning_parts.append(f"{module_name}: YES ({opinion.reasoning})")
            else:
                weighted_no += adjusted_weight
                reasoning_parts.append(f"{module_name}: NO ({opinion.reasoning})")
            
            total_weight += adjusted_weight
            confidence_sum += confidence_multiplier
        
        # Calculate consensus
        if total_weight > 0:
            yes_ratio = weighted_yes / total_weight
            consensus_strength = max(yes_ratio, 1 - yes_ratio)  # Strength of majority
            final_decision = yes_ratio > 0.5
            confidence_score = confidence_sum / len(opinions)
        else:
            yes_ratio = 0.5
            consensus_strength = 0.0
            final_decision = False
            confidence_score = 0.0
        
        consensus_achieved = (
            consensus_strength >= self.consensus_threshold and 
            confidence_score >= 0.4
        )
        
        return ConsensusResult(
            decision_id=decision_id,
            consensus_achieved=consensus_achieved,
            final_decision=final_decision,
            confidence_score=confidence_score,
            participating_modules=participating,
            abstaining_modules=abstaining,
            consensus_strength=consensus_strength,
            reasoning_summary="; ".join(reasoning_parts),
            execution_recommendation="Proceed with decision" if consensus_achieved else "Insufficient consensus",
            calculated_at=datetime.now()
        )
    
    def _calculate_numeric_consensus(self, decision_id: str, opinions: Dict[str, ModuleOpinion],
                                   weights: Dict[str, float], participating: List[str],
                                   abstaining: List[str]) -> ConsensusResult:
        """Calculate consensus for numeric decisions."""
        weighted_sum = 0.0
        total_weight = 0.0
        confidence_sum = 0.0
        values = []
        
        reasoning_parts = []
        
        for module_name, opinion in opinions.items():
            module_weight = weights.get(module_name, 0.3)
            confidence_multiplier = opinion.confidence.value
            adjusted_weight = module_weight * confidence_multiplier
            
            weighted_sum += opinion.vote * adjusted_weight
            total_weight += adjusted_weight
            confidence_sum += confidence_multiplier
            values.append(opinion.vote)
            
            reasoning_parts.append(f"{module_name}: {opinion.vote} ({opinion.reasoning})")
        
        if total_weight > 0:
            final_decision = weighted_sum / total_weight
            confidence_score = confidence_sum / len(opinions)
            
            # Calculate consensus strength based on variance
            mean_value = sum(values) / len(values)
            variance = sum((v - mean_value) ** 2 for v in values) / len(values)
            consensus_strength = max(0.0, 1.0 - (variance / (mean_value + 1e-6)))
        else:
            final_decision = 0.0
            confidence_score = 0.0
            consensus_strength = 0.0
        
        consensus_achieved = (
            consensus_strength >= self.consensus_threshold and
            confidence_score >= 0.4
        )
        
        return ConsensusResult(
            decision_id=decision_id,
            consensus_achieved=consensus_achieved,
            final_decision=final_decision,
            confidence_score=confidence_score,
            participating_modules=participating,
            abstaining_modules=abstaining,
            consensus_strength=consensus_strength,
            reasoning_summary="; ".join(reasoning_parts),
            execution_recommendation="Proceed with numeric decision" if consensus_achieved else "Values too divergent",
            calculated_at=datetime.now()
        )
    
    def _calculate_majority_consensus(self, decision_id: str, opinions: Dict[str, ModuleOpinion],
                                    weights: Dict[str, float], participating: List[str],
                                    abstaining: List[str]) -> ConsensusResult:
        """Calculate consensus using majority voting for mixed vote types."""
        vote_counts = defaultdict(float)
        total_weight = 0.0
        confidence_sum = 0.0
        
        reasoning_parts = []
        
        for module_name, opinion in opinions.items():
            module_weight = weights.get(module_name, 0.3)
            confidence_multiplier = opinion.confidence.value
            adjusted_weight = module_weight * confidence_multiplier
            
            vote_key = str(opinion.vote)
            vote_counts[vote_key] += adjusted_weight
            total_weight += adjusted_weight
            confidence_sum += confidence_multiplier
            
            reasoning_parts.append(f"{module_name}: {opinion.vote} ({opinion.reasoning})")
        
        if vote_counts:
            # Find majority vote
            majority_vote = max(vote_counts.items(), key=lambda x: x[1])
            majority_weight = majority_vote[1]
            final_decision = majority_vote[0]
            
            # Try to convert back to original type
            try:
                if final_decision.lower() in ('true', 'false'):
                    final_decision = final_decision.lower() == 'true'
                elif '.' in final_decision:
                    final_decision = float(final_decision)
                else:
                    final_decision = int(final_decision)
            except (ValueError, AttributeError):
                pass  # Keep as string
            
            consensus_strength = majority_weight / total_weight if total_weight > 0 else 0.0
            confidence_score = confidence_sum / len(opinions)
        else:
            final_decision = None
            consensus_strength = 0.0
            confidence_score = 0.0
        
        # Apply confidence floor for consensus decisions
        confidence_floor = 0.6  # Minimum confidence required
        
        consensus_achieved = (
            consensus_strength >= self.consensus_threshold and
            confidence_score >= confidence_floor
        )
        
        # Generate execution recommendation with confidence consideration
        if confidence_score < confidence_floor:
            execution_recommendation = f"Insufficient confidence ({confidence_score:.1%}) - consider gathering more input or using fallback decision"
        else:
            execution_recommendation = "Proceed with majority decision" if consensus_achieved else "No clear majority"
        
        return ConsensusResult(
            decision_id=decision_id,
            consensus_achieved=consensus_achieved,
            final_decision=final_decision,
            confidence_score=confidence_score,
            participating_modules=participating,
            abstaining_modules=abstaining,
            consensus_strength=consensus_strength,
            reasoning_summary="; ".join(reasoning_parts),
            execution_recommendation=execution_recommendation,
            calculated_at=datetime.now()
        )
    
    def execute_unified_decision(self, consensus_result: ConsensusResult) -> str:
        """
        Execute decision with full system coordination.
        
        Args:
            consensus_result: Result of consensus calculation
            
        Returns:
            Execution result description
        """
        if not consensus_result.consensus_achieved:
            return "Cannot execute - no consensus achieved"
        
        try:
            # Broadcast decision execution via consciousness bus
            if self.consciousness_bus:
                execution_event = {
                    'decision_id': consensus_result.decision_id,
                    'final_decision': consensus_result.final_decision,
                    'confidence_score': consensus_result.confidence_score,
                    'consensus_strength': consensus_result.consensus_strength,
                    'participating_modules': consensus_result.participating_modules,
                    'execution_timestamp': datetime.now().isoformat()
                }
                
                self.consciousness_bus.broadcast_event(
                    'decision_execution',
                    execution_event,
                    'consensus_engine',
                    target_modules=consensus_result.participating_modules
                )
            
            # Update metrics
            self.metrics['decisions_processed'] += 1
            
            logger.info(f"🤝 Decision executed: {consensus_result.decision_id}")
            return f"Successfully executed decision: {consensus_result.final_decision}"
            
        except Exception as e:
            logger.error(f"Failed to execute decision {consensus_result.decision_id}: {e}")
            return f"Execution failed: {e}"
    
    def emergency_override(self, decision_type: DecisionType, override_decision: Any,
                          override_reason: str, overriding_module: str = 'emergency_system') -> str:
        """Execute emergency override bypassing normal consensus."""
        # Don't execute emergency overrides if system is stopped
        if hasattr(self, '_stopped') and self._stopped:
            logger.debug(f"Skipping emergency override {decision_type.value} - consensus engine stopped")
            return f"Emergency override skipped - system stopped"
        
        override_id = str(uuid.uuid4())
        
        # Create emergency consensus result
        emergency_result = ConsensusResult(
            decision_id=override_id,
            consensus_achieved=True,
            final_decision=override_decision,
            confidence_score=1.0,
            participating_modules=[overriding_module],
            abstaining_modules=[],
            consensus_strength=1.0,
            reasoning_summary=f"EMERGENCY OVERRIDE: {override_reason}",
            execution_recommendation="Execute immediately - emergency override",
            calculated_at=datetime.now()
        )
        
        # Execute immediately
        execution_result = self.execute_unified_decision(emergency_result)
        
        # Update metrics
        self.metrics['emergency_overrides'] += 1
        
        # Store in history
        self.decision_history.append(emergency_result)
        
        logger.warning(f"🚨 Emergency override executed: {decision_type.value} -> {override_decision}")
        return f"Emergency override {override_id}: {execution_result}"
    
    def get_recent_decisions(self, limit: int = 10) -> List[ConsensusResult]:
        """Get recent consensus decisions."""
        return list(self.decision_history)[-limit:]
    
    def get_consensus_metrics(self) -> Dict[str, Any]:
        """Get consensus engine performance metrics."""
        if self.metrics['decisions_processed'] > 0:
            success_rate = len([d for d in self.decision_history if d.consensus_achieved]) / len(self.decision_history)
        else:
            success_rate = 0.0
        
        return {
            'engine_id': self.engine_id,
            'uptime_seconds': (datetime.now() - self.creation_time).total_seconds(),
            'active_decisions': len(self.active_decisions),
            'decisions_processed': self.metrics['decisions_processed'],
            'consensus_success_rate': success_rate,
            'emergency_overrides': self.metrics['emergency_overrides'],
            'average_decision_time': self.metrics['average_decision_time'],
            'module_participation_rates': dict(self.metrics['module_participation_rates'])
        }
    
    def _handle_decision_response(self, event) -> None:
        """Handle decision response events from modules."""
        try:
            event_data = event.data
            decision_id = event_data.get('decision_id')
            module_name = event.source_module
            
            if not decision_id or decision_id not in self.active_decisions:
                return
            
            # Create module opinion from event data
            opinion = ModuleOpinion(
                module_name=module_name,
                decision_id=decision_id,
                vote=event_data.get('vote'),
                confidence=VoteConfidence(event_data.get('confidence', 0.6)),
                reasoning=event_data.get('reasoning', ''),
                data_quality=event_data.get('data_quality', 0.8),
                response_time=event_data.get('response_time', 0.0),
                timestamp=datetime.now(),
                contextual_factors=event_data.get('contextual_factors', {})
            )
            
            # Store opinion
            with self.decision_lock:
                self.module_opinions[decision_id][module_name] = opinion
            
            logger.debug(f"🤝 Received opinion from {module_name} for decision {decision_id}")
            
        except Exception as e:
            logger.error(f"Error handling decision response: {e}")
    
    def _get_required_modules_for_decision(self, decision_type: DecisionType) -> List[str]:
        """Get required modules for a decision type."""
        requirements = {
            DecisionType.SIGIL_ACTIVATION: ['owl_bridge', 'symbolic_anatomy', 'entropy_analyzer'],
            DecisionType.MEMORY_REBLOOM: ['memory_router', 'symbolic_anatomy'],
            DecisionType.RECURSIVE_DEPTH_CHANGE: ['recursive_bubble', 'owl_bridge'],
            DecisionType.SYSTEM_STATE_TRANSITION: ['entropy_analyzer', 'owl_bridge', 'symbolic_anatomy'],
            DecisionType.EMERGENCY_OVERRIDE: [],  # No requirements for emergency
            DecisionType.GENERAL_COORDINATION: ['owl_bridge']
        }
        return requirements.get(decision_type, [])
    
    def _get_optional_modules_for_decision(self, decision_type: DecisionType) -> List[str]:
        """Get optional modules for a decision type."""
        optionals = {
            DecisionType.SIGIL_ACTIVATION: ['memory_router', 'visual_consciousness'],
            DecisionType.MEMORY_REBLOOM: ['entropy_analyzer', 'owl_bridge'],
            DecisionType.RECURSIVE_DEPTH_CHANGE: ['entropy_analyzer', 'symbolic_anatomy'],
            DecisionType.SYSTEM_STATE_TRANSITION: ['memory_router', 'visual_consciousness', 'artistic_expression'],
            DecisionType.EMERGENCY_OVERRIDE: [],
            DecisionType.GENERAL_COORDINATION: ['memory_router', 'entropy_analyzer', 'symbolic_anatomy']
        }
        return optionals.get(decision_type, [])
    
    def _get_decision_weights(self, decision_type: DecisionType) -> Dict[str, float]:
        """Get module weights for a specific decision type."""
        weights = {}
        for module_name, module_weights in self.decision_weights.items():
            weights[module_name] = module_weights.get(decision_type, 0.3)  # Default weight
        return weights
    
    def _update_decision_metrics(self, consensus_result: ConsensusResult) -> None:
        """Update consensus engine metrics."""
        # Update participation rates
        total_modules = len(consensus_result.participating_modules + consensus_result.abstaining_modules)
        if total_modules > 0:
            for module in consensus_result.participating_modules:
                current_rate = self.metrics['module_participation_rates'][module]
                self.metrics['module_participation_rates'][module] = (current_rate + 1.0) / 2.0
        
        # Update decision timing (handle negative deltas gracefully)
        raw_processing_time = (consensus_result.calculated_at - datetime.now()).total_seconds()
        if raw_processing_time < 0:
            # Decision time calculated in future - likely clock skew or async timing
            processing_time = 0.001  # Use minimal positive duration
            logger.debug(f"Negative decision time delta ({raw_processing_time:.3f}s) - using minimal duration")
        else:
            processing_time = raw_processing_time
        
        current_avg = self.metrics['average_decision_time']
        decisions_count = self.metrics['decisions_processed']
        
        if decisions_count > 0:
            self.metrics['average_decision_time'] = (
                (current_avg * decisions_count + processing_time) / (decisions_count + 1)
            )
        else:
            self.metrics['average_decision_time'] = processing_time


def demo_consensus_engine():
    """Demonstrate consensus engine functionality."""
    print("🤝 " + "="*60)
    print("🤝 DAWN CONSENSUS ENGINE DEMO")
    print("🤝 " + "="*60)
    print()
    
    # Create mock consciousness bus
    class MockConsciousnessBus:
        def __init__(self):
            self.events = []
        
        def subscribe_to_state(self, module, callback, event_types):
            return "mock_subscription"
        
        def register_module(self, name, capabilities, schema):
            return True
        
        def broadcast_event(self, event_type, data, source, target_modules=None):
            self.events.append((event_type, data, source, target_modules))
            return "mock_event_id"
    
    # Initialize consensus engine
    mock_bus = MockConsciousnessBus()
    consensus = ConsensusEngine(mock_bus)
    print(f"✅ Consensus Engine initialized: {consensus.engine_id}")
    print()
    
    # Simulate decision request
    decision_id = consensus.request_decision(
        DecisionType.SIGIL_ACTIVATION,
        {
            'sigil_type': 'consciousness_unity',
            'activation_strength': 0.8,
            'target_coherence': 0.9
        },
        requesting_module='visual_consciousness',
        description="Activate consciousness unity sigil for coherence improvement"
    )
    
    print(f"📝 Decision requested: {decision_id}")
    print()
    
    # Simulate module opinions
    mock_opinions = {
        'owl_bridge': ModuleOpinion(
            module_name='owl_bridge',
            decision_id=decision_id,
            vote=True,
            confidence=VoteConfidence.HIGH,
            reasoning="Philosophical analysis supports unity sigil activation",
            data_quality=0.9,
            response_time=0.2,
            timestamp=datetime.now()
        ),
        'symbolic_anatomy': ModuleOpinion(
            module_name='symbolic_anatomy',
            decision_id=decision_id,
            vote=True,
            confidence=VoteConfidence.MEDIUM,
            reasoning="Organ states indicate readiness for sigil activation",
            data_quality=0.8,
            response_time=0.3,
            timestamp=datetime.now()
        ),
        'entropy_analyzer': ModuleOpinion(
            module_name='entropy_analyzer',
            decision_id=decision_id,
            vote=False,
            confidence=VoteConfidence.HIGH,
            reasoning="Current entropy levels too high for safe activation",
            data_quality=0.95,
            response_time=0.1,
            timestamp=datetime.now()
        )
    }
    
    print("🗳️ Simulated module opinions:")
    for module, opinion in mock_opinions.items():
        print(f"   {module}: {'✅ YES' if opinion.vote else '❌ NO'} "
              f"(confidence: {opinion.confidence.name})")
        print(f"      Reasoning: {opinion.reasoning}")
    print()
    
    # Calculate consensus
    weights = consensus._get_decision_weights(DecisionType.SIGIL_ACTIVATION)
    consensus_result = consensus.calculate_consensus(decision_id, mock_opinions, weights)
    
    print("🤝 Consensus Result:")
    print(f"   Decision: {'✅ ACHIEVED' if consensus_result.consensus_achieved else '❌ FAILED'}")
    print(f"   Final Decision: {consensus_result.final_decision}")
    print(f"   Confidence: {consensus_result.confidence_score:.3f}")
    print(f"   Consensus Strength: {consensus_result.consensus_strength:.3f}")
    print(f"   Participating: {', '.join(consensus_result.participating_modules)}")
    print(f"   Recommendation: {consensus_result.execution_recommendation}")
    print()
    
    # Test emergency override
    print("🚨 Testing emergency override...")
    override_result = consensus.emergency_override(
        DecisionType.SYSTEM_STATE_TRANSITION,
        "emergency_stabilization",
        "Critical stability threshold breached",
        "safety_monitor"
    )
    print(f"   Override result: {override_result}")
    print()
    
    # Show metrics
    metrics = consensus.get_consensus_metrics()
    print("📊 Consensus Engine Metrics:")
    print(f"   Decisions Processed: {metrics['decisions_processed']}")
    print(f"   Success Rate: {metrics['consensus_success_rate']:.1%}")
    print(f"   Emergency Overrides: {metrics['emergency_overrides']}")
    print(f"   Active Decisions: {metrics['active_decisions']}")
    print()
    
    print("🤝 Demo complete! Consensus engine enables unified decision-making.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    demo_consensus_engine()
