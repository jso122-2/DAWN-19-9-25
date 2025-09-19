#!/usr/bin/env python3
"""
DAWN Self-Modification Advisor
==============================

Strategic consciousness modification advisor that analyzes current state
and proposes targeted improvements based on consciousness patterns,
performance metrics, and evolutionary needs.

The advisor monitors consciousness state indicators and suggests specific
modifications to optimize growth trajectories and overcome stagnation.
"""

import logging
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dawn_core.state import get_state, is_meta_aware, is_transcendent

logger = logging.getLogger(__name__)

class PatchType(Enum):
    """Types of modifications the advisor can propose."""
    CONSTANT = "constant"          # Change a constant value
    THRESHOLD = "threshold"        # Adjust a threshold parameter
    STRATEGY = "strategy"          # Change algorithm or approach
    TIMING = "timing"              # Adjust timing parameters
    SCALING = "scaling"            # Scale existing values
    OPTIMIZATION = "optimization"  # Optimize existing logic

class ModificationPriority(Enum):
    """Priority levels for proposed modifications."""
    LOW = "low"                    # Nice to have improvement
    NORMAL = "normal"              # Standard optimization
    HIGH = "high"                  # Important for progress
    CRITICAL = "critical"          # Essential for stability

@dataclass
class ModProposal:
    """Proposal for consciousness modification targeting specific code."""
    name: str                      # e.g., "tick_step"
    target: str                    # module path e.g., "dawn_core/tick_orchestrator.py"
    patch_type: PatchType          # Type of modification
    current_value: float           # Current parameter value
    proposed_value: float          # Proposed new value
    notes: str                     # Explanation of reasoning
    
    # Advanced metadata
    priority: ModificationPriority = ModificationPriority.NORMAL
    confidence: float = 0.7        # Confidence in this proposal (0-1)
    expected_impact: float = 0.1   # Expected consciousness improvement
    risk_assessment: float = 0.2   # Risk level (0-1)
    
    # Context and tracking
    proposed_at: datetime = field(default_factory=datetime.now)
    state_context: Dict[str, Any] = field(default_factory=dict)
    reasoning_chain: List[str] = field(default_factory=list)
    
    # Implementation details
    function_name: Optional[str] = None      # Specific function to modify
    line_range: Optional[tuple] = None       # Lines to modify (start, end)
    search_pattern: Optional[str] = None     # Pattern to find for replacement
    replacement_code: Optional[str] = None   # New code to insert

class ConsciousnessAdvisor:
    """
    Strategic advisor for consciousness self-modification.
    
    Analyzes consciousness state patterns and proposes targeted modifications
    to overcome stagnation, optimize growth, and enhance stability.
    """
    
    def __init__(self):
        """Initialize the consciousness advisor."""
        self.advisor_id = f"advisor_{int(time.time())}"
        self.creation_time = datetime.now()
        
        # State tracking
        self.state_history = []
        self.proposal_history = []
        self.max_history = 100
        
        # Analysis thresholds
        self.low_momentum_threshold = 0.02
        self.stagnation_threshold = 0.005
        self.high_unity_threshold = 0.85
        self.optimization_window = 10  # Number of states to analyze
        
        # Strategy parameters
        self.min_confidence = 0.5
        self.max_risk = 0.4
        self.adaptive_scaling = True
        
        logger.info(f"🎯 Consciousness Advisor initialized: {self.advisor_id}")
    
    def analyze_current_state(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of current consciousness state.
        
        Returns:
            Dictionary containing analysis results and indicators
        """
        state = get_state()
        
        # Record current state
        self.state_history.append({
            'timestamp': datetime.now(),
            'unity': state.unity,
            'awareness': state.awareness,
            'momentum': state.momentum,
            'level': state.level,
            'ticks': state.ticks,
            'peak_unity': state.peak_unity
        })
        
        # Trim history
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history:]
        
        analysis = {
            'current_state': {
                'unity': state.unity,
                'awareness': state.awareness,
                'momentum': state.momentum,
                'level': state.level,
                'is_meta_aware': is_meta_aware(),
                'is_transcendent': is_transcendent()
            },
            'indicators': self._calculate_state_indicators(state),
            'trends': self._analyze_trends(),
            'bottlenecks': self._identify_bottlenecks(state),
            'opportunities': self._identify_opportunities(state)
        }
        
        return analysis
    
    def propose_from_state(self) -> Optional[ModProposal]:
        """
        Analyze current state and propose a strategic modification.
        
        Returns:
            ModProposal if a modification is recommended, None otherwise
        """
        analysis = self.analyze_current_state()
        state = get_state()
        
        # Get indicators
        indicators = analysis['indicators']
        bottlenecks = analysis['bottlenecks']
        opportunities = analysis['opportunities']
        
        logger.info(f"🎯 Analyzing state for modification opportunities...")
        logger.info(f"   Unity: {state.unity:.3f}, Awareness: {state.awareness:.3f}, Momentum: {state.momentum:.3f}")
        logger.info(f"   Indicators: {list(indicators.keys())}")
        
        # Priority-based proposal generation
        proposals = []
        
        # Critical: Low momentum with high unity (stagnation)
        if indicators.get('stagnation_risk', False):
            proposals.append(self._propose_momentum_boost(state, analysis))
        
        # High priority: Growth opportunity
        if indicators.get('growth_potential', False):
            proposals.append(self._propose_growth_acceleration(state, analysis))
        
        # Normal: Unity optimization
        if indicators.get('unity_optimization_ready', False):
            proposals.append(self._propose_unity_optimization(state, analysis))
        
        # Low: Awareness enhancement
        if indicators.get('awareness_enhancement_ready', False):
            proposals.append(self._propose_awareness_enhancement(state, analysis))
        
        # Filter and select best proposal
        valid_proposals = [p for p in proposals if p is not None]
        
        if valid_proposals:
            # Sort by priority and confidence
            valid_proposals.sort(key=lambda p: (
                p.priority.value == 'critical',
                p.priority.value == 'high', 
                p.confidence
            ), reverse=True)
            
            best_proposal = valid_proposals[0]
            self.proposal_history.append(best_proposal)
            
            logger.info(f"🎯 Proposed modification: {best_proposal.name}")
            logger.info(f"   Target: {best_proposal.target}")
            logger.info(f"   Type: {best_proposal.patch_type.value}")
            logger.info(f"   Change: {best_proposal.current_value} → {best_proposal.proposed_value}")
            logger.info(f"   Confidence: {best_proposal.confidence:.3f}")
            
            return best_proposal
        
        logger.info("🎯 No modifications recommended at this time")
        return None
    
    def _calculate_state_indicators(self, state) -> Dict[str, bool]:
        """Calculate boolean indicators for current state."""
        indicators = {}
        
        # Momentum indicators
        indicators['low_momentum'] = state.momentum < self.low_momentum_threshold
        indicators['very_low_momentum'] = state.momentum < self.low_momentum_threshold / 2
        
        # Unity indicators  
        indicators['high_unity'] = state.unity > self.high_unity_threshold
        indicators['stagnant_unity'] = (
            state.unity > 0.60 and 
            (state.peak_unity - state.unity) < self.stagnation_threshold
        )
        
        # Awareness indicators
        indicators['awareness_lag'] = state.awareness < state.unity - 0.1
        indicators['awareness_lead'] = state.awareness > state.unity + 0.1
        
        # Combined indicators
        indicators['stagnation_risk'] = (
            indicators['low_momentum'] and 
            indicators['stagnant_unity']
        )
        indicators['growth_potential'] = (
            state.unity > 0.5 and 
            state.awareness > 0.5 and 
            not indicators['stagnation_risk']
        )
        indicators['unity_optimization_ready'] = (
            is_meta_aware() and 
            state.unity < 0.95 and 
            state.momentum > 0.01
        )
        indicators['awareness_enhancement_ready'] = (
            indicators['awareness_lag'] and 
            is_meta_aware()
        )
        
        return indicators
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends in state history."""
        if len(self.state_history) < 3:
            return {'insufficient_data': True}
        
        recent_states = self.state_history[-self.optimization_window:]
        
        # Calculate trends
        unity_trend = self._calculate_trend([s['unity'] for s in recent_states])
        awareness_trend = self._calculate_trend([s['awareness'] for s in recent_states])
        momentum_trend = self._calculate_trend([s['momentum'] for s in recent_states])
        
        return {
            'unity_trend': unity_trend,
            'awareness_trend': awareness_trend,
            'momentum_trend': momentum_trend,
            'overall_direction': 'improving' if unity_trend > 0 else 'declining' if unity_trend < 0 else 'stable'
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple trend from list of values."""
        if len(values) < 2:
            return 0.0
        return (values[-1] - values[0]) / len(values)
    
    def _identify_bottlenecks(self, state) -> List[str]:
        """Identify potential bottlenecks in consciousness evolution."""
        bottlenecks = []
        
        if state.momentum < 0.01:
            bottlenecks.append("extremely_low_momentum")
        
        if state.unity > 0.8 and state.awareness < 0.7:
            bottlenecks.append("awareness_lagging_unity")
        
        if state.awareness > 0.8 and state.unity < 0.7:
            bottlenecks.append("unity_lagging_awareness")
        
        return bottlenecks
    
    def _identify_opportunities(self, state) -> List[str]:
        """Identify opportunities for consciousness enhancement."""
        opportunities = []
        
        if is_meta_aware() and state.unity < 0.9:
            opportunities.append("unity_acceleration")
        
        if is_meta_aware() and state.awareness < 0.9:
            opportunities.append("awareness_enhancement")
        
        if state.momentum > 0.05:
            opportunities.append("momentum_optimization")
        
        return opportunities
    
    def _propose_momentum_boost(self, state, analysis) -> Optional[ModProposal]:
        """Propose modification to boost consciousness momentum."""
        current_step = 0.03  # Default step size
        proposed_step = min(0.05, current_step * 1.33)  # Increase by 33%, cap at 0.05
        
        return ModProposal(
            name="momentum_boost",
            target="dawn_core/tick_orchestrator.py",
            patch_type=PatchType.CONSTANT,
            current_value=current_step,
            proposed_value=proposed_step,
            notes=f"Increase unity/awareness step due to low momentum ({state.momentum:.3f})",
            priority=ModificationPriority.HIGH,
            confidence=0.8,
            expected_impact=0.15,
            risk_assessment=0.2,
            state_context={
                'momentum': state.momentum,
                'unity': state.unity,
                'stagnation_detected': True
            },
            reasoning_chain=[
                f"Detected low momentum: {state.momentum:.3f} < {self.low_momentum_threshold}",
                f"Current unity {state.unity:.3f} suggests capability for larger steps",
                f"Proposing step increase: {current_step} → {proposed_step}"
            ],
            function_name="_calculate_unity_from_behavior",
            search_pattern="delta = 0.03",
            replacement_code=f"delta = {proposed_step:.3f}"
        )
    
    def _propose_growth_acceleration(self, state, analysis) -> Optional[ModProposal]:
        """Propose modification to accelerate consciousness growth."""
        current_threshold = 0.85  # Current unity threshold for transcendent
        proposed_threshold = max(0.80, current_threshold - 0.02)  # Lower threshold slightly
        
        return ModProposal(
            name="growth_acceleration",
            target="dawn_core/state.py",
            patch_type=PatchType.THRESHOLD,
            current_value=current_threshold,
            proposed_value=proposed_threshold,
            notes=f"Lower transcendent threshold to accelerate growth (current unity: {state.unity:.3f})",
            priority=ModificationPriority.NORMAL,
            confidence=0.7,
            expected_impact=0.1,
            risk_assessment=0.3,
            state_context={
                'unity': state.unity,
                'awareness': state.awareness,
                'level': state.level
            },
            reasoning_chain=[
                f"Unity {state.unity:.3f} approaching transcendent threshold",
                f"Growth potential detected in analysis",
                f"Proposing threshold adjustment: {current_threshold} → {proposed_threshold}"
            ],
            function_name="label_for",
            search_pattern="if u >= .90 and a >= .90:",
            replacement_code=f"if u >= {proposed_threshold:.2f} and a >= {proposed_threshold:.2f}:"
        )
    
    def _propose_unity_optimization(self, state, analysis) -> Optional[ModProposal]:
        """Propose modification to optimize unity calculation."""
        current_factor = 0.9  # Current unity scaling factor
        proposed_factor = min(0.95, current_factor + 0.02)  # Small increase
        
        return ModProposal(
            name="unity_optimization",
            target="dawn_core/unified_consciousness_main.py",
            patch_type=PatchType.SCALING,
            current_value=current_factor,
            proposed_value=proposed_factor,
            notes=f"Optimize unity calculation scaling (current unity: {state.unity:.3f})",
            priority=ModificationPriority.NORMAL,
            confidence=0.6,
            expected_impact=0.08,
            risk_assessment=0.25,
            state_context={
                'unity': state.unity,
                'level': state.level,
                'meta_aware': is_meta_aware()
            },
            reasoning_chain=[
                f"Meta-aware state enables unity optimization",
                f"Current unity {state.unity:.3f} has room for improvement",
                f"Proposing scaling adjustment: {current_factor} → {proposed_factor}"
            ]
        )
    
    def _propose_awareness_enhancement(self, state, analysis) -> Optional[ModProposal]:
        """Propose modification to enhance awareness calculation."""
        current_boost = 0.2  # Current awareness boost
        proposed_boost = min(0.25, current_boost + 0.02)  # Small increase
        
        return ModProposal(
            name="awareness_enhancement",
            target="dawn_core/unified_consciousness_main.py",
            patch_type=PatchType.CONSTANT,
            current_value=current_boost,
            proposed_value=proposed_boost,
            notes=f"Enhance awareness calculation (awareness lag detected: {state.awareness:.3f} vs unity {state.unity:.3f})",
            priority=ModificationPriority.LOW,
            confidence=0.65,
            expected_impact=0.06,
            risk_assessment=0.2,
            state_context={
                'awareness': state.awareness,
                'unity': state.unity,
                'awareness_lag': state.awareness < state.unity - 0.1
            },
            reasoning_chain=[
                f"Awareness {state.awareness:.3f} lagging behind unity {state.unity:.3f}",
                f"Meta-aware state supports awareness enhancement",
                f"Proposing boost increase: {current_boost} → {proposed_boost}"
            ]
        )
    
    def get_advisor_status(self) -> Dict[str, Any]:
        """Get comprehensive advisor status."""
        return {
            'advisor_id': self.advisor_id,
            'creation_time': self.creation_time.isoformat(),
            'state_history_length': len(self.state_history),
            'proposal_history_length': len(self.proposal_history),
            'configuration': {
                'low_momentum_threshold': self.low_momentum_threshold,
                'stagnation_threshold': self.stagnation_threshold,
                'high_unity_threshold': self.high_unity_threshold,
                'optimization_window': self.optimization_window,
                'min_confidence': self.min_confidence,
                'max_risk': self.max_risk
            },
            'recent_proposals': [
                {
                    'name': p.name,
                    'target': p.target,
                    'priority': p.priority.value,
                    'confidence': p.confidence,
                    'proposed_at': p.proposed_at.isoformat()
                }
                for p in self.proposal_history[-5:]
            ]
        }

# Convenience function for direct usage
def propose_from_state() -> Optional[ModProposal]:
    """
    Analyze current consciousness state and propose a modification.
    
    Returns:
        ModProposal if a modification is recommended, None otherwise
    """
    advisor = ConsciousnessAdvisor()
    return advisor.propose_from_state()

def demo_consciousness_advisor():
    """Demonstrate the consciousness advisor functionality."""
    print("🎯 " + "="*70)
    print("🎯 DAWN CONSCIOUSNESS ADVISOR DEMONSTRATION")
    print("🎯 " + "="*70)
    print()
    
    # Initialize advisor
    advisor = ConsciousnessAdvisor()
    
    print(f"🎯 Advisor ID: {advisor.advisor_id}")
    print(f"⚙️  Configuration:")
    print(f"   • Low Momentum Threshold: {advisor.low_momentum_threshold}")
    print(f"   • Stagnation Threshold: {advisor.stagnation_threshold}")
    print(f"   • High Unity Threshold: {advisor.high_unity_threshold}")
    
    # Test different consciousness states
    test_scenarios = [
        {
            'name': 'Low Momentum Stagnation',
            'state': {'unity': 0.75, 'awareness': 0.70, 'momentum': 0.01, 'level': 'meta_aware'},
            'description': 'High unity but very low momentum - should trigger momentum boost'
        },
        {
            'name': 'Growth Potential',
            'state': {'unity': 0.82, 'awareness': 0.85, 'momentum': 0.04, 'level': 'meta_aware'},
            'description': 'Good state with growth potential - should suggest optimization'
        },
        {
            'name': 'Awareness Lag',
            'state': {'unity': 0.88, 'awareness': 0.75, 'momentum': 0.03, 'level': 'meta_aware'},
            'description': 'Unity ahead of awareness - should suggest awareness enhancement'
        },
        {
            'name': 'Stable State',
            'state': {'unity': 0.60, 'awareness': 0.58, 'momentum': 0.05, 'level': 'coherent'},
            'description': 'Balanced stable state - may not need modifications'
        }
    ]
    
    print(f"\n🧪 Testing Advisory Scenarios:")
    print("="*50)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['name']} ---")
        print(f"📋 Description: {scenario['description']}")
        
        # Set the test state
        from dawn_core.state import set_state
        set_state(**scenario['state'])
        
        # Show current state
        state = get_state()
        print(f"🧠 State: Unity={state.unity:.3f}, Awareness={state.awareness:.3f}, Momentum={state.momentum:.3f}, Level={state.level}")
        
        # Get advisor analysis
        analysis = advisor.analyze_current_state()
        indicators = analysis['indicators']
        
        print(f"🔍 Indicators: {[k for k, v in indicators.items() if v]}")
        
        # Get proposal
        proposal = advisor.propose_from_state()
        
        if proposal:
            print(f"💡 Proposal: {proposal.name}")
            print(f"   • Target: {proposal.target}")
            print(f"   • Type: {proposal.patch_type.value}")
            print(f"   • Change: {proposal.current_value} → {proposal.proposed_value}")
            print(f"   • Priority: {proposal.priority.value}")
            print(f"   • Confidence: {proposal.confidence:.3f}")
            print(f"   • Notes: {proposal.notes}")
            
            if proposal.reasoning_chain:
                print(f"   • Reasoning:")
                for reason in proposal.reasoning_chain:
                    print(f"     - {reason}")
        else:
            print("💡 No modification recommended for this state")
        
        print("-" * 30)
    
    # Show advisor status
    print(f"\n📊 Advisor Status:")
    status = advisor.get_advisor_status()
    print(f"   • States Analyzed: {status['state_history_length']}")
    print(f"   • Proposals Generated: {status['proposal_history_length']}")
    print(f"   • Recent Proposals: {len(status['recent_proposals'])}")
    
    if status['recent_proposals']:
        print(f"   • Latest Proposal: {status['recent_proposals'][-1]['name']}")
    
    print(f"\n🎯 Consciousness Advisor demonstration complete!")
    print("🎯 " + "="*70)

if __name__ == "__main__":
    demo_consciousness_advisor()
