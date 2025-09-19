"""
DAWN Schema Health Index (SHI) Calculator
=========================================
Advanced SHI calculation engine implementing the mathematical formulas
specified in DAWN documentation for comprehensive schema health monitoring.

Mathematical Foundation:
SHI = 1 - (α·E_s + β·V_e + γ·D_t + δ·(1-S_c) + φ·residue_term)

Where:
- E_s: Sigil entropy (symbolic thinking chaos)
- V_e: Edge volatility (schema boundary instability)  
- D_t: Tracer divergence (disagreement among tracers)
- S_c: Current SCUP value (semantic coherence under pressure)
- residue_term: Soot/ash balance impact

Author: DAWN Development Team
Generated: 2025-09-18
"""

import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading

# Import torch with fallback for systems without PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[SHICalculator] Warning: PyTorch not available, using numpy fallback")


class HealthStatus(Enum):
    """Overall schema health classifications"""
    VIBRANT = "vibrant"         # SHI > 0.8 - Optimal health
    HEALTHY = "healthy"         # SHI 0.6-0.8 - Good condition
    STABLE = "stable"          # SHI 0.4-0.6 - Acceptable
    DEGRADED = "degraded"      # SHI 0.2-0.4 - Needs attention
    CRITICAL = "critical"      # SHI < 0.2 - Immediate action required


class HealthComponent(Enum):
    """Components that contribute to schema health"""
    BLOOM_DYNAMICS = "bloom_dynamics"
    SIGIL_COHERENCE = "sigil_coherence"
    PULSE_STABILITY = "pulse_stability"
    NUTRIENT_FLOW = "nutrient_flow"
    CONSCIOUSNESS_INTEGRITY = "consciousness_integrity"
    TEMPORAL_ALIGNMENT = "temporal_alignment"


@dataclass
class HealthMetrics:
    """Detailed health metrics for schema analysis"""
    shi_value: float = 0.5
    status: HealthStatus = HealthStatus.STABLE
    bloom_health: float = 0.5
    sigil_health: float = 0.5
    pulse_health: float = 0.5
    nutrient_health: float = 0.5
    consciousness_health: float = 0.5
    temporal_health: float = 0.5
    
    # Component contributions
    sigil_entropy_contribution: float = 0.0
    edge_volatility_contribution: float = 0.0
    tracer_divergence_contribution: float = 0.0
    scup_coherence_contribution: float = 0.0
    residue_balance_contribution: float = 0.0
    
    def get_weakest_component(self) -> Tuple[str, float]:
        """Identify the weakest health component"""
        components = {
            "bloom_health": self.bloom_health,
            "sigil_health": self.sigil_health,
            "pulse_health": self.pulse_health,
            "nutrient_health": self.nutrient_health,
            "consciousness_health": self.consciousness_health,
            "temporal_health": self.temporal_health
        }
        return min(components.items(), key=lambda x: x[1])
    
    def get_strongest_component(self) -> Tuple[str, float]:
        """Identify the strongest health component"""
        components = {
            "bloom_health": self.bloom_health,
            "sigil_health": self.sigil_health,
            "pulse_health": self.pulse_health,
            "nutrient_health": self.nutrient_health,
            "consciousness_health": self.consciousness_health,
            "temporal_health": self.temporal_health
        }
        return max(components.items(), key=lambda x: x[1])


@dataclass
class SHICalculationResult:
    """Complete SHI calculation results"""
    shi_value: float
    status: HealthStatus
    metrics: HealthMetrics
    components: Dict[str, float]
    recommendations: List[str]
    calculation_time: float
    method_used: str


class SHICalculator:
    """
    Comprehensive Schema Health Index calculator implementing
    DAWN's mathematical formula for schema health monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SHI calculator with configurable weights
        
        Args:
            config: Optional configuration for weights and thresholds
        """
        self.config = config or {}
        
        # Core SHI formula weights (α, β, γ, δ, φ)
        self.formula_weights = {
            'sigil_entropy': self.config.get('alpha', 0.25),      # α
            'edge_volatility': self.config.get('beta', 0.20),    # β  
            'tracer_divergence': self.config.get('gamma', 0.15), # γ
            'scup_coherence': self.config.get('delta', 0.25),    # δ
            'residue_balance': self.config.get('phi', 0.15)      # φ
        }
        
        # Health component weights for detailed analysis
        self.component_weights = {
            HealthComponent.BLOOM_DYNAMICS: 0.20,
            HealthComponent.SIGIL_COHERENCE: 0.25,
            HealthComponent.PULSE_STABILITY: 0.15,
            HealthComponent.NUTRIENT_FLOW: 0.15,
            HealthComponent.CONSCIOUSNESS_INTEGRITY: 0.15,
            HealthComponent.TEMPORAL_ALIGNMENT: 0.10
        }
        
        # Status thresholds
        self.status_thresholds = {
            'critical': self.config.get('critical_threshold', 0.2),
            'degraded': self.config.get('degraded_threshold', 0.4),
            'stable': self.config.get('stable_threshold', 0.6),
            'healthy': self.config.get('healthy_threshold', 0.8)
        }
        
        # History tracking
        self.calculation_history = deque(maxlen=1000)
        self.metrics_history = deque(maxlen=1000)
        self.anomaly_history = deque(maxlen=100)
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0
        self.error_count = 0
        
        # Recovery system
        self.recovery_active = False
        self.recovery_start_time = 0.0
        self.recovery_boost_factor = 0.1
        
        # Thread safety
        self.lock = threading.RLock()
        
        print("[SHICalculator] 📊 SHI calculation engine initialized")
        
    def calculate_shi(self, 
                     sigil_entropy: float,
                     edge_volatility: float,
                     tracer_divergence: float,
                     scup_value: float,
                     residue_balance: float,
                     additional_metrics: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate Schema Health Index using the core mathematical formula
        
        Args:
            sigil_entropy: Symbolic thinking chaos level (0-1)
            edge_volatility: Schema boundary instability (0-1)
            tracer_divergence: Disagreement among tracers (0-1)
            scup_value: Current SCUP coherence value (0-1)
            residue_balance: Soot/ash balance impact (0-1)
            additional_metrics: Optional additional metrics
            
        Returns:
            float: SHI value in range [0.0, 1.0]
        """
        start_time = time.time()
        
        try:
            with self.lock:
                # Normalize inputs to [0,1] range
                E_s = self._normalize_value(sigil_entropy, "entropy")
                V_e = self._normalize_value(edge_volatility, "volatility")
                D_t = self._normalize_value(tracer_divergence, "divergence")
                S_c = max(0.0, min(1.0, scup_value))
                residue_term = self._calculate_residue_penalty(residue_balance)
                
                # Apply core SHI formula: SHI = 1 - (α·E_s + β·V_e + γ·D_t + δ·(1-S_c) + φ·residue_term)
                penalty = (
                    self.formula_weights['sigil_entropy'] * E_s +
                    self.formula_weights['edge_volatility'] * V_e +
                    self.formula_weights['tracer_divergence'] * D_t +
                    self.formula_weights['scup_coherence'] * (1 - S_c) +
                    self.formula_weights['residue_balance'] * residue_term
                )
                
                # Calculate base SHI
                shi = 1.0 - penalty
                
                # Apply recovery boost if active
                if self.recovery_active:
                    recovery_duration = time.time() - self.recovery_start_time
                    recovery_boost = min(self.recovery_boost_factor, recovery_duration / 300)  # Max boost over 5 minutes
                    shi = shi + recovery_boost
                
                # Ensure bounds [0, 1]
                shi = max(0.0, min(1.0, shi))
                
                # Update tracking
                calculation_time = time.time() - start_time
                self._update_tracking(shi, calculation_time, {
                    'sigil_entropy': E_s,
                    'edge_volatility': V_e,
                    'tracer_divergence': D_t,
                    'scup_coherence': S_c,
                    'residue_balance': residue_term
                })
                
                # Check for critical conditions
                if shi < self.status_thresholds['critical']:
                    self._trigger_critical_intervention(shi)
                
                return round(shi, 4)
                
        except Exception as e:
            self.error_count += 1
            print(f"[SHICalculator] ❌ Error in calculate_shi: {str(e)}")
            return 0.0  # Emergency fallback
    
    def calculate_detailed_shi(self, 
                              sigil_entropy: float,
                              edge_volatility: float,
                              tracer_divergence: float,
                              scup_value: float,
                              residue_balance: float,
                              additional_metrics: Optional[Dict[str, Any]] = None) -> SHICalculationResult:
        """
        Calculate detailed SHI with comprehensive analysis
        
        Returns:
            SHICalculationResult: Complete calculation results with detailed breakdown
        """
        start_time = time.time()
        
        # Calculate base SHI
        shi = self.calculate_shi(
            sigil_entropy, edge_volatility, tracer_divergence, 
            scup_value, residue_balance, additional_metrics
        )
        
        # Classify health status
        status = self._classify_health_status(shi)
        
        # Calculate detailed metrics
        metrics = self._calculate_detailed_metrics(
            sigil_entropy, edge_volatility, tracer_divergence, 
            scup_value, residue_balance, additional_metrics
        )
        
        # Get component breakdown
        components = self._get_component_breakdown(
            sigil_entropy, edge_volatility, tracer_divergence, 
            scup_value, residue_balance
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(shi, metrics, components)
        
        calculation_time = time.time() - start_time
        
        return SHICalculationResult(
            shi_value=shi,
            status=status,
            metrics=metrics,
            components=components,
            recommendations=recommendations,
            calculation_time=calculation_time,
            method_used="detailed"
        )
    
    def _normalize_value(self, value: float, value_type: str) -> float:
        """Normalize input values to [0,1] range with type-specific handling"""
        # Clamp to reasonable bounds first
        value = max(0.0, min(10.0, value))
        
        if value_type == "entropy":
            # Entropy normalization - higher values indicate more chaos
            return min(1.0, value / 2.0)  # Assume max meaningful entropy is 2.0
        elif value_type == "volatility":
            # Volatility normalization - direct mapping
            return min(1.0, value)
        elif value_type == "divergence":
            # Divergence normalization - already calculated as normalized score
            return min(1.0, value)
        else:
            # Default normalization
            return min(1.0, value)
    
    def _calculate_residue_penalty(self, residue_balance: float) -> float:
        """
        Calculate residue penalty term for SHI formula
        
        Args:
            residue_balance: Residue health impact score
            
        Returns:
            float: Penalty term for residue imbalance
        """
        # residue_balance is expected to be a health impact score
        # Higher values = better health, so penalty is inverse
        penalty = 1.0 - max(0.0, min(1.0, residue_balance))
        
        # Apply sigmoid smoothing to avoid sharp transitions
        penalty = 1.0 / (1.0 + math.exp(-5.0 * (penalty - 0.5)))
        
        return penalty
    
    def _classify_health_status(self, shi: float) -> HealthStatus:
        """Classify SHI value into health status category"""
        if shi > self.status_thresholds['healthy']:
            return HealthStatus.VIBRANT
        elif shi > self.status_thresholds['stable']:
            return HealthStatus.HEALTHY
        elif shi > self.status_thresholds['degraded']:
            return HealthStatus.STABLE
        elif shi > self.status_thresholds['critical']:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.CRITICAL
    
    def _calculate_detailed_metrics(self, 
                                   sigil_entropy: float,
                                   edge_volatility: float,
                                   tracer_divergence: float,
                                   scup_value: float,
                                   residue_balance: float,
                                   additional_metrics: Optional[Dict[str, Any]]) -> HealthMetrics:
        """Calculate detailed health metrics for all components"""
        
        # Calculate individual component healths
        bloom_health = self._calculate_bloom_health(additional_metrics)
        sigil_health = 1.0 - self._normalize_value(sigil_entropy, "entropy")
        pulse_health = self._calculate_pulse_health(additional_metrics)
        nutrient_health = self._calculate_nutrient_health(additional_metrics)
        consciousness_health = scup_value  # SCUP represents consciousness coherence
        temporal_health = self._calculate_temporal_health(additional_metrics)
        
        # Calculate component contributions to SHI
        sigil_contribution = self.formula_weights['sigil_entropy'] * self._normalize_value(sigil_entropy, "entropy")
        edge_contribution = self.formula_weights['edge_volatility'] * edge_volatility
        tracer_contribution = self.formula_weights['tracer_divergence'] * tracer_divergence
        scup_contribution = self.formula_weights['scup_coherence'] * (1 - scup_value)
        residue_contribution = self.formula_weights['residue_balance'] * self._calculate_residue_penalty(residue_balance)
        
        return HealthMetrics(
            bloom_health=bloom_health,
            sigil_health=sigil_health,
            pulse_health=pulse_health,
            nutrient_health=nutrient_health,
            consciousness_health=consciousness_health,
            temporal_health=temporal_health,
            sigil_entropy_contribution=sigil_contribution,
            edge_volatility_contribution=edge_contribution,
            tracer_divergence_contribution=tracer_contribution,
            scup_coherence_contribution=scup_contribution,
            residue_balance_contribution=residue_contribution
        )
    
    def _calculate_bloom_health(self, additional_metrics: Optional[Dict[str, Any]]) -> float:
        """Calculate bloom dynamics health"""
        if not additional_metrics or 'bloom_metrics' not in additional_metrics:
            return 0.5  # Default neutral health
        
        bloom_data = additional_metrics['bloom_metrics']
        
        # Calculate based on bloom success rates and stability
        active_ratio = bloom_data.get('active_blooms', 0) / max(1, bloom_data.get('total_blooms', 1))
        failure_rate = bloom_data.get('failure_rate', 0.0)
        
        health = active_ratio * (1.0 - failure_rate)
        return max(0.0, min(1.0, health))
    
    def _calculate_pulse_health(self, additional_metrics: Optional[Dict[str, Any]]) -> float:
        """Calculate pulse stability health"""
        if not additional_metrics or 'pulse_metrics' not in additional_metrics:
            return 0.5  # Default neutral health
        
        pulse_data = additional_metrics['pulse_metrics']
        
        # Calculate based on pulse regularity and amplitude
        stability = 1.0 - pulse_data.get('variance', 0.5)
        amplitude_health = 1.0 - abs(pulse_data.get('amplitude', 0.5) - 0.5) * 2
        
        health = (stability + amplitude_health) / 2.0
        return max(0.0, min(1.0, health))
    
    def _calculate_nutrient_health(self, additional_metrics: Optional[Dict[str, Any]]) -> float:
        """Calculate nutrient flow health"""
        if not additional_metrics or 'nutrient_metrics' not in additional_metrics:
            return 0.5  # Default neutral health
        
        nutrient_data = additional_metrics['nutrient_metrics']
        
        # Calculate based on flow rates and distribution
        flow_rate = nutrient_data.get('flow_rate', 0.5)
        distribution_balance = 1.0 - nutrient_data.get('distribution_variance', 0.5)
        
        health = (flow_rate + distribution_balance) / 2.0
        return max(0.0, min(1.0, health))
    
    def _calculate_temporal_health(self, additional_metrics: Optional[Dict[str, Any]]) -> float:
        """Calculate temporal alignment health"""
        if not additional_metrics or 'temporal_metrics' not in additional_metrics:
            return 0.5  # Default neutral health
        
        temporal_data = additional_metrics['temporal_metrics']
        
        # Calculate based on time synchronization and drift
        sync_quality = temporal_data.get('sync_quality', 0.5)
        drift_penalty = temporal_data.get('temporal_drift', 0.0)
        
        health = sync_quality * (1.0 - drift_penalty)
        return max(0.0, min(1.0, health))
    
    def _get_component_breakdown(self, 
                                sigil_entropy: float,
                                edge_volatility: float,
                                tracer_divergence: float,
                                scup_value: float,
                                residue_balance: float) -> Dict[str, float]:
        """Get detailed breakdown of SHI component contributions"""
        
        E_s = self._normalize_value(sigil_entropy, "entropy")
        V_e = self._normalize_value(edge_volatility, "volatility")
        D_t = self._normalize_value(tracer_divergence, "divergence")
        residue_term = self._calculate_residue_penalty(residue_balance)
        
        return {
            'sigil_entropy_penalty': self.formula_weights['sigil_entropy'] * E_s,
            'edge_volatility_penalty': self.formula_weights['edge_volatility'] * V_e,
            'tracer_divergence_penalty': self.formula_weights['tracer_divergence'] * D_t,
            'scup_coherence_penalty': self.formula_weights['scup_coherence'] * (1 - scup_value),
            'residue_balance_penalty': self.formula_weights['residue_balance'] * residue_term,
            'total_penalty': (
                self.formula_weights['sigil_entropy'] * E_s +
                self.formula_weights['edge_volatility'] * V_e +
                self.formula_weights['tracer_divergence'] * D_t +
                self.formula_weights['scup_coherence'] * (1 - scup_value) +
                self.formula_weights['residue_balance'] * residue_term
            )
        }
    
    def _generate_recommendations(self, 
                                 shi: float, 
                                 metrics: HealthMetrics, 
                                 components: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on SHI analysis"""
        recommendations = []
        
        # Critical SHI recommendations
        if shi < self.status_thresholds['critical']:
            recommendations.append("🚨 CRITICAL: Immediate intervention required - SHI below critical threshold")
            recommendations.append("🔧 Activate emergency recovery protocols")
            recommendations.append("🛡️ Enable schema stabilization mechanisms")
        
        # Component-specific recommendations
        weakest_component, weakest_value = metrics.get_weakest_component()
        if weakest_value < 0.3:
            recommendations.append(f"⚠️ Address weak {weakest_component}: {weakest_value:.3f}")
        
        # Penalty-specific recommendations
        if components['sigil_entropy_penalty'] > 0.15:
            recommendations.append("🔤 High sigil entropy - consider symbolic stabilization")
        
        if components['edge_volatility_penalty'] > 0.12:
            recommendations.append("🕸️ High edge volatility - review schema boundaries")
        
        if components['tracer_divergence_penalty'] > 0.1:
            recommendations.append("🐛 High tracer divergence - synchronize tracer systems")
        
        if components['scup_coherence_penalty'] > 0.15:
            recommendations.append("🧠 Low SCUP coherence - reduce cognitive pressure")
        
        if components['residue_balance_penalty'] > 0.1:
            recommendations.append("♻️ Poor residue balance - optimize soot/ash conversion")
        
        # Recovery recommendations
        if self.recovery_active:
            recovery_duration = time.time() - self.recovery_start_time
            recommendations.append(f"🔄 Recovery active for {recovery_duration:.1f}s")
        
        return recommendations
    
    def _update_tracking(self, shi: float, calculation_time: float, components: Dict[str, float]):
        """Update calculation tracking and performance metrics"""
        self.calculation_count += 1
        self.total_calculation_time += calculation_time
        
        # Add to history
        self.calculation_history.append({
            'timestamp': time.time(),
            'shi': shi,
            'calculation_time': calculation_time,
            'components': components.copy()
        })
        
        # Check for anomalies
        if len(self.calculation_history) > 10:
            recent_values = [calc['shi'] for calc in list(self.calculation_history)[-10:]]
            variance = np.var(recent_values)
            if variance > 0.1:  # High variance threshold
                self.anomaly_history.append({
                    'timestamp': time.time(),
                    'type': 'high_variance',
                    'variance': variance,
                    'values': recent_values
                })
    
    def _trigger_critical_intervention(self, shi: float):
        """Trigger critical intervention protocols"""
        if not self.recovery_active:
            self.recovery_active = True
            self.recovery_start_time = time.time()
            print(f"[SHICalculator] 🚨 Critical SHI detected ({shi:.3f}) - activating recovery protocols")
    
    def activate_recovery_mode(self, boost_factor: float = 0.1):
        """Manually activate recovery mode"""
        with self.lock:
            self.recovery_active = True
            self.recovery_start_time = time.time()
            self.recovery_boost_factor = boost_factor
            print(f"[SHICalculator] 🔄 Recovery mode activated with boost factor {boost_factor}")
    
    def deactivate_recovery_mode(self):
        """Deactivate recovery mode"""
        with self.lock:
            self.recovery_active = False
            print("[SHICalculator] ✅ Recovery mode deactivated")
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """Get calculation performance statistics"""
        with self.lock:
            avg_time = self.total_calculation_time / max(1, self.calculation_count)
            
            return {
                'calculation_count': self.calculation_count,
                'total_time': self.total_calculation_time,
                'average_time': avg_time,
                'error_count': self.error_count,
                'history_size': len(self.calculation_history),
                'anomaly_count': len(self.anomaly_history),
                'recovery_active': self.recovery_active
            }
    
    def get_health_trend(self, window: int = 50) -> Dict[str, float]:
        """Analyze health trend over recent calculations"""
        with self.lock:
            if len(self.calculation_history) < 2:
                return {'trend': 0.0, 'stability': 1.0, 'direction': 'stable'}
            
            recent_calculations = list(self.calculation_history)[-window:]
            shi_values = [calc['shi'] for calc in recent_calculations]
            
            if len(shi_values) >= 2:
                trend = shi_values[-1] - shi_values[0]
                stability = 1.0 - np.std(shi_values)
                
                if trend > 0.05:
                    direction = 'improving'
                elif trend < -0.05:
                    direction = 'degrading'
                else:
                    direction = 'stable'
            else:
                trend = 0.0
                stability = 1.0
                direction = 'stable'
            
            return {
                'trend': trend,
                'stability': max(0.0, stability),
                'direction': direction,
                'current_shi': shi_values[-1] if shi_values else 0.5,
                'window_size': len(shi_values)
            }


# Factory function for easy instantiation
def create_shi_calculator(config: Optional[Dict[str, Any]] = None) -> SHICalculator:
    """
    Create a configured SHI calculator instance
    
    Args:
        config: Optional configuration for weights and thresholds
        
    Returns:
        SHICalculator: Configured SHI calculator instance
    """
    return SHICalculator(config)


if __name__ == "__main__":
    # Example usage and testing
    print("DAWN SHI Calculator - Example Usage")
    
    # Create calculator
    calculator = create_shi_calculator()
    
    # Test basic calculation
    shi = calculator.calculate_shi(
        sigil_entropy=0.3,
        edge_volatility=0.2,
        tracer_divergence=0.1,
        scup_value=0.7,
        residue_balance=0.8
    )
    
    print(f"Basic SHI: {shi}")
    
    # Test detailed calculation
    result = calculator.calculate_detailed_shi(
        sigil_entropy=0.4,
        edge_volatility=0.3,
        tracer_divergence=0.2,
        scup_value=0.6,
        residue_balance=0.7,
        additional_metrics={
            'bloom_metrics': {'active_blooms': 5, 'total_blooms': 8, 'failure_rate': 0.1},
            'pulse_metrics': {'variance': 0.2, 'amplitude': 0.6}
        }
    )
    
    print(f"Detailed SHI: {result.shi_value}")
    print(f"Status: {result.status.value}")
    print(f"Recommendations: {result.recommendations}")
    
    # Test trend analysis
    for i in range(20):
        # Simulate varying conditions
        entropy = 0.2 + 0.1 * np.sin(i * 0.3)
        volatility = 0.15 + 0.05 * np.cos(i * 0.2)
        
        shi = calculator.calculate_shi(
            sigil_entropy=entropy,
            edge_volatility=volatility,
            tracer_divergence=0.1,
            scup_value=0.7,
            residue_balance=0.8
        )
    
    trend = calculator.get_health_trend()
    print(f"Health Trend: {trend}")
    
    stats = calculator.get_calculation_stats()
    print(f"Performance Stats: {stats}")
