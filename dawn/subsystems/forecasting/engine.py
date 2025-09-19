#!/usr/bin/env python3
"""
🔮 Forecasting Engine - DAWN Anticipatory Vector Layer
═══════════════════════════════════════════════════════════

Implements predictive cognitive state modeling for DAWN.
Projects pressure, entropy, residue balance, and tracer dynamics
across near, mid, and long-term horizons.

Based on RTF specifications from DAWN-docs/Forcasting/
"""

import logging
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import math
from collections import deque

logger = logging.getLogger(__name__)

class ForecastHorizon(Enum):
    """Forecast time horizons"""
    SHORT_TERM = "short_term"    # 1-10 ticks
    MID_TERM = "mid_term"        # 10-100 ticks  
    LONG_TERM = "long_term"      # 100-1000 ticks

class StabilityZone(Enum):
    """System stability zones based on F ratio"""
    STABLE = "stable"        # F < 0.5
    WATCH = "watch"          # 0.5 <= F < 1.0
    ACT = "act"             # F >= 1.0

class InterventionType(Enum):
    """Types of interventions the forecasting engine can recommend"""
    WHALE_SPAWN = "whale_spawn"          # Context stabilization
    CROW_CAP = "crow_cap"                # Limit analysis depth
    PURIFICATION = "purification"        # Entropy reduction
    BEETLE_RECYCLE = "beetle_recycle"    # Resource recovery
    MIRROR_AUDIT = "mirror_audit"        # Schema health check
    WEAVING = "weaving"                  # Schema reinforcement

@dataclass
class AdaptiveCapacity:
    """Components of adaptive capacity A"""
    nutrient_reserves: float = 0.0      # N: mycelial + ash yield
    tracer_slack: float = 0.0           # S_T: idle tracer capacity
    schema_margin: float = 0.0          # M_SHI: schema health headroom
    
    # Weights (must sum to 1.0)
    w_nutrient: float = 0.4
    w_tracer: float = 0.3
    w_schema: float = 0.3
    
    def __post_init__(self):
        # Normalize weights
        total_weight = self.w_nutrient + self.w_tracer + self.w_schema
        if total_weight > 0:
            self.w_nutrient /= total_weight
            self.w_tracer /= total_weight
            self.w_schema /= total_weight
    
    def calculate_total(self) -> float:
        """Calculate total adaptive capacity"""
        return (self.w_nutrient * self.nutrient_reserves + 
                self.w_tracer * self.tracer_slack + 
                self.w_schema * self.schema_margin)

@dataclass
class CognitivePressure:
    """Cognitive pressure components"""
    base_bandwidth: float = 0.0         # B: current processing span
    variance: float = 0.0               # σ²: signal variance
    
    def calculate_pressure(self) -> float:
        """Calculate cognitive pressure P = B * σ²"""
        return self.base_bandwidth * self.variance

@dataclass
class ForecastResult:
    """Result of forecasting calculation"""
    horizon: ForecastHorizon
    F_value: float                      # Raw forecasting function value
    F_normalized: float                 # Normalized and clamped value
    F_smoothed: float                   # Smoothed value for decisions
    confidence: float                   # Confidence in prediction [0,1]
    stability_zone: StabilityZone
    recommended_interventions: List[InterventionType] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemInputs:
    """Input data for forecasting calculations"""
    tick: int
    
    # Pressure components
    active_node_count: int = 0
    average_node_health: float = 0.0
    nutrient_throughput: float = 0.0
    tracer_concurrency: float = 0.0
    signal_variance: float = 0.0
    
    # Adaptive capacity components
    nutrient_budget: float = 0.0
    ash_yield: float = 0.0
    idle_tracer_count: int = 0
    retirable_tracer_capacity: float = 0.0
    current_SHI: float = 0.0
    safe_SHI_threshold: float = 0.8
    
    # Additional context
    entropy_level: float = 0.0
    residue_ash_ratio: float = 0.0
    residue_soot_ratio: float = 0.0
    shimmer_decay_rate: float = 0.0
    drift_vectors: List[float] = field(default_factory=list)

class ForecastingEngine:
    """
    DAWN's Anticipatory Vector Layer - projects cognitive state forward in time.
    Enables proactive preparation rather than reactive responses.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Forecasting parameters
        self.F_scale = self.config.get('F_scale', 2.0)
        self.F_cap = self.config.get('F_cap', 5.0)
        self.epsilon = self.config.get('epsilon', 1e-4)  # Prevent divide by zero
        self.smoothing_alpha = self.config.get('smoothing_alpha', 0.4)
        
        # Horizon parameters
        self.short_term_ticks = self.config.get('short_term_ticks', 10)
        self.mid_term_ticks = self.config.get('mid_term_ticks', 100) 
        self.long_term_ticks = self.config.get('long_term_ticks', 1000)
        
        # History tracking
        self.forecast_history: List[Dict[str, ForecastResult]] = []
        self.smoothed_F_history: Dict[ForecastHorizon, deque] = {
            horizon: deque(maxlen=100) for horizon in ForecastHorizon
        }
        
        # Performance tracking
        self.prediction_errors: Dict[str, deque] = {
            'pressure_mae': deque(maxlen=1000),
            'drift_error': deque(maxlen=1000), 
            'entropy_error': deque(maxlen=1000)
        }
        
        # SCUP coupling
        self.scup_early_warning_threshold = self.config.get('scup_threshold', 0.8)
        self.coherence_loss_probability_cache = {}
        
        logger.info("🔮 Forecasting Engine initialized")
    
    def generate_forecast(self, inputs: SystemInputs) -> Dict[ForecastHorizon, ForecastResult]:
        """
        Generate comprehensive forecast across all time horizons.
        
        Args:
            inputs: Current system state inputs
            
        Returns:
            Dictionary of forecast results by horizon
        """
        logger.info(f"🔮 Generating forecast for tick {inputs.tick}")
        
        # Calculate base cognitive pressure and adaptive capacity
        pressure = self._calculate_cognitive_pressure(inputs)
        capacity = self._calculate_adaptive_capacity(inputs)
        
        # Generate forecasts for each horizon
        forecasts = {}
        
        for horizon in ForecastHorizon:
            forecast = self._generate_horizon_forecast(
                horizon, pressure, capacity, inputs
            )
            forecasts[horizon] = forecast
        
        # Apply smoothing and generate interventions
        self._apply_smoothing(forecasts)
        self._generate_interventions(forecasts, inputs)
        
        # Store in history
        self.forecast_history.append(forecasts)
        if len(self.forecast_history) > 1000:
            self.forecast_history = self.forecast_history[-1000:]
        
        # Update SCUP coupling
        self._update_scup_coupling(forecasts, inputs)
        
        logger.info(f"🔮 Forecast complete - Zones: "
                   f"ST:{forecasts[ForecastHorizon.SHORT_TERM].stability_zone.value}, "
                   f"MT:{forecasts[ForecastHorizon.MID_TERM].stability_zone.value}, "
                   f"LT:{forecasts[ForecastHorizon.LONG_TERM].stability_zone.value}")
        
        return forecasts
    
    def _calculate_cognitive_pressure(self, inputs: SystemInputs) -> CognitivePressure:
        """Calculate cognitive pressure P = B * σ²"""
        # Calculate base bandwidth B
        base_bandwidth = (
            inputs.active_node_count * inputs.average_node_health * 
            inputs.nutrient_throughput * inputs.tracer_concurrency
        )
        
        # Ensure minimum bandwidth
        base_bandwidth = max(base_bandwidth, self.epsilon)
        
        # Use signal variance as σ²
        variance = max(inputs.signal_variance, 0.0)
        
        pressure = CognitivePressure(
            base_bandwidth=base_bandwidth,
            variance=variance
        )
        
        logger.debug(f"🔮 Calculated pressure: B={base_bandwidth:.3f}, σ²={variance:.3f}, "
                    f"P={pressure.calculate_pressure():.3f}")
        
        return pressure
    
    def _calculate_adaptive_capacity(self, inputs: SystemInputs) -> AdaptiveCapacity:
        """Calculate adaptive capacity A with weighted components"""
        # Nutrient reserves N
        nutrient_reserves = inputs.nutrient_budget + inputs.ash_yield
        
        # Tracer slack S_T
        tracer_slack = inputs.idle_tracer_count * inputs.retirable_tracer_capacity
        
        # Schema margin M_SHI
        schema_margin = max(0.0, inputs.current_SHI - inputs.safe_SHI_threshold)
        
        capacity = AdaptiveCapacity(
            nutrient_reserves=nutrient_reserves,
            tracer_slack=tracer_slack,
            schema_margin=schema_margin
        )
        
        total_capacity = capacity.calculate_total()
        
        logger.debug(f"🔮 Calculated capacity: N={nutrient_reserves:.3f}, "
                    f"S_T={tracer_slack:.3f}, M_SHI={schema_margin:.3f}, "
                    f"A={total_capacity:.3f}")
        
        return capacity
    
    def _generate_horizon_forecast(
        self, 
        horizon: ForecastHorizon, 
        pressure: CognitivePressure, 
        capacity: AdaptiveCapacity,
        inputs: SystemInputs
    ) -> ForecastResult:
        """Generate forecast for specific time horizon"""
        
        # Calculate horizon-specific adjustments
        if horizon == ForecastHorizon.SHORT_TERM:
            # Linear extrapolation (Euler step)
            P_projected = pressure.calculate_pressure()
            A_projected = capacity.calculate_total()
            confidence = 0.9  # High confidence for short term
            
        elif horizon == ForecastHorizon.MID_TERM:
            # EMA over pressure/entropy curves with shimmer decay
            P_projected = pressure.calculate_pressure() * (1 + inputs.shimmer_decay_rate)
            A_projected = capacity.calculate_total() * (1 - inputs.entropy_level * 0.1)
            confidence = 0.7  # Medium confidence
            
        else:  # LONG_TERM
            # Monte Carlo simulation factors
            residue_factor = 1.0 + (inputs.residue_soot_ratio - inputs.residue_ash_ratio) * 0.2
            drift_factor = 1.0 + (sum(inputs.drift_vectors) / max(len(inputs.drift_vectors), 1)) * 0.15
            
            P_projected = pressure.calculate_pressure() * residue_factor * drift_factor
            A_projected = capacity.calculate_total() * (1 - inputs.entropy_level * 0.3)
            confidence = 0.5  # Lower confidence for long term
        
        # Prevent divide by zero
        A_projected = max(A_projected, self.epsilon)
        
        # Calculate forecasting function F = P/A
        F_raw = P_projected / A_projected
        
        # Normalize and clamp
        F_normalized = min(F_raw / self.F_scale, self.F_cap)
        
        # Determine stability zone
        if F_normalized < 0.5:
            zone = StabilityZone.STABLE
        elif F_normalized < 1.0:
            zone = StabilityZone.WATCH
        else:
            zone = StabilityZone.ACT
        
        forecast = ForecastResult(
            horizon=horizon,
            F_value=F_raw,
            F_normalized=F_normalized,
            F_smoothed=F_normalized,  # Will be updated in smoothing step
            confidence=confidence,
            stability_zone=zone,
            metadata={
                'P_projected': P_projected,
                'A_projected': A_projected,
                'horizon_ticks': self._get_horizon_ticks(horizon)
            }
        )
        
        return forecast
    
    def _apply_smoothing(self, forecasts: Dict[ForecastHorizon, ForecastResult]) -> None:
        """Apply temporal smoothing to avoid decision thrashing"""
        for horizon, forecast in forecasts.items():
            history = self.smoothed_F_history[horizon]
            
            if history:
                # Apply exponential moving average smoothing
                previous_smoothed = history[-1]
                smoothed = (self.smoothing_alpha * forecast.F_normalized + 
                           (1 - self.smoothing_alpha) * previous_smoothed)
            else:
                smoothed = forecast.F_normalized
            
            forecast.F_smoothed = smoothed
            history.append(smoothed)
            
            # Update stability zone based on smoothed value
            if smoothed < 0.5:
                forecast.stability_zone = StabilityZone.STABLE
            elif smoothed < 1.0:
                forecast.stability_zone = StabilityZone.WATCH
            else:
                forecast.stability_zone = StabilityZone.ACT
    
    def _generate_interventions(
        self, 
        forecasts: Dict[ForecastHorizon, ForecastResult],
        inputs: SystemInputs
    ) -> None:
        """Generate intervention recommendations based on forecasts"""
        
        for horizon, forecast in forecasts.items():
            interventions = []
            
            # Check for rapid F increase
            history = self.smoothed_F_history[horizon]
            if len(history) >= 2:
                F_delta = history[-1] - history[-2]
                if F_delta > 0.2:  # Rapid increase
                    interventions.extend([
                        InterventionType.WHALE_SPAWN,
                        InterventionType.CROW_CAP,
                        InterventionType.PURIFICATION
                    ])
            
            # Check adaptive capacity limitations
            if forecast.metadata.get('A_projected', 0) < 0.3:
                if inputs.current_SHI < inputs.safe_SHI_threshold + 0.1:
                    interventions.extend([
                        InterventionType.MIRROR_AUDIT,
                        InterventionType.WEAVING
                    ])
                
                if inputs.nutrient_budget < 0.2:
                    interventions.append(InterventionType.BEETLE_RECYCLE)
            
            # Zone-specific interventions
            if forecast.stability_zone == StabilityZone.ACT:
                if InterventionType.PURIFICATION not in interventions:
                    interventions.append(InterventionType.PURIFICATION)
                
                if inputs.entropy_level > 0.7:
                    interventions.append(InterventionType.WHALE_SPAWN)
            
            forecast.recommended_interventions = interventions
    
    def _update_scup_coupling(
        self, 
        forecasts: Dict[ForecastHorizon, ForecastResult],
        inputs: SystemInputs
    ) -> None:
        """Update SCUP (Semantic Coherence Under Pressure) coupling"""
        
        # Calculate coherence loss probability for each horizon
        for horizon, forecast in forecasts.items():
            # Simple model: probability increases with F and decreases with confidence
            base_prob = min(0.95, forecast.F_smoothed / 2.0)
            confidence_adjusted = base_prob * (1.0 - forecast.confidence * 0.3)
            
            # Factor in current system stress
            stress_factor = 1.0 + (1.0 - inputs.current_SHI) * 0.5
            final_prob = min(0.99, confidence_adjusted * stress_factor)
            
            self.coherence_loss_probability_cache[horizon] = final_prob
            
            # Generate SCUP early warning if threshold exceeded
            if final_prob > self.scup_early_warning_threshold:
                forecast.metadata['scup_early_warning'] = True
                forecast.metadata['coherence_loss_probability'] = final_prob
                
                logger.warning(f"🔮 SCUP Early Warning: {horizon.value} horizon "
                              f"coherence loss probability: {final_prob:.2f}")
    
    def get_scup_early_warning_index(self) -> Dict[str, Any]:
        """Get SCUP Early Warning Index for Owl audits"""
        return {
            'timestamp': datetime.now().isoformat(),
            'coherence_loss_probabilities': {
                horizon.value: prob for horizon, prob 
                in self.coherence_loss_probability_cache.items()
            },
            'early_warning_active': any(
                prob > self.scup_early_warning_threshold 
                for prob in self.coherence_loss_probability_cache.values()
            ),
            'recommended_owl_audit': any(
                prob > 0.9 for prob in self.coherence_loss_probability_cache.values()
            )
        }
    
    def record_prediction_error(self, error_type: str, error_value: float) -> None:
        """Record prediction error for backtesting"""
        if error_type in self.prediction_errors:
            self.prediction_errors[error_type].append(error_value)
    
    def get_backtesting_metrics(self) -> Dict[str, Any]:
        """Get backtesting and error metrics"""
        metrics = {}
        
        for error_type, errors in self.prediction_errors.items():
            if errors:
                metrics[error_type] = {
                    'mean_absolute_error': np.mean(np.abs(list(errors))),
                    'rmse': np.sqrt(np.mean(np.square(list(errors)))),
                    'count': len(errors),
                    'recent_error': errors[-1] if errors else 0.0
                }
        
        return {
            'error_metrics': metrics,
            'total_forecasts': len(self.forecast_history),
            'smoothing_alpha': self.smoothing_alpha,
            'confidence_trends': self._calculate_confidence_trends()
        }
    
    def _calculate_confidence_trends(self) -> Dict[str, float]:
        """Calculate confidence trends across horizons"""
        if len(self.forecast_history) < 10:
            return {}
        
        trends = {}
        recent_forecasts = self.forecast_history[-10:]
        
        for horizon in ForecastHorizon:
            confidences = [f[horizon].confidence for f in recent_forecasts]
            if len(confidences) >= 2:
                # Simple linear trend
                x = np.arange(len(confidences))
                coeffs = np.polyfit(x, confidences, 1)
                trends[horizon.value] = coeffs[0]  # Slope
        
        return trends
    
    def _get_horizon_ticks(self, horizon: ForecastHorizon) -> int:
        """Get tick count for horizon"""
        return {
            ForecastHorizon.SHORT_TERM: self.short_term_ticks,
            ForecastHorizon.MID_TERM: self.mid_term_ticks,
            ForecastHorizon.LONG_TERM: self.long_term_ticks
        }[horizon]
    
    def get_forecast_summary(self) -> Dict[str, Any]:
        """Get comprehensive forecast system summary"""
        if not self.forecast_history:
            return {'status': 'no_forecasts_generated'}
        
        latest_forecasts = self.forecast_history[-1]
        
        return {
            'latest_forecasts': {
                horizon.value: {
                    'F_smoothed': forecast.F_smoothed,
                    'stability_zone': forecast.stability_zone.value,
                    'confidence': forecast.confidence,
                    'interventions': [i.value for i in forecast.recommended_interventions]
                }
                for horizon, forecast in latest_forecasts.items()
            },
            'scup_coupling': self.get_scup_early_warning_index(),
            'system_status': {
                'total_forecasts': len(self.forecast_history),
                'smoothing_alpha': self.smoothing_alpha,
                'error_tracking': {
                    error_type: len(errors) 
                    for error_type, errors in self.prediction_errors.items()
                }
            },
            'backtesting_metrics': self.get_backtesting_metrics()
        }


# Global forecasting engine instance
_forecasting_engine = None

def get_forecasting_engine(config: Dict[str, Any] = None) -> ForecastingEngine:
    """Get the global forecasting engine instance"""
    global _forecasting_engine
    if _forecasting_engine is None:
        _forecasting_engine = ForecastingEngine(config)
    return _forecasting_engine


# Example usage and testing
if __name__ == "__main__":
    print("🔮 Testing DAWN Forecasting Engine")
    print("=" * 50)
    
    # Create forecasting engine
    engine = ForecastingEngine()
    
    # Create test inputs
    test_inputs = SystemInputs(
        tick=142990,
        active_node_count=25,
        average_node_health=0.85,
        nutrient_throughput=0.7,
        tracer_concurrency=0.6,
        signal_variance=0.3,
        nutrient_budget=0.8,
        ash_yield=0.2,
        idle_tracer_count=5,
        retirable_tracer_capacity=0.4,
        current_SHI=0.9,
        safe_SHI_threshold=0.8,
        entropy_level=0.4,
        residue_ash_ratio=0.6,
        residue_soot_ratio=0.4,
        shimmer_decay_rate=0.05,
        drift_vectors=[0.1, -0.05, 0.02]
    )
    
    # Generate forecast
    forecasts = engine.generate_forecast(test_inputs)
    
    # Display results
    print(f"Forecast Results for Tick {test_inputs.tick}:")
    for horizon, forecast in forecasts.items():
        print(f"\n{horizon.value.upper()}:")
        print(f"  F_smoothed: {forecast.F_smoothed:.3f}")
        print(f"  Stability Zone: {forecast.stability_zone.value}")
        print(f"  Confidence: {forecast.confidence:.2f}")
        print(f"  Interventions: {[i.value for i in forecast.recommended_interventions]}")
    
    # Get SCUP early warning
    scup_index = engine.get_scup_early_warning_index()
    print(f"\nSCUP Early Warning Index:")
    print(f"  Early Warning Active: {scup_index['early_warning_active']}")
    print(f"  Coherence Loss Probabilities: {scup_index['coherence_loss_probabilities']}")
    
    # Get summary
    summary = engine.get_forecast_summary()
    print(f"\nSystem Summary:")
    print(f"  Total Forecasts: {summary['system_status']['total_forecasts']}")
    print(f"  SCUP Coupling Active: {summary['scup_coupling']['early_warning_active']}")
    
    print("\n🔮 Forecasting Engine operational!")
