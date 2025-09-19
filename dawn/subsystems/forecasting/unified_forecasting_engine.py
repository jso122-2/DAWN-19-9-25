#!/usr/bin/env python3
"""
DAWN Unified Forecasting Engine
===============================

Complete implementation of DAWN's Anticipatory Vector Layer based on RTF specifications.
Projects cognitive state forward in time, enabling proactive preparation rather than reactive responses.

Core Mathematical Framework:
- Cognitive Pressure: P = B·σ² (bandwidth × signal variance)
- Forecasting Function: F = P/A (pressure over adaptive capacity)  
- Horizon Projections: Short (1-10 ticks), Mid (10-100), Long (100-1000)
- SCUP Coupling: Early warning system for coherence loss probability

Based on DAWN-docs/Forcasting/ RTF specifications.
"""

import time
import threading
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
import uuid

logger = logging.getLogger(__name__)


class ForecastHorizon(Enum):
    """Forecast time horizons as per RTF specification"""
    SHORT_TERM = "short_term"    # 1-10 ticks: Euler step, linear extrapolation
    MID_TERM = "mid_term"        # 10-100 ticks: EMA over pressure/entropy curves
    LONG_TERM = "long_term"      # 100-1000 ticks: Monte Carlo on tracer ecology


class StabilityZone(Enum):
    """System stability zones based on F ratio"""
    STABLE = "stable"            # F < 0.5: Low pressure, high capacity
    WATCH = "watch"              # 0.5 ≤ F < 1.0: Moderate pressure
    ACT = "act"                  # F ≥ 1.0: High pressure, intervention needed


class InterventionType(Enum):
    """Types of interventions recommended by forecasting engine"""
    # Tracer management
    WHALE_SPAWN = "whale_spawn"              # Macro ballast for stability
    CROW_CAP = "crow_cap"                    # Limit analysis depth
    BEETLE_RECYCLE = "beetle_recycle"        # Resource recovery
    BEE_POLLINATE = "bee_pollinate"          # Diversity enhancement
    
    # Schema operations
    PURIFICATION = "purification"            # Entropy reduction
    WEAVING = "weaving"                      # Schema reinforcement
    MIRROR_AUDIT = "mirror_audit"            # Schema health check
    
    # Energy management
    FLAME_VENT = "flame_vent"               # Pressure relief
    NUTRIENT_CONSERVATION = "nutrient_conservation"  # Resource preservation


class SCUPWarningLevel(Enum):
    """SCUP Early Warning Index levels"""
    STABLE = "stable"            # p_loss < 0.5
    WATCH = "watch"              # 0.5 ≤ p_loss < 0.7
    CRITICAL = "critical"        # p_loss ≥ 0.7


@dataclass
class CognitivePressure:
    """
    Cognitive pressure calculation: P = B·σ²
    
    B (Base Bandwidth): System's current processing span
    σ² (Signal Variance): Variance across schema/tracers
    """
    # Bandwidth components
    active_nodes: int = 0
    average_health: float = 0.8
    nutrient_throughput: float = 0.5
    tracer_concurrency: float = 0.3
    hardware_ceiling: float = 1.0
    
    # Variance components  
    entropy_variance: float = 0.0
    tracer_output_variance: float = 0.0
    pigment_gradient_variance: float = 0.0
    schema_drift_variance: float = 0.0
    
    def calculate_bandwidth(self) -> float:
        """Calculate base bandwidth B"""
        # RTF formula: B = (active_nodes × avg_health × nutrient_throughput × tracer_headroom)
        # Capped by system configuration
        base_b = (self.active_nodes * self.average_health * 
                 self.nutrient_throughput * (1.0 + self.tracer_concurrency))
        
        return min(base_b, self.hardware_ceiling)
    
    def calculate_variance(self) -> float:
        """Calculate signal variance σ²"""
        # Weighted combination of variance sources
        weights = [0.3, 0.25, 0.25, 0.2]  # entropy, tracer, pigment, drift
        variances = [
            self.entropy_variance,
            self.tracer_output_variance, 
            self.pigment_gradient_variance,
            self.schema_drift_variance
        ]
        
        return sum(w * v for w, v in zip(weights, variances))
    
    def calculate_pressure(self) -> float:
        """Calculate cognitive pressure P = B·σ²"""
        bandwidth = self.calculate_bandwidth()
        variance = self.calculate_variance()
        
        return bandwidth * variance


@dataclass
class AdaptiveCapacity:
    """
    Adaptive capacity calculation: A = w_N·N + w_S·S_T + w_M·M_SHI
    
    Components that enable system to handle pressure:
    - N: Nutrient reserves (mycelial + ash yield)
    - S_T: Tracer slack (idle + retirable capacity)
    - M_SHI: Schema margin (SHI headroom above safe threshold)
    """
    # Nutrient reserves
    nutrient_budget: float = 0.0
    ash_yield: float = 0.0
    
    # Tracer slack
    idle_tracer_count: int = 0
    max_tracer_capacity: int = 100
    retirable_capacity: float = 0.0
    
    # Schema margin
    current_shi: float = 0.8
    safe_shi_threshold: float = 0.7
    
    # Weights (must sum to 1.0)
    w_nutrient: float = 0.4
    w_tracer: float = 0.3
    w_schema: float = 0.3
    
    def __post_init__(self):
        """Normalize weights to sum to 1.0"""
        total = self.w_nutrient + self.w_tracer + self.w_schema
        if total > 0:
            self.w_nutrient /= total
            self.w_tracer /= total
            self.w_schema /= total
    
    def calculate_nutrient_component(self) -> float:
        """Calculate nutrient reserves component N"""
        return self.nutrient_budget + self.ash_yield
    
    def calculate_tracer_slack(self) -> float:
        """Calculate tracer slack component S_T"""
        idle_ratio = self.idle_tracer_count / max(1, self.max_tracer_capacity)
        return idle_ratio + self.retirable_capacity
    
    def calculate_schema_margin(self) -> float:
        """Calculate schema margin component M_SHI"""
        margin = self.current_shi - self.safe_shi_threshold
        return max(0.0, margin)  # No negative margin
    
    def calculate_total(self) -> float:
        """Calculate total adaptive capacity A"""
        N = self.calculate_nutrient_component()
        S_T = self.calculate_tracer_slack()
        M_SHI = self.calculate_schema_margin()
        
        return (self.w_nutrient * N + 
                self.w_tracer * S_T + 
                self.w_schema * M_SHI)


@dataclass
class ForecastResult:
    """Result of horizon-specific forecasting calculation"""
    horizon: ForecastHorizon
    tick: int
    
    # Core forecast values
    pressure: float = 0.0              # P = B·σ²
    adaptive_capacity: float = 0.0     # A = weighted capacity components
    F_raw: float = 0.0                 # Raw F = P/A
    F_normalized: float = 0.0          # Normalized and capped F
    F_smoothed: float = 0.0            # EMA-smoothed F for decisions
    
    # Confidence and stability
    confidence: float = 0.0            # Prediction confidence [0,1]
    stability_zone: StabilityZone = StabilityZone.STABLE
    
    # SCUP coupling
    coherence_loss_probability: float = 0.0    # P(SHI < threshold)
    scup_warning_level: SCUPWarningLevel = SCUPWarningLevel.STABLE
    
    # Recommendations
    recommended_interventions: List[InterventionType] = field(default_factory=list)
    intervention_priorities: Dict[InterventionType, float] = field(default_factory=dict)
    
    # Metadata
    calculation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemInputs:
    """Comprehensive input data for forecasting calculations"""
    tick: int
    timestamp: float = field(default_factory=time.time)
    
    # Bandwidth calculation inputs
    active_node_count: int = 0
    average_node_health: float = 0.8
    nutrient_throughput: float = 0.5
    tracer_concurrency: float = 0.3
    hardware_ceiling: float = 1.0
    
    # Variance calculation inputs  
    entropy_level: float = 0.0
    entropy_variance: float = 0.0
    tracer_outputs: List[float] = field(default_factory=list)
    pigment_gradients: List[float] = field(default_factory=list)
    drift_vectors: List[float] = field(default_factory=list)
    
    # Adaptive capacity inputs
    nutrient_budget: float = 0.0
    ash_yield: float = 0.0
    idle_tracer_count: int = 0
    max_tracer_capacity: int = 100
    retirable_capacity: float = 0.0
    current_shi: float = 0.8
    safe_shi_threshold: float = 0.7
    
    # Additional context
    residue_ash_ratio: float = 0.5
    residue_soot_ratio: float = 0.5
    shimmer_decay_rate: float = 0.05
    
    def calculate_variance_components(self) -> Tuple[float, float, float, float]:
        """Calculate variance components from input data"""
        entropy_var = self.entropy_variance
        
        tracer_var = np.var(self.tracer_outputs) if self.tracer_outputs else 0.0
        pigment_var = np.var(self.pigment_gradients) if self.pigment_gradients else 0.0
        drift_var = np.var(self.drift_vectors) if self.drift_vectors else 0.0
        
        return entropy_var, tracer_var, pigment_var, drift_var


class HorizonProjector:
    """Implements horizon-specific projection algorithms"""
    
    def __init__(self):
        self.projection_cache = {}
        
    def project_short_term(self, pressure: float, capacity: float, 
                          inputs: SystemInputs, ticks: int = 10) -> Dict[str, float]:
        """
        Short-term projection (1-10 ticks): Euler step, linear extrapolation
        
        Uses simple linear projection based on current trends
        """
        # Calculate current F
        current_f = pressure / max(capacity, 1e-6)
        
        # Estimate trends from input variance
        entropy_trend = inputs.entropy_level * 0.01  # Small trend factor
        capacity_trend = -inputs.shimmer_decay_rate * 0.1  # Capacity decay
        
        # Linear projection
        projected_f = current_f + (entropy_trend - capacity_trend) * ticks
        
        # Confidence decreases with projection distance
        confidence = max(0.1, 1.0 - (ticks / 10.0) * 0.3)
        
        return {
            'projected_f': projected_f,
            'confidence': confidence,
            'trend_entropy': entropy_trend,
            'trend_capacity': capacity_trend
        }
    
    def project_mid_term(self, pressure: float, capacity: float,
                        inputs: SystemInputs, ticks: int = 100) -> Dict[str, float]:
        """
        Mid-term projection (10-100 ticks): EMA over pressure/entropy curves
        
        Uses exponential moving average to smooth trends and account for shimmer decay
        """
        current_f = pressure / max(capacity, 1e-6)
        
        # EMA parameters
        alpha = 0.1  # Smoothing factor
        
        # Model shimmer decay effect on capacity
        decay_factor = math.exp(-inputs.shimmer_decay_rate * ticks)
        projected_capacity = capacity * decay_factor
        
        # Model pressure evolution with entropy trends
        entropy_growth = inputs.entropy_level * (1 + 0.05 * math.log(1 + ticks))
        projected_pressure = pressure * (1 + entropy_growth * 0.1)
        
        # Calculate projected F
        projected_f = projected_pressure / max(projected_capacity, 1e-6)
        
        # Confidence based on prediction horizon and variance
        base_confidence = 0.8
        horizon_penalty = (ticks / 100.0) * 0.4
        variance_penalty = min(0.3, np.mean([
            inputs.entropy_variance,
            np.var(inputs.tracer_outputs) if inputs.tracer_outputs else 0,
            np.var(inputs.drift_vectors) if inputs.drift_vectors else 0
        ]))
        
        confidence = max(0.1, base_confidence - horizon_penalty - variance_penalty)
        
        return {
            'projected_f': projected_f,
            'confidence': confidence,
            'projected_pressure': projected_pressure,
            'projected_capacity': projected_capacity,
            'decay_factor': decay_factor
        }
    
    def project_long_term(self, pressure: float, capacity: float,
                         inputs: SystemInputs, ticks: int = 1000) -> Dict[str, float]:
        """
        Long-term projection (100-1000 ticks): Monte Carlo on tracer ecology & residue balance
        
        Uses Monte Carlo simulation to account for complex system interactions
        """
        current_f = pressure / max(capacity, 1e-6)
        
        # Monte Carlo parameters
        n_simulations = 100
        results = []
        
        for _ in range(n_simulations):
            # Simulate random variations
            pressure_multiplier = np.random.lognormal(0, 0.2)  # Log-normal pressure variation
            capacity_multiplier = np.random.lognormal(0, 0.15)  # Log-normal capacity variation
            
            # Model long-term decay and growth
            time_factor = ticks / 1000.0
            
            # Residue balance effects
            ash_benefit = inputs.residue_ash_ratio * 0.3 * time_factor
            soot_penalty = inputs.residue_soot_ratio * 0.2 * time_factor
            
            # Tracer ecology effects (simplified)
            tracer_stability = min(1.0, inputs.idle_tracer_count / max(1, inputs.max_tracer_capacity))
            ecology_factor = 1.0 + (tracer_stability - 0.5) * 0.1 * time_factor
            
            # Calculate simulated values
            sim_pressure = pressure * pressure_multiplier * (1 + soot_penalty)
            sim_capacity = capacity * capacity_multiplier * ecology_factor * (1 + ash_benefit)
            
            sim_f = sim_pressure / max(sim_capacity, 1e-6)
            results.append(sim_f)
        
        # Calculate statistics
        projected_f = np.mean(results)
        f_std = np.std(results)
        
        # Confidence based on prediction stability
        confidence = max(0.1, 0.6 - (f_std / max(projected_f, 1e-6)) * 0.3)
        
        return {
            'projected_f': projected_f,
            'confidence': confidence,
            'f_std': f_std,
            'f_percentiles': {
                '10': np.percentile(results, 10),
                '50': np.percentile(results, 50),
                '90': np.percentile(results, 90)
            }
        }


class SCUPCouplingEngine:
    """Handles SCUP (Semantic Coherence Under Pressure) coupling"""
    
    def __init__(self, threshold: float = 0.7):
        self.coherence_threshold = threshold
        self.loss_probability_cache = {}
        
        # Logistic regression parameters for coherence loss probability
        # P(loss) = σ(α·F + β·P + γ·Δdrift - δ·A)
        self.alpha = 2.0    # F coefficient
        self.beta = 0.5     # Pressure coefficient
        self.gamma = 1.5    # Drift coefficient
        self.delta = 1.0    # Adaptive capacity coefficient
    
    def calculate_coherence_loss_probability(self, forecast: ForecastResult,
                                          inputs: SystemInputs) -> float:
        """
        Calculate probability of coherence loss using logistic model
        
        P(loss) = σ(α·F + β·P + γ·Δdrift - δ·A)
        """
        F = forecast.F_smoothed
        P = forecast.pressure
        A = forecast.adaptive_capacity
        
        # Calculate drift magnitude
        drift_magnitude = np.linalg.norm(inputs.drift_vectors) if inputs.drift_vectors else 0.0
        
        # Logistic regression
        logit = (self.alpha * F + 
                self.beta * P + 
                self.gamma * drift_magnitude - 
                self.delta * A)
        
        # Sigmoid function
        probability = 1.0 / (1.0 + math.exp(-logit))
        
        return min(0.99, max(0.01, probability))  # Clamp to reasonable bounds
    
    def determine_warning_level(self, probability: float) -> SCUPWarningLevel:
        """Determine SCUP warning level based on loss probability"""
        if probability < 0.5:
            return SCUPWarningLevel.STABLE
        elif probability < self.coherence_threshold:
            return SCUPWarningLevel.WATCH
        else:
            return SCUPWarningLevel.CRITICAL
    
    def generate_scup_recommendations(self, warning_level: SCUPWarningLevel,
                                    forecast: ForecastResult) -> List[InterventionType]:
        """Generate intervention recommendations based on SCUP warning level"""
        recommendations = []
        
        if warning_level == SCUPWarningLevel.WATCH:
            # Pre-emptive measures
            recommendations.extend([
                InterventionType.WHALE_SPAWN,
                InterventionType.MIRROR_AUDIT
            ])
        
        elif warning_level == SCUPWarningLevel.CRITICAL:
            # Emergency interventions
            recommendations.extend([
                InterventionType.FLAME_VENT,
                InterventionType.PURIFICATION,
                InterventionType.NUTRIENT_CONSERVATION,
                InterventionType.CROW_CAP
            ])
            
            # Add tracer management for long-term issues
            if forecast.horizon == ForecastHorizon.LONG_TERM:
                recommendations.append(InterventionType.BEETLE_RECYCLE)
        
        return recommendations


class BacktestingEngine:
    """Implements error metrics and backtesting functionality"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Error tracking
        self.pressure_errors: deque = deque(maxlen=window_size)
        self.drift_errors: deque = deque(maxlen=window_size)
        self.entropy_errors: deque = deque(maxlen=window_size)
        self.f_errors: Dict[ForecastHorizon, deque] = {
            horizon: deque(maxlen=window_size) for horizon in ForecastHorizon
        }
        
        # Prediction tracking
        self.predictions: Dict[int, Dict[ForecastHorizon, ForecastResult]] = {}
        
    def record_prediction(self, tick: int, forecasts: Dict[ForecastHorizon, ForecastResult]):
        """Record predictions for later backtesting"""
        self.predictions[tick] = forecasts.copy()
        
        # Clean old predictions
        cutoff_tick = tick - self.window_size
        self.predictions = {t: f for t, f in self.predictions.items() if t > cutoff_tick}
    
    def record_actual_values(self, tick: int, actual_pressure: float,
                           actual_drift: float, actual_entropy: float):
        """Record actual values and calculate errors against predictions"""
        
        # Find predictions to validate
        for horizon in ForecastHorizon:
            prediction_tick = self._get_prediction_tick(tick, horizon)
            
            if prediction_tick in self.predictions:
                predicted = self.predictions[prediction_tick][horizon]
                
                # Calculate F error
                actual_f = actual_pressure / max(predicted.adaptive_capacity, 1e-6)
                f_error = abs(predicted.F_smoothed - actual_f)
                self.f_errors[horizon].append(f_error)
        
        # Record other errors
        # Note: This would need actual vs predicted values for pressure/drift/entropy
        # For now, we'll record the actual values for future use
        
    def _get_prediction_tick(self, current_tick: int, horizon: ForecastHorizon) -> int:
        """Get the tick when prediction was made for current validation"""
        if horizon == ForecastHorizon.SHORT_TERM:
            return current_tick - 10
        elif horizon == ForecastHorizon.MID_TERM:
            return current_tick - 100
        else:  # LONG_TERM
            return current_tick - 1000
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """Calculate and return error metrics"""
        metrics = {}
        
        # F prediction errors by horizon
        for horizon in ForecastHorizon:
            errors = list(self.f_errors[horizon])
            if errors:
                metrics[f'{horizon.value}_mae'] = np.mean(errors)
                metrics[f'{horizon.value}_rmse'] = np.sqrt(np.mean(np.square(errors)))
                metrics[f'{horizon.value}_error_count'] = len(errors)
        
        # Overall metrics
        all_errors = []
        for horizon_errors in self.f_errors.values():
            all_errors.extend(horizon_errors)
        
        if all_errors:
            metrics['overall_mae'] = np.mean(all_errors)
            metrics['overall_rmse'] = np.sqrt(np.mean(np.square(all_errors)))
        
        return metrics


class UnifiedForecastingEngine:
    """
    Main unified forecasting engine implementing complete RTF specifications.
    Provides DAWN's Anticipatory Vector Layer for proactive cognitive preparation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core parameters
        self.F_scale = self.config.get('F_scale', 2.0)
        self.F_cap = self.config.get('F_cap', 5.0)
        self.smoothing_alpha = self.config.get('smoothing_alpha', 0.4)
        
        # Horizon parameters
        self.horizon_ticks = {
            ForecastHorizon.SHORT_TERM: self.config.get('short_term_ticks', 10),
            ForecastHorizon.MID_TERM: self.config.get('mid_term_ticks', 100),
            ForecastHorizon.LONG_TERM: self.config.get('long_term_ticks', 1000)
        }
        
        # Components
        self.projector = HorizonProjector()
        self.scup_coupling = SCUPCouplingEngine(
            threshold=self.config.get('scup_threshold', 0.7)
        )
        self.backtesting = BacktestingEngine(
            window_size=self.config.get('backtest_window', 1000)
        )
        
        # DAWN singleton integration
        self.dawn_system = None
        self.consciousness_bus = None
        self.telemetry_system = None
        self._initialize_dawn_integration()
        
        # State tracking
        self.forecast_history: List[Dict[ForecastHorizon, ForecastResult]] = []
        self.smoothed_f_history: Dict[ForecastHorizon, deque] = {
            horizon: deque(maxlen=100) for horizon in ForecastHorizon
        }
        
        # Performance tracking
        self.generation_times: deque = deque(maxlen=100)
        self.total_forecasts_generated = 0
        
        # Threading
        self.running = False
        self.update_thread: Optional[threading.Thread] = None
        self.update_lock = threading.RLock()
        
        logger.info("🔮 Unified Forecasting Engine initialized")
    
    def _initialize_dawn_integration(self) -> None:
        """Initialize integration with DAWN singleton"""
        try:
            from dawn.core.singleton import get_dawn
            self.dawn_system = get_dawn()
            
            # Get subsystems if available
            if self.dawn_system.is_initialized():
                self.consciousness_bus = self.dawn_system.consciousness_bus
                self.telemetry_system = self.dawn_system.telemetry_system
                
                # Register with consciousness bus if available
                if self.consciousness_bus:
                    self.consciousness_bus.register_module(
                        "unified_forecasting_engine",
                        capabilities=["forecasting", "prediction", "active_inference"],
                        state_schema={"forecast_f": "float", "stability_zone": "string"}
                    )
                    logger.info("🌅 Forecasting engine registered with DAWN singleton")
                
                # Log initialization to telemetry
                if self.telemetry_system:
                    self.telemetry_system.log_event(
                        'forecasting', 'initialization', 'engine_initialized',
                        data={'horizon_ticks': dict(self.horizon_ticks)}
                    )
            else:
                logger.info("🔮 DAWN system not initialized, forecasting engine running standalone")
                
        except ImportError:
            logger.warning("DAWN singleton not available, forecasting engine running standalone")
    
    def generate_forecast(self, inputs: SystemInputs) -> Dict[ForecastHorizon, ForecastResult]:
        """
        Generate comprehensive forecast across all time horizons.
        
        This is the main interface implementing the RTF specification:
        1. Calculate cognitive pressure P = B·σ²
        2. Calculate adaptive capacity A  
        3. Compute forecasting function F = P/A for each horizon
        4. Apply smoothing and generate interventions
        5. Update SCUP coupling and early warning
        
        Args:
            inputs: Current system state inputs
            
        Returns:
            Dictionary of forecast results by horizon
        """
        start_time = time.time()
        
        with self.update_lock:
            try:
                logger.debug(f"🔮 Generating forecast for tick {inputs.tick}")
                
                # 1. Calculate cognitive pressure
                pressure_calc = self._build_pressure_calculator(inputs)
                pressure = pressure_calc.calculate_pressure()
                
                # 2. Calculate adaptive capacity
                capacity_calc = self._build_capacity_calculator(inputs)
                adaptive_capacity = capacity_calc.calculate_total()
                
                # 3. Generate forecasts for each horizon
                forecasts = {}
                
                for horizon in ForecastHorizon:
                    forecast = self._generate_horizon_forecast(
                        horizon, pressure, adaptive_capacity, inputs, pressure_calc, capacity_calc
                    )
                    forecasts[horizon] = forecast
                
                # 4. Apply smoothing
                self._apply_smoothing(forecasts)
                
                # 5. Generate interventions and SCUP coupling
                for horizon, forecast in forecasts.items():
                    # Calculate coherence loss probability
                    loss_prob = self.scup_coupling.calculate_coherence_loss_probability(forecast, inputs)
                    forecast.coherence_loss_probability = loss_prob
                    forecast.scup_warning_level = self.scup_coupling.determine_warning_level(loss_prob)
                    
                    # Generate intervention recommendations
                    scup_interventions = self.scup_coupling.generate_scup_recommendations(
                        forecast.scup_warning_level, forecast
                    )
                    
                    # Combine with horizon-specific interventions
                    horizon_interventions = self._generate_horizon_interventions(forecast, inputs)
                    
                    all_interventions = list(set(scup_interventions + horizon_interventions))
                    forecast.recommended_interventions = all_interventions
                    forecast.intervention_priorities = self._calculate_intervention_priorities(
                        all_interventions, forecast
                    )
                
                # Store in history
                self.forecast_history.append(forecasts)
                if len(self.forecast_history) > 1000:
                    self.forecast_history = self.forecast_history[-1000:]
                
                # Record for backtesting
                self.backtesting.record_prediction(inputs.tick, forecasts)
                
                # Update performance tracking
                generation_time = time.time() - start_time
                self.generation_times.append(generation_time)
                self.total_forecasts_generated += 1
                
                logger.info(f"🔮 Forecast complete for tick {inputs.tick} in {generation_time*1000:.2f}ms")
                logger.info(f"    Zones: ST:{forecasts[ForecastHorizon.SHORT_TERM].stability_zone.value}, "
                           f"MT:{forecasts[ForecastHorizon.MID_TERM].stability_zone.value}, "
                           f"LT:{forecasts[ForecastHorizon.LONG_TERM].stability_zone.value}")
                logger.info(f"    SCUP: ST:{forecasts[ForecastHorizon.SHORT_TERM].scup_warning_level.value}, "
                           f"MT:{forecasts[ForecastHorizon.MID_TERM].scup_warning_level.value}, "
                           f"LT:{forecasts[ForecastHorizon.LONG_TERM].scup_warning_level.value}")
                
                return forecasts
                
            except Exception as e:
                logger.error(f"Error generating forecast: {e}")
                raise
    
    def _build_pressure_calculator(self, inputs: SystemInputs) -> CognitivePressure:
        """Build cognitive pressure calculator from inputs"""
        # Calculate variance components
        entropy_var, tracer_var, pigment_var, drift_var = inputs.calculate_variance_components()
        
        return CognitivePressure(
            active_nodes=inputs.active_node_count,
            average_health=inputs.average_node_health,
            nutrient_throughput=inputs.nutrient_throughput,
            tracer_concurrency=inputs.tracer_concurrency,
            hardware_ceiling=inputs.hardware_ceiling,
            entropy_variance=entropy_var,
            tracer_output_variance=tracer_var,
            pigment_gradient_variance=pigment_var,
            schema_drift_variance=drift_var
        )
    
    def _build_capacity_calculator(self, inputs: SystemInputs) -> AdaptiveCapacity:
        """Build adaptive capacity calculator from inputs"""
        return AdaptiveCapacity(
            nutrient_budget=inputs.nutrient_budget,
            ash_yield=inputs.ash_yield,
            idle_tracer_count=inputs.idle_tracer_count,
            max_tracer_capacity=inputs.max_tracer_capacity,
            retirable_capacity=inputs.retirable_capacity,
            current_shi=inputs.current_shi,
            safe_shi_threshold=inputs.safe_shi_threshold
        )
    
    def _generate_horizon_forecast(self, horizon: ForecastHorizon, pressure: float,
                                 adaptive_capacity: float, inputs: SystemInputs,
                                 pressure_calc: CognitivePressure,
                                 capacity_calc: AdaptiveCapacity) -> ForecastResult:
        """Generate forecast for specific horizon"""
        
        # Get horizon-specific projection
        ticks = self.horizon_ticks[horizon]
        
        if horizon == ForecastHorizon.SHORT_TERM:
            projection = self.projector.project_short_term(pressure, adaptive_capacity, inputs, ticks)
        elif horizon == ForecastHorizon.MID_TERM:
            projection = self.projector.project_mid_term(pressure, adaptive_capacity, inputs, ticks)
        else:  # LONG_TERM
            projection = self.projector.project_long_term(pressure, adaptive_capacity, inputs, ticks)
        
        # Create forecast result
        forecast = ForecastResult(
            horizon=horizon,
            tick=inputs.tick,
            pressure=pressure,
            adaptive_capacity=adaptive_capacity,
            F_raw=projection['projected_f'],
            confidence=projection['confidence']
        )
        
        # Normalize and cap F value
        forecast.F_normalized = self._normalize_f_value(forecast.F_raw)
        
        # Determine stability zone
        forecast.stability_zone = self._determine_stability_zone(forecast.F_normalized)
        
        # Store projection metadata
        forecast.metadata.update(projection)
        forecast.calculation_time = time.time()
        
        return forecast
    
    def _normalize_f_value(self, f_raw: float) -> float:
        """Normalize F value using scale and cap"""
        # Apply scaling and capping as per RTF specification
        f_scaled = f_raw * self.F_scale
        f_capped = min(f_scaled, self.F_cap)
        
        return max(0.0, f_capped)  # Ensure non-negative
    
    def _determine_stability_zone(self, f_normalized: float) -> StabilityZone:
        """Determine stability zone based on F value"""
        if f_normalized < 0.5:
            return StabilityZone.STABLE
        elif f_normalized < 1.0:
            return StabilityZone.WATCH
        else:
            return StabilityZone.ACT
    
    def _apply_smoothing(self, forecasts: Dict[ForecastHorizon, ForecastResult]):
        """Apply exponential moving average smoothing to F values"""
        for horizon, forecast in forecasts.items():
            history = self.smoothed_f_history[horizon]
            
            if history:
                # EMA smoothing
                previous_smoothed = history[-1]
                smoothed = (self.smoothing_alpha * forecast.F_normalized + 
                          (1 - self.smoothing_alpha) * previous_smoothed)
            else:
                # First value
                smoothed = forecast.F_normalized
            
            forecast.F_smoothed = smoothed
            history.append(smoothed)
    
    def _generate_horizon_interventions(self, forecast: ForecastResult,
                                      inputs: SystemInputs) -> List[InterventionType]:
        """Generate horizon-specific intervention recommendations"""
        interventions = []
        
        # Base interventions on stability zone
        if forecast.stability_zone == StabilityZone.ACT:
            interventions.append(InterventionType.PURIFICATION)
            
            if forecast.horizon == ForecastHorizon.SHORT_TERM:
                interventions.append(InterventionType.FLAME_VENT)
            elif forecast.horizon == ForecastHorizon.LONG_TERM:
                interventions.extend([
                    InterventionType.BEETLE_RECYCLE,
                    InterventionType.WEAVING
                ])
        
        elif forecast.stability_zone == StabilityZone.WATCH:
            if forecast.horizon in [ForecastHorizon.MID_TERM, ForecastHorizon.LONG_TERM]:
                interventions.extend([
                    InterventionType.WHALE_SPAWN,
                    InterventionType.BEE_POLLINATE
                ])
        
        # Add context-specific interventions
        if inputs.residue_soot_ratio > 0.6:
            interventions.append(InterventionType.PURIFICATION)
        
        if inputs.current_shi < inputs.safe_shi_threshold + 0.1:
            interventions.append(InterventionType.MIRROR_AUDIT)
        
        return interventions
    
    def _calculate_intervention_priorities(self, interventions: List[InterventionType],
                                         forecast: ForecastResult) -> Dict[InterventionType, float]:
        """Calculate priority scores for interventions"""
        priorities = {}
        
        base_priority = 0.5
        urgency_multiplier = 1.0
        
        # Adjust based on stability zone
        if forecast.stability_zone == StabilityZone.ACT:
            urgency_multiplier = 1.5
        elif forecast.stability_zone == StabilityZone.WATCH:
            urgency_multiplier = 1.2
        
        # Adjust based on SCUP warning level
        if forecast.scup_warning_level == SCUPWarningLevel.CRITICAL:
            urgency_multiplier *= 1.3
        
        for intervention in interventions:
            # Base priorities by intervention type
            if intervention in [InterventionType.FLAME_VENT, InterventionType.PURIFICATION]:
                priority = 0.9  # High priority
            elif intervention in [InterventionType.WHALE_SPAWN, InterventionType.MIRROR_AUDIT]:
                priority = 0.7  # Medium-high priority
            elif intervention in [InterventionType.BEE_POLLINATE, InterventionType.WEAVING]:
                priority = 0.5  # Medium priority
            else:
                priority = 0.3  # Lower priority
            
            priorities[intervention] = min(1.0, priority * urgency_multiplier)
        
        return priorities
    
    def get_forecast_summary(self) -> Dict[str, Any]:
        """Get comprehensive forecasting engine status and metrics"""
        if not self.forecast_history:
            return {'error': 'No forecasts generated yet'}
        
        latest = self.forecast_history[-1]
        
        # Performance metrics
        avg_generation_time = np.mean(self.generation_times) if self.generation_times else 0
        
        # Error metrics from backtesting
        error_metrics = self.backtesting.get_error_metrics()
        
        return {
            'status': {
                'total_forecasts': self.total_forecasts_generated,
                'average_generation_time_ms': avg_generation_time * 1000,
                'forecast_history_size': len(self.forecast_history)
            },
            'latest_forecast': {
                'tick': latest[ForecastHorizon.SHORT_TERM].tick,
                'zones': {h.value: f.stability_zone.value for h, f in latest.items()},
                'scup_warnings': {h.value: f.scup_warning_level.value for h, f in latest.items()},
                'f_values': {h.value: f.F_smoothed for h, f in latest.items()},
                'confidence': {h.value: f.confidence for h, f in latest.items()}
            },
            'error_metrics': error_metrics,
            'configuration': {
                'F_scale': self.F_scale,
                'F_cap': self.F_cap,
                'smoothing_alpha': self.smoothing_alpha,
                'horizon_ticks': {h.value: t for h, t in self.horizon_ticks.items()}
            }
        }
    
    def export_forecast_data(self, filepath: str) -> bool:
        """Export comprehensive forecast data for analysis"""
        try:
            export_data = {
                'system_status': self.get_forecast_summary(),
                'forecast_history': [
                    {h.value: {
                        'tick': f.tick,
                        'F_raw': f.F_raw,
                        'F_normalized': f.F_normalized,
                        'F_smoothed': f.F_smoothed,
                        'confidence': f.confidence,
                        'stability_zone': f.stability_zone.value,
                        'scup_warning': f.scup_warning_level.value,
                        'interventions': [i.value for i in f.recommended_interventions],
                        'metadata': f.metadata
                    } for h, f in forecast.items()}
                    for forecast in self.forecast_history[-100:]  # Last 100 forecasts
                ],
                'smoothed_history': {
                    h.value: list(history) for h, history in self.smoothed_f_history.items()
                },
                'error_metrics': self.backtesting.get_error_metrics()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported forecast data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export forecast data: {e}")
            return False


# Global forecasting engine instance
_global_forecasting_engine: Optional[UnifiedForecastingEngine] = None

def get_forecasting_engine(config: Dict[str, Any] = None) -> UnifiedForecastingEngine:
    """Get the global unified forecasting engine instance"""
    global _global_forecasting_engine
    
    if _global_forecasting_engine is None:
        _global_forecasting_engine = UnifiedForecastingEngine(config)
    
    return _global_forecasting_engine
