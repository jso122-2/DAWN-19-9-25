#!/usr/bin/env python3
"""
DAWN Telemetry Analytics Engine
==============================

Real-time analysis of DAWN's operational telemetry data.
Transforms raw metrics into actionable intelligence with predictive insights,
performance optimization recommendations, and automated system tuning.

Key Features:
- Real-time telemetry stream processing
- Cognitive performance pattern analysis
- Predictive maintenance and capacity planning
- Automated optimization recommendations
- Dashboard data preparation and export
"""

import time
import json
import threading
import uuid
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from pathlib import Path
from enum import Enum
import statistics
import queue
import psutil
import weakref
from concurrent.futures import ThreadPoolExecutor
import sqlite3

logger = logging.getLogger(__name__)

class InsightType(Enum):
    """Types of analytical insights."""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    RESOURCE_ALERT = "resource_alert"
    COGNITIVE_PATTERN = "cognitive_pattern"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    CAPACITY_PLANNING = "capacity_planning"
    ERROR_PATTERN = "error_pattern"
    CONFIGURATION_TUNING = "configuration_tuning"
    BOTTLENECK_DETECTION = "bottleneck_detection"

class RiskLevel(Enum):
    """Risk levels for recommendations."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TelemetryDataPoint:
    """Single telemetry measurement."""
    timestamp: datetime
    source: str
    metric_name: str
    value: Union[int, float, str, bool]
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance analysis results."""
    timestamp: datetime
    tick_rate_trend: Dict[str, float]
    memory_efficiency: Dict[str, float]
    sigil_cascade_efficiency: Dict[str, float]
    recursive_stability: Dict[str, float]
    bottleneck_identification: List[Dict[str, Any]]
    resource_utilization: Dict[str, float]
    overall_health_score: float
    cognitive_load_distribution: Dict[str, float]

@dataclass
class AnalyticalInsight:
    """Automated analytical insight with recommendations."""
    insight_id: str
    insight_type: InsightType
    timestamp: datetime
    confidence: float
    recommendation: str
    reasoning: str
    expected_improvement: str
    risk_level: RiskLevel
    affected_systems: List[str]
    implementation_priority: int
    validation_metrics: List[str]
    rollback_plan: Optional[str] = None
    estimated_impact: Optional[Dict[str, float]] = None

@dataclass
class PredictiveAnalysis:
    """Predictive analytics results."""
    prediction_id: str
    timestamp: datetime
    prediction_horizon: timedelta
    prediction_type: str
    predicted_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    triggering_conditions: List[str]
    recommended_actions: List[str]
    accuracy_score: Optional[float] = None

class TelemetryBuffer:
    """High-performance circular buffer for telemetry data."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.index = {}
        self.lock = threading.RLock()
        
    def add(self, data_point: TelemetryDataPoint):
        """Add a data point to the buffer."""
        with self.lock:
            self.buffer.append(data_point)
            
            # Update index for fast lookups
            key = f"{data_point.source}:{data_point.metric_name}"
            if key not in self.index:
                self.index[key] = deque(maxlen=1000)
            self.index[key].append(len(self.buffer) - 1)
            
    def get_recent(self, source: str, metric: str, count: int = 100) -> List[TelemetryDataPoint]:
        """Get recent data points for a specific metric."""
        with self.lock:
            key = f"{source}:{metric}"
            if key not in self.index:
                return []
                
            indices = list(self.index[key])[-count:]
            return [self.buffer[i] for i in indices if i < len(self.buffer)]
            
    def get_time_range(self, start_time: datetime, end_time: datetime) -> List[TelemetryDataPoint]:
        """Get data points within a time range."""
        with self.lock:
            return [dp for dp in self.buffer 
                   if start_time <= dp.timestamp <= end_time]
                   
    def get_metrics_summary(self) -> Dict[str, int]:
        """Get summary of available metrics."""
        with self.lock:
            return {key: len(indices) for key, indices in self.index.items()}

class CognitivePerformanceAnalyzer:
    """Analyzes cognitive performance patterns and bottlenecks."""
    
    def __init__(self, telemetry_buffer: TelemetryBuffer):
        self.buffer = telemetry_buffer
        self.performance_history = deque(maxlen=500)
        self.baseline_metrics = {}
        self.analysis_cache = {}
        self.cache_ttl = 30  # seconds
        
    def analyze_cognitive_performance(self) -> PerformanceMetrics:
        """Comprehensive cognitive performance analysis."""
        current_time = datetime.now()
        
        # Check cache first
        cache_key = "cognitive_performance"
        if (cache_key in self.analysis_cache and 
            (current_time - self.analysis_cache[cache_key]['timestamp']).seconds < self.cache_ttl):
            return self.analysis_cache[cache_key]['result']
            
        # Analyze tick rate performance
        tick_rate_trend = self._analyze_tick_rate_trend()
        
        # Analyze memory efficiency
        memory_efficiency = self._analyze_memory_efficiency()
        
        # Analyze sigil cascade efficiency
        sigil_cascade_efficiency = self._analyze_sigil_cascade_efficiency()
        
        # Analyze recursive stability
        recursive_stability = self._analyze_recursive_stability()
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks()
        
        # Analyze resource utilization
        resource_utilization = self._analyze_resource_utilization()
        
        # Calculate cognitive load distribution
        cognitive_load = self._analyze_cognitive_load_distribution()
        
        # Calculate overall health score
        health_score = self._calculate_overall_health_score(
            tick_rate_trend, memory_efficiency, sigil_cascade_efficiency,
            recursive_stability, resource_utilization
        )
        
        metrics = PerformanceMetrics(
            timestamp=current_time,
            tick_rate_trend=tick_rate_trend,
            memory_efficiency=memory_efficiency,
            sigil_cascade_efficiency=sigil_cascade_efficiency,
            recursive_stability=recursive_stability,
            bottleneck_identification=bottlenecks,
            resource_utilization=resource_utilization,
            overall_health_score=health_score,
            cognitive_load_distribution=cognitive_load
        )
        
        # Cache result
        self.analysis_cache[cache_key] = {
            'timestamp': current_time,
            'result': metrics
        }
        
        # Store in history
        self.performance_history.append(metrics)
        
        return metrics
        
    def _analyze_tick_rate_trend(self) -> Dict[str, float]:
        """Analyze tick engine performance trends."""
        tick_data = self.buffer.get_recent("tick_engine", "tick_rate", 100)
        
        if len(tick_data) < 10:
            return {'current_rate': 0.0, 'trend': 0.0, 'stability': 0.0, 'efficiency': 0.0}
            
        values = [dp.value for dp in tick_data]
        timestamps = [dp.timestamp.timestamp() for dp in tick_data]
        
        # Calculate trend
        if len(values) > 1:
            trend = np.polyfit(range(len(values)), values, 1)[0]
        else:
            trend = 0.0
            
        # Calculate stability (inverse of coefficient of variation)
        stability = 1.0 - (np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0
        stability = max(0.0, min(1.0, stability))
        
        # Calculate efficiency (rate vs target)
        target_rate = 10.0  # Target ticks per second
        current_rate = np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
        efficiency = min(1.0, current_rate / target_rate) if target_rate > 0 else 0.0
        
        return {
            'current_rate': float(current_rate),
            'trend': float(trend),
            'stability': float(stability),
            'efficiency': float(efficiency),
            'target_rate': float(target_rate),
            'samples': len(values)
        }
        
    def _analyze_memory_efficiency(self) -> Dict[str, float]:
        """Analyze memory usage and rebloom success rates."""
        # Get memory usage data
        memory_data = self.buffer.get_recent("system", "memory_usage", 50)
        rebloom_data = self.buffer.get_recent("memory", "rebloom_success_rate", 50)
        
        memory_efficiency = 1.0
        rebloom_efficiency = 1.0
        
        if memory_data:
            memory_values = [dp.value for dp in memory_data]
            memory_usage = np.mean(memory_values)
            # Efficiency decreases as memory usage increases beyond 70%
            memory_efficiency = max(0.0, 1.0 - max(0.0, memory_usage - 0.7) / 0.3)
            
        if rebloom_data:
            rebloom_values = [dp.value for dp in rebloom_data]
            rebloom_efficiency = np.mean(rebloom_values)
            
        # Calculate memory growth rate
        growth_rate = 0.0
        if len(memory_data) > 10:
            memory_values = [dp.value for dp in memory_data]
            growth_rate = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
            
        return {
            'memory_efficiency': float(memory_efficiency),
            'rebloom_success_rate': float(rebloom_efficiency),
            'memory_growth_rate': float(growth_rate),
            'current_usage': float(memory_values[-1]) if memory_data else 0.0,
            'stability': float(1.0 - np.std(memory_values) / np.mean(memory_values)) if memory_data and np.mean(memory_values) > 0 else 1.0
        }
        
    def _analyze_sigil_cascade_efficiency(self) -> Dict[str, float]:
        """Analyze sigil cascade performance and completion times."""
        cascade_data = self.buffer.get_recent("sigil_engine", "cascade_completion_time", 50)
        success_data = self.buffer.get_recent("sigil_engine", "cascade_success_rate", 50)
        depth_data = self.buffer.get_recent("sigil_engine", "cascade_depth", 50)
        
        completion_efficiency = 1.0
        success_rate = 1.0
        depth_efficiency = 1.0
        
        if cascade_data:
            completion_times = [dp.value for dp in cascade_data]
            # Efficiency based on completion time (target: <100ms)
            avg_completion_time = np.mean(completion_times)
            completion_efficiency = max(0.0, 1.0 - max(0.0, avg_completion_time - 100) / 500)
            
        if success_data:
            success_values = [dp.value for dp in success_data]
            success_rate = np.mean(success_values)
            
        if depth_data:
            depth_values = [dp.value for dp in depth_data]
            avg_depth = np.mean(depth_values)
            # Optimal depth is 2-4, efficiency decreases beyond that
            if avg_depth <= 4:
                depth_efficiency = 1.0
            else:
                depth_efficiency = max(0.0, 1.0 - (avg_depth - 4) / 6)
                
        return {
            'completion_efficiency': float(completion_efficiency),
            'success_rate': float(success_rate),
            'depth_efficiency': float(depth_efficiency),
            'avg_completion_time': float(avg_completion_time) if cascade_data else 0.0,
            'avg_depth': float(avg_depth) if depth_data else 0.0,
            'cascade_stability': float(1.0 - np.std(completion_times) / np.mean(completion_times)) if cascade_data and np.mean(completion_times) > 0 else 1.0
        }
        
    def _analyze_recursive_stability(self) -> Dict[str, float]:
        """Track recursion health over time."""
        depth_data = self.buffer.get_recent("recursive_bubble", "current_depth", 100)
        max_depth_data = self.buffer.get_recent("recursive_bubble", "max_depth_reached", 50)
        stabilization_data = self.buffer.get_recent("recursive_bubble", "stabilization_count", 50)
        
        depth_stability = 1.0
        max_depth_efficiency = 1.0
        stabilization_rate = 1.0
        
        if depth_data:
            depth_values = [dp.value for dp in depth_data]
            # Stability is inverse of depth variance
            if len(depth_values) > 1 and np.mean(depth_values) > 0:
                depth_stability = 1.0 - min(1.0, np.std(depth_values) / np.mean(depth_values))
            else:
                depth_stability = 1.0
                
        if max_depth_data:
            max_depths = [dp.value for dp in max_depth_data]
            avg_max_depth = np.mean(max_depths)
            # Efficiency decreases as max depth exceeds safe limits (target: <8)
            max_depth_efficiency = max(0.0, 1.0 - max(0.0, avg_max_depth - 8) / 8)
            
        if stabilization_data:
            stabilizations = [dp.value for dp in stabilization_data]
            # Higher stabilization rate indicates good recursive health
            stabilization_rate = min(1.0, np.mean(stabilizations) / 10.0)  # Target: 10 stabilizations per period
            
        return {
            'depth_stability': float(depth_stability),
            'max_depth_efficiency': float(max_depth_efficiency),
            'stabilization_rate': float(stabilization_rate),
            'current_depth': float(depth_values[-1]) if depth_data else 0.0,
            'avg_max_depth': float(avg_max_depth) if max_depth_data else 0.0,
            'recursive_health': float((depth_stability + max_depth_efficiency + stabilization_rate) / 3.0)
        }
        
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks across systems."""
        bottlenecks = []
        
        # CPU bottleneck detection
        cpu_data = self.buffer.get_recent("system", "cpu_usage", 20)
        if cpu_data:
            cpu_values = [dp.value for dp in cpu_data]
            avg_cpu = np.mean(cpu_values)
            if avg_cpu > 0.8:  # 80% CPU usage
                bottlenecks.append({
                    'type': 'cpu_bottleneck',
                    'severity': 'high' if avg_cpu > 0.9 else 'medium',
                    'value': float(avg_cpu),
                    'description': f'High CPU usage: {avg_cpu:.1%}',
                    'recommendation': 'Consider reducing cognitive load or optimizing algorithms'
                })
                
        # Memory bottleneck detection
        memory_data = self.buffer.get_recent("system", "memory_usage", 20)
        if memory_data:
            memory_values = [dp.value for dp in memory_data]
            avg_memory = np.mean(memory_values)
            if avg_memory > 0.8:  # 80% memory usage
                bottlenecks.append({
                    'type': 'memory_bottleneck',
                    'severity': 'high' if avg_memory > 0.9 else 'medium',
                    'value': float(avg_memory),
                    'description': f'High memory usage: {avg_memory:.1%}',
                    'recommendation': 'Increase memory allocation or optimize data structures'
                })
                
        # Recursive depth bottleneck
        depth_data = self.buffer.get_recent("recursive_bubble", "current_depth", 20)
        if depth_data:
            depth_values = [dp.value for dp in depth_data]
            max_depth = max(depth_values)
            if max_depth > 6:  # Approaching dangerous recursion levels
                bottlenecks.append({
                    'type': 'recursion_bottleneck',
                    'severity': 'high' if max_depth > 8 else 'medium',
                    'value': float(max_depth),
                    'description': f'Deep recursion detected: depth {max_depth}',
                    'recommendation': 'Increase max_depth limit or optimize recursive algorithms'
                })
                
        # Sigil cascade bottleneck
        cascade_time_data = self.buffer.get_recent("sigil_engine", "cascade_completion_time", 20)
        if cascade_time_data:
            times = [dp.value for dp in cascade_time_data]
            avg_time = np.mean(times)
            if avg_time > 200:  # 200ms average cascade time
                bottlenecks.append({
                    'type': 'cascade_bottleneck',
                    'severity': 'high' if avg_time > 500 else 'medium',
                    'value': float(avg_time),
                    'description': f'Slow sigil cascades: {avg_time:.1f}ms average',
                    'recommendation': 'Optimize sigil execution or reduce cascade complexity'
                })
                
        return bottlenecks
        
    def _analyze_resource_utilization(self) -> Dict[str, float]:
        """Analyze overall resource consumption patterns."""
        current_time = datetime.now()
        
        # Get system resource data
        cpu_data = self.buffer.get_recent("system", "cpu_usage", 50)
        memory_data = self.buffer.get_recent("system", "memory_usage", 50)
        disk_data = self.buffer.get_recent("system", "disk_usage", 50)
        
        cpu_utilization = 0.0
        memory_utilization = 0.0
        disk_utilization = 0.0
        
        if cpu_data:
            cpu_utilization = np.mean([dp.value for dp in cpu_data])
            
        if memory_data:
            memory_utilization = np.mean([dp.value for dp in memory_data])
            
        if disk_data:
            disk_utilization = np.mean([dp.value for dp in disk_data])
            
        # Calculate resource efficiency (lower is better for utilization)
        resource_efficiency = 1.0 - np.mean([cpu_utilization, memory_utilization, disk_utilization])
        resource_efficiency = max(0.0, resource_efficiency)
        
        return {
            'cpu_utilization': float(cpu_utilization),
            'memory_utilization': float(memory_utilization),
            'disk_utilization': float(disk_utilization),
            'resource_efficiency': float(resource_efficiency),
            'balanced_load': float(1.0 - np.std([cpu_utilization, memory_utilization, disk_utilization])),
            'peak_usage': float(max(cpu_utilization, memory_utilization, disk_utilization))
        }
        
    def _analyze_cognitive_load_distribution(self) -> Dict[str, float]:
        """Analyze how cognitive load is distributed across operations."""
        # Get operation timing data
        operations = ['recursive_reflection', 'sigil_execution', 'memory_rebloom', 'owl_observation']
        load_distribution = {}
        
        total_time = 0.0
        for operation in operations:
            op_data = self.buffer.get_recent("operations", f"{operation}_time", 20)
            if op_data:
                op_time = np.mean([dp.value for dp in op_data])
                load_distribution[operation] = op_time
                total_time += op_time
            else:
                load_distribution[operation] = 0.0
                
        # Normalize to percentages
        if total_time > 0:
            for operation in operations:
                load_distribution[operation] = load_distribution[operation] / total_time
        else:
            # Default equal distribution if no data
            for operation in operations:
                load_distribution[operation] = 0.25
                
        return load_distribution
        
    def _calculate_overall_health_score(self, tick_rate: Dict[str, float], 
                                      memory: Dict[str, float], 
                                      sigil: Dict[str, float],
                                      recursive: Dict[str, float], 
                                      resource: Dict[str, float]) -> float:
        """Calculate comprehensive system health score."""
        # Weight different aspects
        weights = {
            'tick_rate': 0.25,
            'memory': 0.20,
            'sigil': 0.20,
            'recursive': 0.20,
            'resource': 0.15
        }
        
        scores = {
            'tick_rate': tick_rate.get('efficiency', 0.0),
            'memory': memory.get('memory_efficiency', 0.0),
            'sigil': sigil.get('completion_efficiency', 0.0),
            'recursive': recursive.get('recursive_health', 0.0),
            'resource': resource.get('resource_efficiency', 0.0)
        }
        
        # Calculate weighted average
        health_score = sum(scores[aspect] * weights[aspect] for aspect in weights.keys())
        
        return min(1.0, max(0.0, health_score))

class PredictiveAnalyticsEngine:
    """Predictive analytics for maintenance and capacity planning."""
    
    def __init__(self, telemetry_buffer: TelemetryBuffer, performance_analyzer: CognitivePerformanceAnalyzer):
        self.buffer = telemetry_buffer
        self.performance_analyzer = performance_analyzer
        self.prediction_models = {}
        self.prediction_history = deque(maxlen=1000)
        self.model_accuracy = defaultdict(list)
        
    def predict_maintenance_needs(self, horizon_hours: int = 24) -> PredictiveAnalysis:
        """Predict when system will need maintenance."""
        prediction_id = str(uuid.uuid4())
        current_time = datetime.now()
        horizon = timedelta(hours=horizon_hours)
        
        # Analyze trends in key metrics
        metrics_to_predict = [
            'memory_usage_trend',
            'cpu_degradation_rate',
            'recursive_depth_growth',
            'cascade_failure_rate'
        ]
        
        predicted_values = {}
        confidence_intervals = {}
        triggering_conditions = []
        recommended_actions = []
        
        # Memory usage prediction
        memory_data = self.buffer.get_recent("system", "memory_usage", 200)
        if len(memory_data) > 10:
            memory_values = [dp.value for dp in memory_data]
            memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
            
            # Predict memory usage in horizon
            current_memory = memory_values[-1]
            predicted_memory = current_memory + (memory_trend * horizon_hours * 60)  # Per minute trend
            predicted_values['memory_usage'] = min(1.0, max(0.0, predicted_memory))
            confidence_intervals['memory_usage'] = (
                max(0.0, predicted_memory - 0.1),
                min(1.0, predicted_memory + 0.1)
            )
            
            if predicted_memory > 0.85:
                triggering_conditions.append(f"Memory usage predicted to reach {predicted_memory:.1%}")
                recommended_actions.append("Schedule memory cleanup or increase allocation")
                
        # CPU degradation prediction
        cpu_data = self.buffer.get_recent("system", "cpu_usage", 200)
        if len(cpu_data) > 10:
            cpu_values = [dp.value for dp in cpu_data]
            cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
            
            current_cpu = cpu_values[-1]
            predicted_cpu = current_cpu + (cpu_trend * horizon_hours * 60)
            predicted_values['cpu_usage'] = min(1.0, max(0.0, predicted_cpu))
            confidence_intervals['cpu_usage'] = (
                max(0.0, predicted_cpu - 0.15),
                min(1.0, predicted_cpu + 0.15)
            )
            
            if predicted_cpu > 0.8:
                triggering_conditions.append(f"CPU usage predicted to reach {predicted_cpu:.1%}")
                recommended_actions.append("Optimize algorithms or scale compute resources")
                
        # Recursive depth growth prediction
        depth_data = self.buffer.get_recent("recursive_bubble", "max_depth_reached", 100)
        if len(depth_data) > 5:
            depth_values = [dp.value for dp in depth_data]
            depth_trend = np.polyfit(range(len(depth_values)), depth_values, 1)[0]
            
            current_max_depth = max(depth_values[-10:]) if len(depth_values) >= 10 else max(depth_values)
            predicted_max_depth = current_max_depth + (depth_trend * horizon_hours)
            predicted_values['max_recursive_depth'] = max(0.0, predicted_max_depth)
            confidence_intervals['max_recursive_depth'] = (
                max(0.0, predicted_max_depth - 2),
                predicted_max_depth + 2
            )
            
            if predicted_max_depth > 8:
                triggering_conditions.append(f"Recursive depth predicted to reach {predicted_max_depth:.1f}")
                recommended_actions.append("Increase max_depth limit or optimize recursive patterns")
                
        # Overall system health prediction
        health_history = [m.overall_health_score for m in self.performance_analyzer.performance_history]
        if len(health_history) > 5:
            health_trend = np.polyfit(range(len(health_history)), health_history, 1)[0]
            current_health = health_history[-1]
            predicted_health = current_health + (health_trend * horizon_hours / 24)  # Daily trend
            predicted_values['system_health'] = min(1.0, max(0.0, predicted_health))
            confidence_intervals['system_health'] = (
                max(0.0, predicted_health - 0.1),
                min(1.0, predicted_health + 0.1)
            )
            
            if predicted_health < 0.7:
                triggering_conditions.append(f"System health predicted to drop to {predicted_health:.1%}")
                recommended_actions.append("Schedule comprehensive system maintenance")
                
        # Calculate prediction accuracy if we have historical data
        accuracy_score = self._calculate_prediction_accuracy(prediction_id, predicted_values)
        
        prediction = PredictiveAnalysis(
            prediction_id=prediction_id,
            timestamp=current_time,
            prediction_horizon=horizon,
            prediction_type="maintenance_needs",
            predicted_values=predicted_values,
            confidence_intervals=confidence_intervals,
            triggering_conditions=triggering_conditions,
            recommended_actions=recommended_actions,
            accuracy_score=accuracy_score
        )
        
        self.prediction_history.append(prediction)
        
        return prediction
        
    def forecast_resource_usage(self, horizon_hours: int = 168) -> PredictiveAnalysis:
        """Forecast resource usage for capacity planning (default: 1 week)."""
        prediction_id = str(uuid.uuid4())
        current_time = datetime.now()
        horizon = timedelta(hours=horizon_hours)
        
        # Analyze resource usage patterns
        predicted_values = {}
        confidence_intervals = {}
        triggering_conditions = []
        recommended_actions = []
        
        # Analyze different time patterns (hourly, daily, weekly)
        for resource in ['cpu_usage', 'memory_usage', 'disk_usage']:
            resource_data = self.buffer.get_recent("system", resource, min(1000, horizon_hours * 10))
            
            if len(resource_data) > 20:
                values = [dp.value for dp in resource_data]
                timestamps = [dp.timestamp for dp in resource_data]
                
                # Detect patterns
                hourly_pattern = self._detect_hourly_pattern(values, timestamps)
                daily_pattern = self._detect_daily_pattern(values, timestamps)
                trend = np.polyfit(range(len(values)), values, 1)[0]
                
                # Forecast based on patterns and trend
                current_value = values[-1]
                seasonal_adjustment = self._calculate_seasonal_adjustment(
                    current_time, horizon, hourly_pattern, daily_pattern
                )
                trend_adjustment = trend * horizon_hours
                
                predicted_value = current_value + trend_adjustment + seasonal_adjustment
                predicted_value = min(1.0, max(0.0, predicted_value))
                
                predicted_values[resource] = predicted_value
                
                # Calculate confidence interval based on historical variance
                variance = np.var(values[-50:]) if len(values) >= 50 else np.var(values)
                std_dev = np.sqrt(variance)
                confidence_intervals[resource] = (
                    max(0.0, predicted_value - 1.96 * std_dev),
                    min(1.0, predicted_value + 1.96 * std_dev)
                )
                
                # Generate capacity planning recommendations
                if predicted_value > 0.8:
                    triggering_conditions.append(f"{resource} predicted to reach {predicted_value:.1%}")
                    if resource == 'cpu_usage':
                        recommended_actions.append("Consider scaling CPU resources or optimizing algorithms")
                    elif resource == 'memory_usage':
                        recommended_actions.append("Plan memory allocation increase or optimize data structures")
                    elif resource == 'disk_usage':
                        recommended_actions.append("Schedule disk cleanup or increase storage capacity")
                        
        prediction = PredictiveAnalysis(
            prediction_id=prediction_id,
            timestamp=current_time,
            prediction_horizon=horizon,
            prediction_type="resource_forecast",
            predicted_values=predicted_values,
            confidence_intervals=confidence_intervals,
            triggering_conditions=triggering_conditions,
            recommended_actions=recommended_actions
        )
        
        self.prediction_history.append(prediction)
        
        return prediction
        
    def detect_optimal_configuration(self) -> Dict[str, Any]:
        """Identify optimal configuration parameters based on performance data."""
        recommendations = {}
        
        # Analyze recursive bubble configuration
        depth_data = self.buffer.get_recent("recursive_bubble", "current_depth", 200)
        max_depth_data = self.buffer.get_recent("recursive_bubble", "max_depth_reached", 100)
        stabilization_data = self.buffer.get_recent("recursive_bubble", "stabilization_count", 100)
        
        if depth_data and max_depth_data:
            depths = [dp.value for dp in depth_data]
            max_depths = [dp.value for dp in max_depth_data]
            
            # Find optimal max_depth
            depth_utilization = np.percentile(max_depths, 95)  # 95th percentile usage
            current_limit = max(max_depths) if max_depths else 5
            
            if depth_utilization > current_limit * 0.8:
                recommended_max_depth = int(depth_utilization * 1.2)  # 20% headroom
                recommendations['recursive_bubble.max_depth'] = {
                    'current': current_limit,
                    'recommended': recommended_max_depth,
                    'reasoning': f'95% of episodes reach depth {depth_utilization:.1f}, current limit too restrictive',
                    'expected_improvement': '15-20% reduction in recursive stabilization overhead'
                }
                
        # Analyze tick engine configuration
        tick_data = self.buffer.get_recent("tick_engine", "tick_rate", 200)
        if tick_data:
            tick_rates = [dp.value for dp in tick_data]
            avg_tick_rate = np.mean(tick_rates)
            target_rate = 10.0
            
            if avg_tick_rate < target_rate * 0.9:
                recommendations['tick_engine.target_rate'] = {
                    'current': target_rate,
                    'recommended': target_rate * 0.8,
                    'reasoning': f'Average tick rate {avg_tick_rate:.1f} below target, reduce target to match capacity',
                    'expected_improvement': 'Reduced CPU overhead and improved stability'
                }
                
        # Analyze memory configuration
        memory_data = self.buffer.get_recent("system", "memory_usage", 200)
        rebloom_data = self.buffer.get_recent("memory", "rebloom_success_rate", 100)
        
        if memory_data and rebloom_data:
            memory_usage = np.mean([dp.value for dp in memory_data])
            rebloom_rate = np.mean([dp.value for dp in rebloom_data])
            
            if memory_usage > 0.8 and rebloom_rate < 0.9:
                recommendations['memory.allocation_size'] = {
                    'current': 'auto',
                    'recommended': 'increase by 25%',
                    'reasoning': f'High memory usage ({memory_usage:.1%}) correlates with low rebloom success ({rebloom_rate:.1%})',
                    'expected_improvement': 'Improved memory efficiency and rebloom success rates'
                }
                
        return recommendations
        
    def _detect_hourly_pattern(self, values: List[float], timestamps: List[datetime]) -> Dict[int, float]:
        """Detect hourly usage patterns."""
        hourly_data = defaultdict(list)
        
        for value, timestamp in zip(values, timestamps):
            hour = timestamp.hour
            hourly_data[hour].append(value)
            
        # Calculate average for each hour
        hourly_pattern = {}
        for hour in range(24):
            if hour in hourly_data and hourly_data[hour]:
                hourly_pattern[hour] = np.mean(hourly_data[hour])
            else:
                # Interpolate missing hours
                hourly_pattern[hour] = np.mean(values) if values else 0.0
                
        return hourly_pattern
        
    def _detect_daily_pattern(self, values: List[float], timestamps: List[datetime]) -> Dict[int, float]:
        """Detect daily usage patterns (day of week)."""
        daily_data = defaultdict(list)
        
        for value, timestamp in zip(values, timestamps):
            day = timestamp.weekday()  # 0=Monday, 6=Sunday
            daily_data[day].append(value)
            
        # Calculate average for each day
        daily_pattern = {}
        for day in range(7):
            if day in daily_data and daily_data[day]:
                daily_pattern[day] = np.mean(daily_data[day])
            else:
                daily_pattern[day] = np.mean(values) if values else 0.0
                
        return daily_pattern
        
    def _calculate_seasonal_adjustment(self, current_time: datetime, horizon: timedelta,
                                     hourly_pattern: Dict[int, float], 
                                     daily_pattern: Dict[int, float]) -> float:
        """Calculate seasonal adjustment for predictions."""
        # Future time
        future_time = current_time + horizon
        
        # Current patterns
        current_hour_avg = hourly_pattern.get(current_time.hour, 0.0)
        current_day_avg = daily_pattern.get(current_time.weekday(), 0.0)
        
        # Future patterns
        future_hour_avg = hourly_pattern.get(future_time.hour, 0.0)
        future_day_avg = daily_pattern.get(future_time.weekday(), 0.0)
        
        # Calculate adjustment (weighted combination)
        hour_adjustment = (future_hour_avg - current_hour_avg) * 0.3
        day_adjustment = (future_day_avg - current_day_avg) * 0.2
        
        return hour_adjustment + day_adjustment
        
    def _calculate_prediction_accuracy(self, prediction_id: str, predicted_values: Dict[str, float]) -> Optional[float]:
        """Calculate accuracy of previous predictions."""
        # This would compare with actual observed values
        # For now, return a placeholder accuracy score
        return 0.85  # 85% accuracy placeholder

class InsightsGenerator:
    """Generates automated insights and optimization recommendations."""
    
    def __init__(self, performance_analyzer: CognitivePerformanceAnalyzer, 
                 predictive_engine: PredictiveAnalyticsEngine):
        self.performance_analyzer = performance_analyzer
        self.predictive_engine = predictive_engine
        self.insights_history = deque(maxlen=1000)
        self.insight_patterns = defaultdict(int)
        
    def generate_insights(self) -> List[AnalyticalInsight]:
        """Generate comprehensive analytical insights."""
        insights = []
        current_time = datetime.now()
        
        # Get current performance metrics
        performance = self.performance_analyzer.analyze_cognitive_performance()
        
        # Generate performance optimization insights
        insights.extend(self._generate_performance_insights(performance, current_time))
        
        # Generate resource insights
        insights.extend(self._generate_resource_insights(performance, current_time))
        
        # Generate predictive insights
        insights.extend(self._generate_predictive_insights(current_time))
        
        # Generate configuration insights
        insights.extend(self._generate_configuration_insights(current_time))
        
        # Store insights
        for insight in insights:
            self.insights_history.append(insight)
            self.insight_patterns[insight.insight_type.value] += 1
            
        return insights
        
    def _generate_performance_insights(self, performance: PerformanceMetrics, 
                                     current_time: datetime) -> List[AnalyticalInsight]:
        """Generate performance-related insights."""
        insights = []
        
        # Tick rate optimization
        if performance.tick_rate_trend['efficiency'] < 0.8:
            insight = AnalyticalInsight(
                insight_id=str(uuid.uuid4()),
                insight_type=InsightType.PERFORMANCE_OPTIMIZATION,
                timestamp=current_time,
                confidence=0.85,
                recommendation="Optimize tick engine performance or reduce target tick rate",
                reasoning=f"Tick efficiency at {performance.tick_rate_trend['efficiency']:.1%}, below optimal range",
                expected_improvement="10-15% improvement in system responsiveness",
                risk_level=RiskLevel.LOW,
                affected_systems=["tick_engine"],
                implementation_priority=2,
                validation_metrics=["tick_rate", "system_responsiveness"],
                estimated_impact={'performance': 0.15, 'stability': 0.05}
            )
            insights.append(insight)
            
        # Memory efficiency optimization
        if performance.memory_efficiency['rebloom_success_rate'] < 0.9:
            insight = AnalyticalInsight(
                insight_id=str(uuid.uuid4()),
                insight_type=InsightType.PERFORMANCE_OPTIMIZATION,
                timestamp=current_time,
                confidence=0.78,
                recommendation="Optimize memory allocation and rebloom algorithms",
                reasoning=f"Rebloom success rate at {performance.memory_efficiency['rebloom_success_rate']:.1%}, indicating memory pressure",
                expected_improvement="20-25% improvement in memory operations",
                risk_level=RiskLevel.MEDIUM,
                affected_systems=["memory_system", "rebloom_engine"],
                implementation_priority=1,
                validation_metrics=["rebloom_success_rate", "memory_efficiency"],
                estimated_impact={'performance': 0.25, 'reliability': 0.20}
            )
            insights.append(insight)
            
        # Recursive optimization
        if performance.recursive_stability['recursive_health'] < 0.8:
            insight = AnalyticalInsight(
                insight_id=str(uuid.uuid4()),
                insight_type=InsightType.PERFORMANCE_OPTIMIZATION,
                timestamp=current_time,
                confidence=0.82,
                recommendation="Increase recursive_bubble.max_depth or optimize recursive patterns",
                reasoning=f"Recursive health at {performance.recursive_stability['recursive_health']:.1%}, frequent depth limiting detected",
                expected_improvement="15-20% reduction in recursive overhead",
                risk_level=RiskLevel.LOW,
                affected_systems=["recursive_bubble"],
                implementation_priority=3,
                validation_metrics=["recursive_depth_efficiency", "stabilization_rate"],
                estimated_impact={'performance': 0.18, 'cognitive_capacity': 0.15}
            )
            insights.append(insight)
            
        return insights
        
    def _generate_resource_insights(self, performance: PerformanceMetrics, 
                                  current_time: datetime) -> List[AnalyticalInsight]:
        """Generate resource-related insights."""
        insights = []
        
        # High CPU usage alert
        if performance.resource_utilization['cpu_utilization'] > 0.85:
            insight = AnalyticalInsight(
                insight_id=str(uuid.uuid4()),
                insight_type=InsightType.RESOURCE_ALERT,
                timestamp=current_time,
                confidence=0.95,
                recommendation="Scale CPU resources or optimize compute-intensive operations",
                reasoning=f"CPU utilization at {performance.resource_utilization['cpu_utilization']:.1%}, approaching capacity limits",
                expected_improvement="Prevent performance degradation and system instability",
                risk_level=RiskLevel.HIGH,
                affected_systems=["system_resources"],
                implementation_priority=1,
                validation_metrics=["cpu_usage", "system_responsiveness"],
                estimated_impact={'stability': 0.30, 'performance': 0.25}
            )
            insights.append(insight)
            
        # Memory usage alert
        if performance.resource_utilization['memory_utilization'] > 0.8:
            insight = AnalyticalInsight(
                insight_id=str(uuid.uuid4()),
                insight_type=InsightType.RESOURCE_ALERT,
                timestamp=current_time,
                confidence=0.90,
                recommendation="Increase memory allocation or implement memory optimization",
                reasoning=f"Memory utilization at {performance.resource_utilization['memory_utilization']:.1%}, high memory pressure detected",
                expected_improvement="Prevent memory-related failures and improve performance",
                risk_level=RiskLevel.HIGH,
                affected_systems=["memory_system"],
                implementation_priority=1,
                validation_metrics=["memory_usage", "rebloom_success_rate"],
                estimated_impact={'reliability': 0.35, 'performance': 0.20}
            )
            insights.append(insight)
            
        return insights
        
    def _generate_predictive_insights(self, current_time: datetime) -> List[AnalyticalInsight]:
        """Generate predictive maintenance insights."""
        insights = []
        
        # Get predictive analysis
        maintenance_prediction = self.predictive_engine.predict_maintenance_needs(24)
        
        if maintenance_prediction.triggering_conditions:
            insight = AnalyticalInsight(
                insight_id=str(uuid.uuid4()),
                insight_type=InsightType.PREDICTIVE_MAINTENANCE,
                timestamp=current_time,
                confidence=0.75,
                recommendation=f"Schedule maintenance within 24 hours: {', '.join(maintenance_prediction.recommended_actions)}",
                reasoning=f"Predictive analysis indicates: {', '.join(maintenance_prediction.triggering_conditions)}",
                expected_improvement="Prevent system degradation and maintain optimal performance",
                risk_level=RiskLevel.MEDIUM,
                affected_systems=["system_maintenance"],
                implementation_priority=2,
                validation_metrics=["system_health", "predicted_metrics"],
                estimated_impact={'reliability': 0.40, 'uptime': 0.20}
            )
            insights.append(insight)
            
        return insights
        
    def _generate_configuration_insights(self, current_time: datetime) -> List[AnalyticalInsight]:
        """Generate configuration optimization insights."""
        insights = []
        
        # Get optimal configuration recommendations
        config_recommendations = self.predictive_engine.detect_optimal_configuration()
        
        for param, recommendation in config_recommendations.items():
            insight = AnalyticalInsight(
                insight_id=str(uuid.uuid4()),
                insight_type=InsightType.CONFIGURATION_TUNING,
                timestamp=current_time,
                confidence=0.80,
                recommendation=f"Update {param} from {recommendation['current']} to {recommendation['recommended']}",
                reasoning=recommendation['reasoning'],
                expected_improvement=recommendation['expected_improvement'],
                risk_level=RiskLevel.LOW,
                affected_systems=[param.split('.')[0]],
                implementation_priority=3,
                validation_metrics=[param.replace('.', '_'), 'system_performance'],
                estimated_impact={'performance': 0.15, 'efficiency': 0.20}
            )
            insights.append(insight)
            
        return insights

class TelemetryAnalytics:
    """
    Main telemetry analytics engine that processes streaming telemetry data
    and generates real-time insights about DAWN's operational performance.
    """
    
    def __init__(self, buffer_size: int = 10000, analysis_interval: float = 30.0):
        """
        Initialize the telemetry analytics engine.
        
        Args:
            buffer_size: Maximum number of telemetry points to keep in memory
            analysis_interval: Seconds between analysis cycles
        """
        self.analytics_id = str(uuid.uuid4())
        self.creation_time = time.time()
        
        # Core components
        self.telemetry_buffer = TelemetryBuffer(buffer_size)
        self.performance_analyzer = CognitivePerformanceAnalyzer(self.telemetry_buffer)
        self.predictive_engine = PredictiveAnalyticsEngine(self.telemetry_buffer, self.performance_analyzer)
        self.insights_generator = InsightsGenerator(self.performance_analyzer, self.predictive_engine)
        
        # Configuration
        self.analysis_interval = analysis_interval
        self.running = False
        self.analysis_thread = None
        
        # Data streams and processors
        self.data_streams = {}
        self.stream_processors = {}
        self.analysis_callbacks = []
        
        # Analytics state
        self.latest_performance = None
        self.latest_insights = []
        self.analytics_history = deque(maxlen=1000)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="telemetry_analytics")
        
        # Performance tracking
        self.metrics = {
            'data_points_processed': 0,
            'analyses_performed': 0,
            'insights_generated': 0,
            'predictions_made': 0,
            'processing_errors': 0,
            'avg_processing_time': 0.0
        }
        
        logger.info(f"📊 TelemetryAnalytics initialized: {self.analytics_id}")
        logger.info(f"   🔄 Analysis interval: {analysis_interval}s")
        logger.info(f"   💾 Buffer size: {buffer_size}")
        
    def start_analytics(self):
        """Start the analytics processing engine."""
        if self.running:
            return
            
        self.running = True
        self.analysis_thread = threading.Thread(
            target=self._analytics_loop,
            name="telemetry_analytics",
            daemon=True
        )
        self.analysis_thread.start()
        
        logger.info("📊 Telemetry analytics engine started")
        
    def stop_analytics(self):
        """Stop the analytics processing engine."""
        self.running = False
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=5.0)
            
        self.executor.shutdown(wait=True)
        
        logger.info("📊 Telemetry analytics engine stopped")
        
    def ingest_telemetry(self, source: str, metric_name: str, 
                        value: Union[int, float, str, bool],
                        tags: Optional[Dict[str, str]] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        """
        Ingest a telemetry data point for analysis.
        
        Args:
            source: Source system or component
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for categorization
            metadata: Optional additional metadata
        """
        data_point = TelemetryDataPoint(
            timestamp=datetime.now(),
            source=source,
            metric_name=metric_name,
            value=value,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.telemetry_buffer.add(data_point)
        self.metrics['data_points_processed'] += 1
        
        # Process real-time alerts if needed
        self._check_real_time_alerts(data_point)
        
    def register_data_stream(self, stream_name: str, processor: Callable):
        """Register a data stream processor for specific telemetry types."""
        self.stream_processors[stream_name] = processor
        logger.info(f"📡 Registered data stream processor: {stream_name}")
        
    def register_analysis_callback(self, callback: Callable):
        """Register a callback for analysis results."""
        self.analysis_callbacks.append(callback)
        
    def get_latest_performance(self) -> Optional[PerformanceMetrics]:
        """Get the latest performance analysis results."""
        return self.latest_performance
        
    def get_latest_insights(self) -> List[AnalyticalInsight]:
        """Get the latest analytical insights."""
        return self.latest_insights.copy()
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Prepare real-time data for dashboard display."""
        current_time = datetime.now()
        
        # Get latest performance metrics
        performance = self.latest_performance
        
        dashboard_data = {
            'timestamp': current_time.isoformat(),
            'system_health': {
                'overall_score': performance.overall_health_score if performance else 0.0,
                'status': self._get_health_status(performance.overall_health_score if performance else 0.0)
            },
            'performance_metrics': {},
            'resource_utilization': {},
            'active_insights': [],
            'trend_data': {},
            'alerts': []
        }
        
        if performance:
            # Performance metrics
            dashboard_data['performance_metrics'] = {
                'tick_rate': performance.tick_rate_trend,
                'memory_efficiency': performance.memory_efficiency,
                'sigil_efficiency': performance.sigil_cascade_efficiency,
                'recursive_stability': performance.recursive_stability
            }
            
            # Resource utilization
            dashboard_data['resource_utilization'] = performance.resource_utilization
            
            # Cognitive load distribution
            dashboard_data['cognitive_load'] = performance.cognitive_load_distribution
            
            # Bottlenecks as alerts
            for bottleneck in performance.bottleneck_identification:
                alert = {
                    'type': 'bottleneck',
                    'severity': bottleneck['severity'],
                    'message': bottleneck['description'],
                    'recommendation': bottleneck['recommendation']
                }
                dashboard_data['alerts'].append(alert)
                
        # Active insights
        for insight in self.latest_insights:
            insight_data = {
                'type': insight.insight_type.value,
                'confidence': insight.confidence,
                'recommendation': insight.recommendation,
                'priority': insight.implementation_priority,
                'risk_level': insight.risk_level.value
            }
            dashboard_data['active_insights'].append(insight_data)
            
        # Historical trend data
        if len(self.analytics_history) > 0:
            # Last 24 data points for trending
            recent_analytics = list(self.analytics_history)[-24:]
            
            dashboard_data['trend_data'] = {
                'timestamps': [a['timestamp'].isoformat() for a in recent_analytics],
                'health_scores': [a['performance']['overall_health_score'] for a in recent_analytics if 'performance' in a],
                'cpu_usage': [a['performance']['resource_utilization']['cpu_utilization'] for a in recent_analytics if 'performance' in a],
                'memory_usage': [a['performance']['resource_utilization']['memory_utilization'] for a in recent_analytics if 'performance' in a]
            }
            
        return dashboard_data
        
    def export_analytics_report(self, start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Export comprehensive analytics report for the specified time range."""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.now()
            
        # Get telemetry data for the time range
        telemetry_data = self.telemetry_buffer.get_time_range(start_time, end_time)
        
        # Aggregate analytics for the period
        period_analytics = [a for a in self.analytics_history 
                          if start_time <= a['timestamp'] <= end_time]
        
        report = {
            'report_id': str(uuid.uuid4()),
            'generated_at': datetime.now().isoformat(),
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration_hours': (end_time - start_time).total_seconds() / 3600
            },
            'summary': {
                'data_points': len(telemetry_data),
                'analyses_performed': len(period_analytics),
                'insights_generated': sum(len(a.get('insights', [])) for a in period_analytics),
                'avg_health_score': np.mean([a['performance']['overall_health_score'] 
                                           for a in period_analytics if 'performance' in a]) if period_analytics else 0.0
            },
            'performance_trends': {},
            'key_insights': [],
            'recommendations': [],
            'system_events': []
        }
        
        # Analyze performance trends
        if period_analytics:
            health_scores = [a['performance']['overall_health_score'] for a in period_analytics if 'performance' in a]
            if health_scores:
                report['performance_trends'] = {
                    'health_score_trend': np.polyfit(range(len(health_scores)), health_scores, 1)[0],
                    'min_health': min(health_scores),
                    'max_health': max(health_scores),
                    'avg_health': np.mean(health_scores),
                    'health_stability': 1.0 - (np.std(health_scores) / np.mean(health_scores)) if np.mean(health_scores) > 0 else 0.0
                }
                
        # Collect insights and recommendations
        all_insights = []
        for analytics in period_analytics:
            if 'insights' in analytics:
                all_insights.extend(analytics['insights'])
                
        # Group insights by type
        insight_groups = defaultdict(list)
        for insight in all_insights:
            insight_groups[insight['insight_type']].append(insight)
            
        report['key_insights'] = [
            {
                'type': insight_type,
                'count': len(insights),
                'avg_confidence': np.mean([i['confidence'] for i in insights]),
                'high_priority_count': len([i for i in insights if i['implementation_priority'] <= 2])
            }
            for insight_type, insights in insight_groups.items()
        ]
        
        # Top recommendations
        all_recommendations = [insight for insight in all_insights if insight['implementation_priority'] <= 2]
        all_recommendations.sort(key=lambda x: (x['implementation_priority'], -x['confidence']))
        
        report['recommendations'] = [
            {
                'recommendation': rec['recommendation'],
                'confidence': rec['confidence'],
                'priority': rec['implementation_priority'],
                'expected_improvement': rec['expected_improvement'],
                'risk_level': rec['risk_level']
            }
            for rec in all_recommendations[:10]  # Top 10 recommendations
        ]
        
        return report
        
    def _analytics_loop(self):
        """Main analytics processing loop."""
        logger.info("📊 Analytics processing loop started")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Perform analysis
                performance = self.performance_analyzer.analyze_cognitive_performance()
                insights = self.insights_generator.generate_insights()
                
                # Update latest results
                self.latest_performance = performance
                self.latest_insights = insights
                
                # Store in history
                analytics_record = {
                    'timestamp': datetime.now(),
                    'performance': asdict(performance),
                    'insights': [asdict(insight) for insight in insights]
                }
                self.analytics_history.append(analytics_record)
                
                # Update metrics
                self.metrics['analyses_performed'] += 1
                self.metrics['insights_generated'] += len(insights)
                
                processing_time = time.time() - start_time
                self.metrics['avg_processing_time'] = (
                    (self.metrics['avg_processing_time'] * (self.metrics['analyses_performed'] - 1) + processing_time) /
                    self.metrics['analyses_performed']
                )
                
                # Notify callbacks
                for callback in self.analysis_callbacks:
                    try:
                        self.executor.submit(callback, performance, insights)
                    except Exception as e:
                        logger.warning(f"Analysis callback error: {e}")
                        
                # Sleep until next analysis
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"Analytics loop error: {e}")
                self.metrics['processing_errors'] += 1
                time.sleep(self.analysis_interval)
                
        logger.info("📊 Analytics processing loop stopped")
        
    def _check_real_time_alerts(self, data_point: TelemetryDataPoint):
        """Check for real-time alert conditions."""
        # Critical CPU usage
        if (data_point.source == "system" and data_point.metric_name == "cpu_usage" and 
            isinstance(data_point.value, (int, float)) and data_point.value > 0.95):
            logger.warning(f"🚨 Critical CPU usage alert: {data_point.value:.1%}")
            
        # Critical memory usage
        if (data_point.source == "system" and data_point.metric_name == "memory_usage" and 
            isinstance(data_point.value, (int, float)) and data_point.value > 0.95):
            logger.warning(f"🚨 Critical memory usage alert: {data_point.value:.1%}")
            
        # Deep recursion alert
        if (data_point.source == "recursive_bubble" and data_point.metric_name == "current_depth" and 
            isinstance(data_point.value, (int, float)) and data_point.value > 8):
            logger.warning(f"🚨 Deep recursion alert: depth {data_point.value}")
            
    def _get_health_status(self, health_score: float) -> str:
        """Convert health score to status string."""
        if health_score >= 0.9:
            return "excellent"
        elif health_score >= 0.8:
            return "good"
        elif health_score >= 0.7:
            return "fair"
        elif health_score >= 0.5:
            return "poor"
        else:
            return "critical"
            
    def get_analytics_status(self) -> Dict[str, Any]:
        """Get current analytics engine status."""
        return {
            'analytics_id': self.analytics_id,
            'running': self.running,
            'uptime_seconds': time.time() - self.creation_time,
            'buffer_utilization': len(self.telemetry_buffer.buffer) / self.telemetry_buffer.max_size,
            'metrics_tracked': len(self.telemetry_buffer.get_metrics_summary()),
            'latest_health_score': self.latest_performance.overall_health_score if self.latest_performance else 0.0,
            'active_insights': len(self.latest_insights),
            'performance_metrics': dict(self.metrics)
        }


# Global instance for easy access
_global_analytics = None
_analytics_lock = threading.Lock()

def get_telemetry_analytics(auto_start: bool = True) -> TelemetryAnalytics:
    """Get the global telemetry analytics instance."""
    global _global_analytics
    
    with _analytics_lock:
        if _global_analytics is None:
            _global_analytics = TelemetryAnalytics()
            if auto_start:
                _global_analytics.start_analytics()
                
    return _global_analytics

def ingest_telemetry_data(source: str, metric_name: str, value: Union[int, float, str, bool],
                         tags: Optional[Dict[str, str]] = None,
                         metadata: Optional[Dict[str, Any]] = None):
    """Convenience function to ingest telemetry data."""
    analytics = get_telemetry_analytics()
    analytics.ingest_telemetry(source, metric_name, value, tags, metadata)


if __name__ == "__main__":
    # Demo/test the telemetry analytics system
    logging.basicConfig(level=logging.INFO)
    
    print("📊 Initializing DAWN Telemetry Analytics System...")
    
    analytics = TelemetryAnalytics(analysis_interval=5.0)  # 5 second intervals for demo
    analytics.start_analytics()
    
    print("\n📡 Simulating telemetry data...")
    
    # Simulate various telemetry data
    import random
    
    for i in range(50):
        # System metrics
        analytics.ingest_telemetry("system", "cpu_usage", random.uniform(0.3, 0.9))
        analytics.ingest_telemetry("system", "memory_usage", random.uniform(0.4, 0.8))
        analytics.ingest_telemetry("system", "disk_usage", random.uniform(0.2, 0.6))
        
        # Cognitive metrics
        analytics.ingest_telemetry("tick_engine", "tick_rate", random.uniform(8.0, 12.0))
        analytics.ingest_telemetry("recursive_bubble", "current_depth", random.randint(1, 6))
        analytics.ingest_telemetry("recursive_bubble", "max_depth_reached", random.randint(3, 8))
        analytics.ingest_telemetry("sigil_engine", "cascade_completion_time", random.uniform(50, 300))
        analytics.ingest_telemetry("memory", "rebloom_success_rate", random.uniform(0.7, 1.0))
        
        time.sleep(0.1)
        
    print("\n⏳ Waiting for analysis...")
    time.sleep(8)
    
    # Get results
    performance = analytics.get_latest_performance()
    insights = analytics.get_latest_insights()
    dashboard_data = analytics.get_dashboard_data()
    
    print(f"\n📊 Analysis Results:")
    if performance:
        print(f"   Overall Health Score: {performance.overall_health_score:.3f}")
        print(f"   Tick Rate Efficiency: {performance.tick_rate_trend['efficiency']:.3f}")
        print(f"   Memory Efficiency: {performance.memory_efficiency['memory_efficiency']:.3f}")
        print(f"   Recursive Health: {performance.recursive_stability['recursive_health']:.3f}")
        print(f"   Bottlenecks Detected: {len(performance.bottleneck_identification)}")
        
    print(f"\n💡 Generated Insights: {len(insights)}")
    for insight in insights[:3]:  # Show first 3 insights
        print(f"   🔍 {insight.insight_type.value}: {insight.recommendation[:60]}...")
        print(f"      Confidence: {insight.confidence:.1%}, Priority: {insight.implementation_priority}")
        
    # Show dashboard data
    print(f"\n📊 Dashboard Data:")
    print(f"   System Status: {dashboard_data['system_health']['status']}")
    print(f"   Active Alerts: {len(dashboard_data['alerts'])}")
    print(f"   CPU Usage: {dashboard_data['resource_utilization'].get('cpu_utilization', 0):.1%}")
    print(f"   Memory Usage: {dashboard_data['resource_utilization'].get('memory_utilization', 0):.1%}")
    
    # Generate report
    report = analytics.export_analytics_report()
    print(f"\n📋 Analytics Report Generated:")
    print(f"   Data Points: {report['summary']['data_points']}")
    print(f"   Analyses: {report['summary']['analyses_performed']}")
    print(f"   Insights: {report['summary']['insights_generated']}")
    print(f"   Avg Health: {report['summary']['avg_health_score']:.3f}")
    
    # Show status
    status = analytics.get_analytics_status()
    print(f"\n🔧 Analytics Engine Status:")
    print(f"   Running: {'✅' if status['running'] else '❌'}")
    print(f"   Buffer Utilization: {status['buffer_utilization']:.1%}")
    print(f"   Metrics Tracked: {status['metrics_tracked']}")
    print(f"   Data Points Processed: {status['performance_metrics']['data_points_processed']}")
    
    print(f"\n📊 Telemetry analytics demonstration complete")
    analytics.stop_analytics()
