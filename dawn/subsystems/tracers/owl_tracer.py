#!/usr/bin/env python3
"""
🦉 Owl Tracer - Long-Memory Auditor & Archivist
═══════════════════════════════════════════════════════════

The Owl preserves and audits DAWN's long-term memory - dialogue, 
internal reflections, and slow-forming context. It converts lived 
tick history into durable records and provides continuity analysis.

Based on RTF specifications from DAWN-docs/Tracers/Owl.rtf
"""

import logging
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
import os

logger = logging.getLogger(__name__)

class LogSource(Enum):
    """Sources of data for Owl auditing"""
    FRACTAL_LOG = "fractal_log"
    DIALOGUE_LOG = "dialogue_log"
    JULIET_EVENTS = "juliet_events"
    SOOT_ASH = "soot_ash"
    MYCELIAL_METRICS = "mycelial_metrics"
    SCHEMA_EVENTS = "schema_events"
    TRACER_LIFECYCLE = "tracer_lifecycle"

class AuditLevel(Enum):
    """Levels of audit scrutiny"""
    LIGHT = "light"          # Basic continuity check
    STANDARD = "standard"    # Full schema drift analysis
    DEEP = "deep"           # Comprehensive historical analysis
    EMERGENCY = "emergency"  # Crisis-level audit

class RecommendationAction(Enum):
    """Actions Owl can recommend"""
    CAP_REBLOOM_RATE = "cap_rebloom_rate"
    POLLINATE_CLUSTER = "pollinate_cluster"
    ARCHIVE_MEMORY_CLUSTER = "archive_memory_cluster"
    TRIGGER_WEAVING = "trigger_weaving"
    SPAWN_BEETLE = "spawn_beetle"
    SPAWN_BEE = "spawn_bee"
    SPAWN_WHALE = "spawn_whale"
    PURGE_SOOT_BUILDUP = "purge_soot_buildup"
    SCHEDULE_MIRRORS = "schedule_mirrors"

@dataclass
class KeyDecision:
    """A significant decision tracked by Owl"""
    decision_id: str
    tick: int
    pressure: float
    entropy: float
    outcome: str
    consequences: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RebloomEvent:
    """A rebloom event tracked for analysis"""
    memory_id: str
    depth: int
    trigger: str
    impact_score: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SchemaDrift:
    """Schema drift analysis"""
    magnitude: float           # Overall drift magnitude
    hot_edges: List[str]      # Edges with high volatility
    stability_score: float    # Overall stability
    trend_direction: float    # Positive = expanding, negative = contracting
    critical_nodes: List[str] = field(default_factory=list)

@dataclass
class ResidueBalance:
    """Ash/Soot balance analysis"""
    soot_ratio: float
    ash_ratio: float
    ash_bias: List[float]     # RGB bias in ash composition
    balance_stability: float  # Stability of the balance over time
    cleanup_urgency: float    # How urgently cleanup is needed

@dataclass
class EpochSummary:
    """Summary of a time epoch for archival"""
    tick_window: Tuple[int, int]
    narrative: str
    key_decisions: List[KeyDecision]
    reblooms: List[RebloomEvent]
    schema_drift: SchemaDrift
    residue_balance: ResidueBalance
    sources_analyzed: List[LogSource]
    coherence_trend: float
    stability_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class OwlRecommendation:
    """A recommendation from the Owl"""
    action: RecommendationAction
    reason: str
    priority: float          # 0.0 to 1.0, higher = more urgent
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8

class OwlTracer:
    """
    Long-Memory Auditor & Archivist for DAWN.
    Guards against slow drift, narrative creep, and incremental bias.
    Provides continuity across long horizons.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Configuration
        self.audit_window_size = self.config.get('audit_window_size', 1000)  # ticks
        self.epoch_size = self.config.get('epoch_size', 10000)  # ticks per epoch
        self.max_archives = self.config.get('max_archives', 100)
        self.drift_threshold = self.config.get('drift_threshold', 0.25)
        self.coherence_threshold = self.config.get('coherence_threshold', 0.7)
        
        # Storage paths
        self.archive_dir = self.config.get('archive_dir', 'data/archives/owl')
        self.dialogue_log_path = self.config.get('dialogue_log', 'data/logs/dialogue.json')
        
        # Internal state
        self.current_epoch_start = 0
        self.archived_epochs: List[EpochSummary] = []
        self.continuous_logs: Dict[LogSource, deque] = {
            source: deque(maxlen=self.audit_window_size) for source in LogSource
        }
        
        # Analytics
        self.schema_history: deque = deque(maxlen=1000)
        self.coherence_history: deque = deque(maxlen=1000)
        self.decision_registry: Dict[str, KeyDecision] = {}
        self.rebloom_tracker: deque = deque(maxlen=5000)
        
        # Performance tracking
        self.audit_stats = {
            'total_audits': 0,
            'epochs_archived': 0,
            'decisions_tracked': 0,
            'drift_alerts': 0,
            'coherence_warnings': 0,
            'recommendations_made': 0
        }
        
        # Initialize storage
        self._ensure_directories()
        
        logger.info("🦉 Owl Tracer initialized - Long-memory auditor ready")
    
    def log_event(self, source: LogSource, event_data: Dict[str, Any]) -> None:
        """
        Log an event from a specific source for later analysis.
        
        Args:
            source: The source of the event
            event_data: Event data to log
        """
        event_entry = {
            'timestamp': datetime.now().isoformat(),
            'tick': event_data.get('tick', 0),
            'data': event_data
        }
        
        self.continuous_logs[source].append(event_entry)
        
        # Real-time processing for critical events
        if source == LogSource.SCHEMA_EVENTS:
            self._process_schema_event(event_data)
        elif source == LogSource.JULIET_EVENTS:
            self._process_rebloom_event(event_data)
        
        logger.debug(f"🦉 Logged {source.value} event at tick {event_data.get('tick', 0)}")
    
    def track_decision(self, decision: KeyDecision) -> None:
        """Track a key decision for long-term analysis"""
        self.decision_registry[decision.decision_id] = decision
        self.audit_stats['decisions_tracked'] += 1
        
        logger.info(f"🦉 Tracking decision {decision.decision_id} "
                   f"(P:{decision.pressure:.2f}, E:{decision.entropy:.2f})")
    
    def perform_audit(
        self, 
        current_tick: int,
        level: AuditLevel = AuditLevel.STANDARD,
        sources: Optional[List[LogSource]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive audit of system state and history.
        
        Args:
            current_tick: Current system tick
            level: Depth of audit to perform
            sources: Specific sources to audit (default: all)
            
        Returns:
            Comprehensive audit report
        """
        start_time = time.time()
        
        logger.info(f"🦉 Starting {level.value} audit at tick {current_tick}")
        
        if sources is None:
            sources = list(LogSource)
        
        audit_result = {
            'tracer': 'Owl',
            'type': 'long_memory_audit',
            'level': level.value,
            'tick_window': [current_tick - self.audit_window_size, current_tick],
            'sources': [s.value for s in sources],
            'timestamp': datetime.now().isoformat(),
            'audit_duration': 0.0
        }
        
        # Perform analysis based on audit level
        if level in [AuditLevel.STANDARD, AuditLevel.DEEP, AuditLevel.EMERGENCY]:
            audit_result['summary'] = self._generate_comprehensive_summary(current_tick, sources)
            audit_result['recommendations'] = self._generate_recommendations(audit_result['summary'])
        
        if level in [AuditLevel.DEEP, AuditLevel.EMERGENCY]:
            audit_result['historical_analysis'] = self._perform_deep_historical_analysis(current_tick)
            audit_result['provenance_chains'] = self._trace_provenance_chains()
        
        if level == AuditLevel.EMERGENCY:
            audit_result['crisis_assessment'] = self._assess_crisis_state(current_tick)
            audit_result['emergency_interventions'] = self._recommend_emergency_interventions()
        
        # Update statistics
        audit_result['audit_duration'] = time.time() - start_time
        self.audit_stats['total_audits'] += 1
        
        # Check if epoch archival is needed
        if current_tick - self.current_epoch_start >= self.epoch_size:
            self._archive_epoch(current_tick, audit_result['summary'])
        
        logger.info(f"🦉 Audit complete in {audit_result['audit_duration']:.3f}s - "
                   f"{len(audit_result.get('recommendations', []))} recommendations")
        
        return audit_result
    
    def _generate_comprehensive_summary(
        self, 
        current_tick: int, 
        sources: List[LogSource]
    ) -> Dict[str, Any]:
        """Generate comprehensive summary of current system state"""
        
        # Analyze recent decisions
        recent_decisions = [
            d for d in self.decision_registry.values()
            if current_tick - d.tick <= self.audit_window_size
        ]
        
        # Analyze rebloom patterns
        recent_reblooms = [
            r for r in self.rebloom_tracker
            if (datetime.now() - r.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        # Calculate schema drift
        schema_drift = self._calculate_schema_drift()
        
        # Analyze residue balance
        residue_balance = self._analyze_residue_balance()
        
        # Generate narrative
        narrative = self._generate_narrative_summary(
            recent_decisions, recent_reblooms, schema_drift, residue_balance
        )
        
        summary = {
            'narrative': narrative,
            'key_decisions': [
                {
                    'id': d.decision_id,
                    'pressure': d.pressure,
                    'entropy': d.entropy,
                    'outcome': d.outcome,
                    'tick': d.tick
                } for d in recent_decisions[-10:]  # Last 10 decisions
            ],
            'reblooms': [
                {
                    'memory_id': r.memory_id,
                    'depth': r.depth,
                    'impact_score': r.impact_score,
                    'trigger': r.trigger
                } for r in recent_reblooms[-20:]  # Last 20 reblooms
            ],
            'schema_drift': {
                'magnitude': schema_drift.magnitude,
                'hot_edges': schema_drift.hot_edges,
                'stability_score': schema_drift.stability_score,
                'trend_direction': schema_drift.trend_direction
            },
            'residue_balance': {
                'soot_ratio': residue_balance.soot_ratio,
                'ash_ratio': residue_balance.ash_ratio,
                'ash_bias': residue_balance.ash_bias,
                'balance_stability': residue_balance.balance_stability
            },
            'coherence_metrics': self._calculate_coherence_metrics(),
            'system_health': self._assess_system_health()
        }
        
        return summary
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on audit summary"""
        recommendations = []
        
        # Check juliet density (rebloom rate)
        if len(summary['reblooms']) > 15:  # High rebloom rate
            recommendations.append(OwlRecommendation(
                action=RecommendationAction.CAP_REBLOOM_RATE,
                reason="juliet_density_high",
                priority=0.8,
                confidence=0.9
            ))
        
        # Check schema drift
        schema_drift = summary['schema_drift']
        if schema_drift['magnitude'] > self.drift_threshold:
            if schema_drift['stability_score'] < 0.6:
                recommendations.append(OwlRecommendation(
                    action=RecommendationAction.TRIGGER_WEAVING,
                    reason="schema_drift_excessive",
                    priority=0.9,
                    confidence=0.85
                ))
        
        # Check residue balance
        residue = summary['residue_balance']
        if residue['soot_ratio'] > 0.4:  # Too much soot
            recommendations.append(OwlRecommendation(
                action=RecommendationAction.SPAWN_BEETLE,
                reason="soot_buildup_cleanup_needed",
                priority=0.7,
                target="soot_heavy_regions",
                confidence=0.8
            ))
        
        # Check for clusters needing pollination
        if schema_drift['stability_score'] < 0.7 and len(schema_drift['hot_edges']) > 3:
            recommendations.append(OwlRecommendation(
                action=RecommendationAction.POLLINATE_CLUSTER,
                reason="schema_instability_detected",
                priority=0.6,
                target="cluster-high-volatility",
                parameters={'with': 'Bee'},
                confidence=0.75
            ))
        
        # Check coherence trends
        coherence = summary['coherence_metrics']
        if coherence.get('trend', 0) < -0.1:  # Declining coherence
            recommendations.append(OwlRecommendation(
                action=RecommendationAction.SPAWN_WHALE,
                reason="coherence_decline_trend",
                priority=0.8,
                confidence=0.7
            ))
        
        self.audit_stats['recommendations_made'] += len(recommendations)
        
        return [
            {
                'action': rec.action.value,
                'reason': rec.reason,
                'priority': rec.priority,
                'target': rec.target,
                'with': rec.parameters.get('with'),
                'confidence': rec.confidence
            }
            for rec in recommendations
        ]
    
    def _calculate_schema_drift(self) -> SchemaDrift:
        """Calculate current schema drift metrics"""
        # Simulate schema drift calculation based on historical data
        if len(self.schema_history) < 10:
            return SchemaDrift(
                magnitude=0.0,
                hot_edges=[],
                stability_score=1.0,
                trend_direction=0.0
            )
        
        # Calculate drift magnitude from recent schema changes
        recent_schemas = list(self.schema_history)[-50:]  # Last 50 schema states
        
        # Simulate drift calculation
        magnitude = np.std([s.get('drift_metric', 0.0) for s in recent_schemas])
        
        # Identify hot edges (high volatility connections)
        hot_edges = []
        if magnitude > 0.15:
            hot_edges = [f"edge-{i}" for i in range(min(5, int(magnitude * 20)))]
        
        # Calculate stability score
        stability_score = max(0.0, 1.0 - magnitude * 2)
        
        # Calculate trend direction
        if len(recent_schemas) >= 10:
            early_avg = np.mean([s.get('drift_metric', 0.0) for s in recent_schemas[:10]])
            late_avg = np.mean([s.get('drift_metric', 0.0) for s in recent_schemas[-10:]])
            trend_direction = late_avg - early_avg
        else:
            trend_direction = 0.0
        
        drift = SchemaDrift(
            magnitude=magnitude,
            hot_edges=hot_edges,
            stability_score=stability_score,
            trend_direction=trend_direction,
            critical_nodes=[f"node-{i}" for i in range(min(3, int(magnitude * 10)))]
        )
        
        # Alert if drift is excessive
        if magnitude > self.drift_threshold:
            self.audit_stats['drift_alerts'] += 1
            logger.warning(f"🦉 Schema drift alert: magnitude {magnitude:.3f} > threshold {self.drift_threshold}")
        
        return drift
    
    def _analyze_residue_balance(self) -> ResidueBalance:
        """Analyze current ash/soot residue balance"""
        # Simulate residue analysis based on logged events
        soot_events = [e for e in self.continuous_logs[LogSource.SOOT_ASH] 
                      if e['data'].get('type') == 'soot']
        ash_events = [e for e in self.continuous_logs[LogSource.SOOT_ASH] 
                     if e['data'].get('type') == 'ash']
        
        total_events = len(soot_events) + len(ash_events)
        soot_ratio = len(soot_events) / max(total_events, 1)
        ash_ratio = len(ash_events) / max(total_events, 1)
        
        # Simulate ash color bias (RGB components)
        ash_bias = [
            np.random.uniform(0.2, 0.8),  # R
            np.random.uniform(0.2, 0.8),  # G  
            np.random.uniform(0.2, 0.8)   # B
        ]
        
        # Calculate balance stability over time
        balance_stability = 1.0 - abs(soot_ratio - 0.3)  # Optimal soot ratio ~0.3
        
        # Calculate cleanup urgency
        cleanup_urgency = max(0.0, soot_ratio - 0.4)  # Urgent if soot > 40%
        
        return ResidueBalance(
            soot_ratio=soot_ratio,
            ash_ratio=ash_ratio,
            ash_bias=ash_bias,
            balance_stability=balance_stability,
            cleanup_urgency=cleanup_urgency
        )
    
    def _generate_narrative_summary(
        self,
        decisions: List[KeyDecision],
        reblooms: List[RebloomEvent],
        schema_drift: SchemaDrift,
        residue_balance: ResidueBalance
    ) -> str:
        """Generate a narrative summary of the current epoch"""
        
        narrative_parts = []
        
        # System state overview
        if schema_drift.stability_score > 0.8:
            narrative_parts.append("System exhibits strong schema stability")
        elif schema_drift.stability_score > 0.6:
            narrative_parts.append("Moderate schema fluctuations observed")
        else:
            narrative_parts.append("Significant schema instability detected")
        
        # Decision patterns
        if len(decisions) > 10:
            avg_pressure = np.mean([d.pressure for d in decisions])
            if avg_pressure > 0.7:
                narrative_parts.append("Operating under sustained high pressure")
            elif avg_pressure > 0.4:
                narrative_parts.append("Moderate pressure environment maintained")
            else:
                narrative_parts.append("Low pressure operations predominant")
        
        # Rebloom activity
        if len(reblooms) > 15:
            narrative_parts.append("High memory rebloom activity indicating active recall")
        elif len(reblooms) > 5:
            narrative_parts.append("Normal memory access patterns")
        else:
            narrative_parts.append("Low memory access suggesting stable operation")
        
        # Residue balance
        if residue_balance.soot_ratio > 0.4:
            narrative_parts.append("Elevated soot levels requiring attention")
        elif residue_balance.balance_stability > 0.8:
            narrative_parts.append("Well-balanced residue composition")
        
        # Schema drift concerns
        if schema_drift.magnitude > self.drift_threshold:
            narrative_parts.append(f"Schema drift magnitude {schema_drift.magnitude:.2f} exceeds safety threshold")
        
        return ". ".join(narrative_parts) + "."
    
    def _calculate_coherence_metrics(self) -> Dict[str, float]:
        """Calculate coherence trend metrics"""
        if len(self.coherence_history) < 10:
            return {'current': 0.8, 'trend': 0.0, 'stability': 1.0}
        
        recent_coherence = list(self.coherence_history)[-20:]
        current = recent_coherence[-1] if recent_coherence else 0.8
        
        # Calculate trend
        if len(recent_coherence) >= 10:
            x = np.arange(len(recent_coherence))
            coeffs = np.polyfit(x, recent_coherence, 1)
            trend = coeffs[0]  # Slope
        else:
            trend = 0.0
        
        # Calculate stability
        stability = 1.0 - np.std(recent_coherence)
        
        return {
            'current': current,
            'trend': trend,
            'stability': max(0.0, stability)
        }
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health"""
        schema_drift = self._calculate_schema_drift()
        coherence = self._calculate_coherence_metrics()
        
        # Calculate composite health score
        health_factors = {
            'schema_stability': schema_drift.stability_score,
            'coherence_current': coherence['current'],
            'coherence_stability': coherence['stability'],
            'trend_health': 1.0 - abs(coherence['trend']) if abs(coherence['trend']) < 0.5 else 0.0
        }
        
        overall_health = np.mean(list(health_factors.values()))
        
        # Determine health status
        if overall_health > 0.8:
            status = "excellent"
        elif overall_health > 0.6:
            status = "good"
        elif overall_health > 0.4:
            status = "fair"
        else:
            status = "concerning"
        
        return {
            'overall_score': overall_health,
            'status': status,
            'factors': health_factors,
            'alerts': self._generate_health_alerts(health_factors)
        }
    
    def _generate_health_alerts(self, health_factors: Dict[str, float]) -> List[str]:
        """Generate health alerts based on factors"""
        alerts = []
        
        if health_factors['schema_stability'] < 0.5:
            alerts.append("Schema stability critically low")
        
        if health_factors['coherence_current'] < 0.6:
            alerts.append("Coherence below operational threshold")
            self.audit_stats['coherence_warnings'] += 1
        
        if health_factors['coherence_stability'] < 0.4:
            alerts.append("Coherence highly unstable")
        
        return alerts
    
    def _process_schema_event(self, event_data: Dict[str, Any]) -> None:
        """Process a schema event in real-time"""
        schema_entry = {
            'tick': event_data.get('tick', 0),
            'drift_metric': event_data.get('drift_magnitude', 0.0),
            'stability': event_data.get('stability', 1.0),
            'timestamp': datetime.now()
        }
        
        self.schema_history.append(schema_entry)
    
    def _process_rebloom_event(self, event_data: Dict[str, Any]) -> None:
        """Process a Juliet rebloom event"""
        rebloom = RebloomEvent(
            memory_id=event_data.get('memory_id', 'unknown'),
            depth=event_data.get('depth', 1),
            trigger=event_data.get('trigger', 'unknown'),
            impact_score=event_data.get('impact_score', 0.5)
        )
        
        self.rebloom_tracker.append(rebloom)
    
    def _perform_deep_historical_analysis(self, current_tick: int) -> Dict[str, Any]:
        """Perform deep historical analysis for comprehensive audit"""
        return {
            'epoch_count': len(self.archived_epochs),
            'historical_drift_patterns': self._analyze_historical_drift(),
            'decision_outcome_analysis': self._analyze_decision_outcomes(),
            'long_term_coherence_trends': self._analyze_long_term_coherence(),
            'archive_integrity': self._check_archive_integrity()
        }
    
    def _analyze_historical_drift(self) -> Dict[str, Any]:
        """Analyze historical schema drift patterns"""
        if not self.archived_epochs:
            return {'pattern': 'insufficient_data'}
        
        drift_magnitudes = [epoch.schema_drift.magnitude for epoch in self.archived_epochs]
        
        return {
            'average_drift': np.mean(drift_magnitudes),
            'drift_trend': np.polyfit(range(len(drift_magnitudes)), drift_magnitudes, 1)[0],
            'stability_pattern': np.std(drift_magnitudes),
            'peak_drift_epochs': [i for i, d in enumerate(drift_magnitudes) if d > 0.3]
        }
    
    def _trace_provenance_chains(self) -> Dict[str, Any]:
        """Trace provenance chains for major decisions and outcomes"""
        # Simulate provenance tracing
        return {
            'tracked_decisions': len(self.decision_registry),
            'provenance_depth': 5,  # Average depth of provenance chains
            'integrity_score': 0.92,  # Provenance integrity
            'missing_links': 3  # Number of missing provenance links
        }
    
    def _archive_epoch(self, current_tick: int, summary: Dict[str, Any]) -> None:
        """Archive the current epoch for long-term storage"""
        epoch_summary = EpochSummary(
            tick_window=(self.current_epoch_start, current_tick),
            narrative=summary['narrative'],
            key_decisions=[KeyDecision(
                decision_id=d['id'],
                tick=d['tick'],
                pressure=d['pressure'],
                entropy=d['entropy'],
                outcome=d['outcome']
            ) for d in summary['key_decisions']],
            reblooms=[RebloomEvent(
                memory_id=r['memory_id'],
                depth=r['depth'],
                trigger=r['trigger'],
                impact_score=r['impact_score']
            ) for r in summary['reblooms']],
            schema_drift=SchemaDrift(
                magnitude=summary['schema_drift']['magnitude'],
                hot_edges=summary['schema_drift']['hot_edges'],
                stability_score=summary['schema_drift']['stability_score'],
                trend_direction=summary['schema_drift']['trend_direction']
            ),
            residue_balance=ResidueBalance(
                soot_ratio=summary['residue_balance']['soot_ratio'],
                ash_ratio=summary['residue_balance']['ash_ratio'],
                ash_bias=summary['residue_balance']['ash_bias'],
                balance_stability=summary['residue_balance']['balance_stability'],
                cleanup_urgency=0.0
            ),
            sources_analyzed=list(LogSource),
            coherence_trend=summary.get('coherence_metrics', {}).get('trend', 0.0),
            stability_metrics=summary.get('coherence_metrics', {})
        )
        
        # Add to archive
        self.archived_epochs.append(epoch_summary)
        
        # Trim archive if too large
        if len(self.archived_epochs) > self.max_archives:
            self.archived_epochs = self.archived_epochs[-self.max_archives:]
        
        # Save to disk
        self._save_epoch_archive(epoch_summary)
        
        # Update epoch tracking
        self.current_epoch_start = current_tick
        self.audit_stats['epochs_archived'] += 1
        
        logger.info(f"🦉 Archived epoch {len(self.archived_epochs)} "
                   f"(ticks {epoch_summary.tick_window[0]}-{epoch_summary.tick_window[1]})")
    
    def _save_epoch_archive(self, epoch: EpochSummary) -> None:
        """Save epoch archive to disk"""
        try:
            filename = f"epoch_{epoch.tick_window[0]}_{epoch.tick_window[1]}.json"
            filepath = os.path.join(self.archive_dir, filename)
            
            # Convert to serializable format
            epoch_data = {
                'tick_window': epoch.tick_window,
                'narrative': epoch.narrative,
                'key_decisions': [
                    {
                        'decision_id': d.decision_id,
                        'tick': d.tick,
                        'pressure': d.pressure,
                        'entropy': d.entropy,
                        'outcome': d.outcome,
                        'timestamp': d.timestamp.isoformat()
                    } for d in epoch.key_decisions
                ],
                'schema_drift': {
                    'magnitude': epoch.schema_drift.magnitude,
                    'hot_edges': epoch.schema_drift.hot_edges,
                    'stability_score': epoch.schema_drift.stability_score,
                    'trend_direction': epoch.schema_drift.trend_direction
                },
                'residue_balance': {
                    'soot_ratio': epoch.residue_balance.soot_ratio,
                    'ash_ratio': epoch.residue_balance.ash_ratio,
                    'ash_bias': epoch.residue_balance.ash_bias,
                    'balance_stability': epoch.residue_balance.balance_stability
                },
                'coherence_trend': epoch.coherence_trend,
                'timestamp': epoch.timestamp.isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(epoch_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"🦉 Failed to save epoch archive: {e}")
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist"""
        os.makedirs(self.archive_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.dialogue_log_path), exist_ok=True)
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get comprehensive audit summary"""
        return {
            'tracer_type': 'Owl',
            'role': 'long_memory_auditor',
            'statistics': self.audit_stats.copy(),
            'current_epoch': {
                'start_tick': self.current_epoch_start,
                'logged_events': {source.value: len(events) 
                                for source, events in self.continuous_logs.items()},
                'decisions_tracked': len(self.decision_registry),
                'reblooms_tracked': len(self.rebloom_tracker)
            },
            'archives': {
                'total_epochs': len(self.archived_epochs),
                'oldest_epoch': self.archived_epochs[0].tick_window if self.archived_epochs else None,
                'newest_epoch': self.archived_epochs[-1].tick_window if self.archived_epochs else None
            },
            'health_indicators': {
                'schema_drift_alerts': self.audit_stats['drift_alerts'],
                'coherence_warnings': self.audit_stats['coherence_warnings'],
                'archive_integrity': 'good'  # Could be calculated
            }
        }


# Global Owl tracer instance
_owl_tracer = None

def get_owl_tracer(config: Dict[str, Any] = None) -> OwlTracer:
    """Get the global Owl tracer instance"""
    global _owl_tracer
    if _owl_tracer is None:
        _owl_tracer = OwlTracer(config)
    return _owl_tracer


# Example usage and testing
if __name__ == "__main__":
    print("🦉 Testing DAWN Owl Tracer")
    print("=" * 50)
    
    # Create Owl tracer
    owl = OwlTracer()
    
    # Simulate some events
    owl.log_event(LogSource.FRACTAL_LOG, {
        'tick': 1000,
        'event_type': 'memory_access',
        'memory_id': 'mem_001',
        'depth': 3
    })
    
    owl.log_event(LogSource.SCHEMA_EVENTS, {
        'tick': 1001,
        'drift_magnitude': 0.15,
        'stability': 0.85,
        'affected_nodes': ['node_a', 'node_b']
    })
    
    # Track a decision
    decision = KeyDecision(
        decision_id="decision_001",
        tick=1002,
        pressure=0.65,
        entropy=0.3,
        outcome="schema_weaving_triggered"
    )
    owl.track_decision(decision)
    
    # Perform audit
    audit_result = owl.perform_audit(1100, AuditLevel.STANDARD)
    
    print(f"Audit Results:")
    print(f"  Sources analyzed: {audit_result['sources']}")
    print(f"  Recommendations: {len(audit_result['recommendations'])}")
    
    for rec in audit_result['recommendations']:
        print(f"    - {rec['action']}: {rec['reason']} (priority: {rec['priority']:.2f})")
    
    # Get summary
    summary = owl.get_audit_summary()
    print(f"\nOwl Summary:")
    print(f"  Total audits: {summary['statistics']['total_audits']}")
    print(f"  Decisions tracked: {summary['statistics']['decisions_tracked']}")
    print(f"  Epochs archived: {summary['statistics']['epochs_archived']}")
    
    print("\n🦉 Owl Tracer operational - Long-memory guardian active!")
