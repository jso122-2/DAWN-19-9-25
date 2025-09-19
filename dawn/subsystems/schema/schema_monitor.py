"""
DAWN Schema Monitor
==================
Real-time monitoring system for schema health and anomaly detection.

Author: DAWN Development Team
Generated: 2025-09-18
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque
import threading


class AnomalySeverity(Enum):
    """Anomaly severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Detected anomaly information"""
    timestamp: float
    metric: str
    severity: AnomalySeverity
    value: float
    threshold: float
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'metric': self.metric,
            'severity': self.severity.value,
            'value': self.value,
            'threshold': self.threshold,
            'description': self.description
        }


class SchemaMonitor:
    """Real-time schema monitoring and anomaly detection"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Thresholds
        self.shi_critical = 0.2
        self.shi_warning = 0.4
        self.scup_critical = 0.3
        self.scup_warning = 0.5
        
        # History tracking
        self.shi_history = deque(maxlen=100)
        self.scup_history = deque(maxlen=100)
        self.anomaly_history = deque(maxlen=500)
        
        # Performance tracking
        self.monitoring_count = 0
        self.anomaly_count = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        print("[SchemaMonitor] 👁️ Schema monitoring system initialized")
        
    def monitor_real_time(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform real-time monitoring"""
        start_time = time.time()
        
        try:
            with self.lock:
                shi = monitoring_data.get('shi', 0.5)
                scup = monitoring_data.get('scup', 0.5)
                
                # Update histories
                self.shi_history.append(shi)
                self.scup_history.append(scup)
                
                # Detect anomalies
                anomalies = []
                
                # SHI anomalies
                if shi < self.shi_critical:
                    anomalies.append(Anomaly(
                        timestamp=time.time(),
                        metric="shi_degradation",
                        severity=AnomalySeverity.CRITICAL,
                        value=shi,
                        threshold=self.shi_critical,
                        description=f"Critical SHI level: {shi:.3f}"
                    ))
                elif shi < self.shi_warning:
                    anomalies.append(Anomaly(
                        timestamp=time.time(),
                        metric="shi_degradation",
                        severity=AnomalySeverity.HIGH,
                        value=shi,
                        threshold=self.shi_warning,
                        description=f"Low SHI level: {shi:.3f}"
                    ))
                
                # SCUP anomalies
                if scup < self.scup_critical:
                    anomalies.append(Anomaly(
                        timestamp=time.time(),
                        metric="scup_instability",
                        severity=AnomalySeverity.CRITICAL,
                        value=scup,
                        threshold=self.scup_critical,
                        description=f"Critical SCUP level: {scup:.3f}"
                    ))
                elif scup < self.scup_warning:
                    anomalies.append(Anomaly(
                        timestamp=time.time(),
                        metric="scup_instability",
                        severity=AnomalySeverity.HIGH,
                        value=scup,
                        threshold=self.scup_warning,
                        description=f"Low SCUP level: {scup:.3f}"
                    ))
                
                # Coherence mismatch
                coherence_diff = abs(shi - scup)
                if coherence_diff > 0.4:
                    anomalies.append(Anomaly(
                        timestamp=time.time(),
                        metric="coherence_mismatch",
                        severity=AnomalySeverity.MEDIUM,
                        value=coherence_diff,
                        threshold=0.4,
                        description=f"SHI-SCUP mismatch: {coherence_diff:.3f}"
                    ))
                
                # Update counters
                self.monitoring_count += 1
                if anomalies:
                    self.anomaly_count += len(anomalies)
                    self.anomaly_history.extend(anomalies)
                
                # Calculate health status
                health_status = self._calculate_health_status(shi, scup, anomalies)
                
                return {
                    'health_status': health_status,
                    'anomalies': [a.to_dict() for a in anomalies],
                    'monitoring_time': time.time() - start_time
                }
                
        except Exception as e:
            print(f"[SchemaMonitor] ❌ Error: {str(e)}")
            return {
                'health_status': {'status': 'error', 'message': str(e)},
                'anomalies': [],
                'monitoring_time': 0.0
            }
    
    def _calculate_health_status(self, shi: float, scup: float, anomalies: List[Anomaly]) -> Dict[str, Any]:
        """Calculate overall health status"""
        critical_count = sum(1 for a in anomalies if a.severity == AnomalySeverity.CRITICAL)
        high_count = sum(1 for a in anomalies if a.severity == AnomalySeverity.HIGH)
        
        if critical_count > 0:
            status = "critical"
            message = f"{critical_count} critical anomalies"
        elif high_count > 0:
            status = "degraded"
            message = f"{high_count} high-severity anomalies"
        elif shi > 0.7 and scup > 0.7:
            status = "healthy"
            message = "Operating normally"
        else:
            status = "stable"
            message = "Stable monitoring"
        
        return {
            'status': status,
            'message': message,
            'shi': shi,
            'scup': scup
        }
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            'monitoring_count': self.monitoring_count,
            'anomaly_count': self.anomaly_count,
            'history_sizes': {
                'shi': len(self.shi_history),
                'scup': len(self.scup_history),
                'anomalies': len(self.anomaly_history)
            }
        }


def create_schema_monitor(config: Optional[Dict[str, Any]] = None) -> SchemaMonitor:
    """Create schema monitor"""
    return SchemaMonitor(config)