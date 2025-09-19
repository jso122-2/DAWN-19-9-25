#!/usr/bin/env python3
"""
DAWN Telemetry Exporters
========================

Export telemetry data to various formats and destinations.
Supports JSON, CSV, Prometheus, InfluxDB, and other telemetry systems.
"""

import json
import csv
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, IO
from datetime import datetime
from abc import ABC, abstractmethod
import gzip
import threading

from .logger import TelemetryEvent

logger = logging.getLogger(__name__)

class TelemetryExporter(ABC):
    """Base class for telemetry exporters."""
    
    def __init__(self, output_path: str, config: Dict[str, Any] = None):
        """
        Initialize exporter.
        
        Args:
            output_path: Output file or directory path
            config: Exporter-specific configuration
        """
        self.output_path = Path(output_path)
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.lock = threading.Lock()
        
        # Create output directory if needed
        if self.output_path.suffix == '':  # Directory
            self.output_path.mkdir(parents=True, exist_ok=True)
        else:  # File
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def write_events(self, events: List[TelemetryEvent]) -> None:
        """Write events to output destination."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close exporter and cleanup resources."""
        pass

class JSONExporter(TelemetryExporter):
    """Export telemetry events to JSON Lines format."""
    
    def __init__(self, output_path: str, config: Dict[str, Any] = None):
        super().__init__(output_path, config)
        
        self.compress = self.config.get('compress', False)
        self.pretty_print = self.config.get('pretty_print', False)
        self.max_file_size_mb = self.config.get('max_file_size_mb', 100)
        self.max_files = self.config.get('max_files', 10)
        
        # File rotation
        self.current_file = None
        self.current_file_size = 0
        self.file_counter = 0
        
        # Determine if output is directory or file
        if self.output_path.suffix == '':
            self.is_directory = True
            self.base_filename = "telemetry"
        else:
            self.is_directory = False
            self.base_filename = self.output_path.stem
    
    def write_events(self, events: List[TelemetryEvent]) -> None:
        """Write events to JSON Lines file."""
        if not self.enabled or not events:
            return
        
        with self.lock:
            try:
                # Check if we need to rotate file
                if self._should_rotate_file():
                    self._rotate_file()
                
                # Open file if not already open
                if self.current_file is None:
                    self._open_new_file()
                
                # Write events
                for event in events:
                    event_dict = event.to_dict()
                    
                    if self.pretty_print:
                        json_line = json.dumps(event_dict, indent=2)
                    else:
                        json_line = json.dumps(event_dict, separators=(',', ':'))
                    
                    json_line += '\n'
                    
                    if self.compress and hasattr(self.current_file, 'write'):
                        # For gzip files
                        self.current_file.write(json_line.encode('utf-8'))
                    else:
                        self.current_file.write(json_line)
                    
                    self.current_file_size += len(json_line.encode('utf-8'))
                
                # Flush to ensure data is written
                self.current_file.flush()
                
            except Exception as e:
                logger.error(f"Error writing events to JSON: {e}")
    
    def _should_rotate_file(self) -> bool:
        """Check if file should be rotated."""
        if self.current_file is None:
            return False
        
        max_size_bytes = self.max_file_size_mb * 1024 * 1024
        return self.current_file_size >= max_size_bytes
    
    def _rotate_file(self) -> None:
        """Rotate current file."""
        if self.current_file:
            self.current_file.close()
            self.current_file = None
            self.current_file_size = 0
        
        # Clean up old files if we exceed max_files
        self._cleanup_old_files()
    
    def _open_new_file(self) -> None:
        """Open a new output file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.is_directory:
            if self.compress:
                filename = f"{self.base_filename}_{timestamp}_{self.file_counter:03d}.jsonl.gz"
                filepath = self.output_path / filename
                self.current_file = gzip.open(filepath, 'wt', encoding='utf-8')
            else:
                filename = f"{self.base_filename}_{timestamp}_{self.file_counter:03d}.jsonl"
                filepath = self.output_path / filename
                self.current_file = open(filepath, 'w', encoding='utf-8')
        else:
            if self.compress:
                filepath = self.output_path.with_suffix('.jsonl.gz')
                self.current_file = gzip.open(filepath, 'wt', encoding='utf-8')
            else:
                self.current_file = open(self.output_path, 'w', encoding='utf-8')
        
        self.file_counter += 1
        self.current_file_size = 0
        
        logger.info(f"Opened new JSON telemetry file: {filepath}")
    
    def _cleanup_old_files(self) -> None:
        """Remove old files if we exceed max_files."""
        if not self.is_directory:
            return
        
        # Find all telemetry files
        pattern = f"{self.base_filename}_*.jsonl*"
        files = list(self.output_path.glob(pattern))
        
        # Sort by creation time
        files.sort(key=lambda f: f.stat().st_ctime)
        
        # Remove oldest files if we exceed limit
        while len(files) >= self.max_files:
            oldest_file = files.pop(0)
            try:
                oldest_file.unlink()
                logger.info(f"Removed old telemetry file: {oldest_file}")
            except Exception as e:
                logger.error(f"Error removing old file {oldest_file}: {e}")
    
    def close(self) -> None:
        """Close the exporter."""
        with self.lock:
            if self.current_file:
                self.current_file.close()
                self.current_file = None

class CSVExporter(TelemetryExporter):
    """Export telemetry events to CSV format."""
    
    def __init__(self, output_path: str, config: Dict[str, Any] = None):
        super().__init__(output_path, config)
        
        self.fieldnames = [
            'timestamp', 'level', 'subsystem', 'component', 'event_type',
            'tick_id', 'session_id', 'data', 'metadata'
        ]
        
        self.max_file_size_mb = self.config.get('max_file_size_mb', 100)
        self.max_files = self.config.get('max_files', 10)
        
        self.current_file = None
        self.csv_writer = None
        self.current_file_size = 0
        self.file_counter = 0
        self.header_written = False
    
    def write_events(self, events: List[TelemetryEvent]) -> None:
        """Write events to CSV file."""
        if not self.enabled or not events:
            return
        
        with self.lock:
            try:
                # Check if we need to rotate file
                if self._should_rotate_file():
                    self._rotate_file()
                
                # Open file if not already open
                if self.current_file is None:
                    self._open_new_file()
                
                # Write events
                for event in events:
                    row = {
                        'timestamp': event.timestamp,
                        'level': event.level,
                        'subsystem': event.subsystem,
                        'component': event.component,
                        'event_type': event.event_type,
                        'tick_id': event.tick_id or '',
                        'session_id': event.session_id,
                        'data': json.dumps(event.data) if event.data else '',
                        'metadata': json.dumps(event.metadata) if event.metadata else ''
                    }
                    
                    self.csv_writer.writerow(row)
                    
                    # Estimate file size (rough approximation)
                    self.current_file_size += len(str(row))
                
                # Flush to ensure data is written
                self.current_file.flush()
                
            except Exception as e:
                logger.error(f"Error writing events to CSV: {e}")
    
    def _should_rotate_file(self) -> bool:
        """Check if file should be rotated."""
        if self.current_file is None:
            return False
        
        max_size_bytes = self.max_file_size_mb * 1024 * 1024
        return self.current_file_size >= max_size_bytes
    
    def _rotate_file(self) -> None:
        """Rotate current file."""
        if self.current_file:
            self.current_file.close()
            self.current_file = None
            self.csv_writer = None
            self.current_file_size = 0
            self.header_written = False
        
        self._cleanup_old_files()
    
    def _open_new_file(self) -> None:
        """Open a new CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.output_path.suffix == '':
            filename = f"telemetry_{timestamp}_{self.file_counter:03d}.csv"
            filepath = self.output_path / filename
        else:
            filepath = self.output_path
        
        self.current_file = open(filepath, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(self.current_file, fieldnames=self.fieldnames)
        
        # Write header
        if not self.header_written:
            self.csv_writer.writeheader()
            self.header_written = True
        
        self.file_counter += 1
        self.current_file_size = 0
        
        logger.info(f"Opened new CSV telemetry file: {filepath}")
    
    def _cleanup_old_files(self) -> None:
        """Remove old CSV files if we exceed max_files."""
        if self.output_path.suffix != '':
            return
        
        pattern = "telemetry_*.csv"
        files = list(self.output_path.glob(pattern))
        files.sort(key=lambda f: f.stat().st_ctime)
        
        while len(files) >= self.max_files:
            oldest_file = files.pop(0)
            try:
                oldest_file.unlink()
                logger.info(f"Removed old CSV file: {oldest_file}")
            except Exception as e:
                logger.error(f"Error removing old CSV file {oldest_file}: {e}")
    
    def close(self) -> None:
        """Close the CSV exporter."""
        with self.lock:
            if self.current_file:
                self.current_file.close()
                self.current_file = None
                self.csv_writer = None

class PrometheusExporter(TelemetryExporter):
    """Export telemetry metrics to Prometheus format."""
    
    def __init__(self, output_path: str, config: Dict[str, Any] = None):
        super().__init__(output_path, config)
        
        self.metrics_cache = {}
        self.last_export_time = 0
        self.export_interval = self.config.get('export_interval', 60)  # seconds
        
        # Prometheus metric types
        self.counters = {}
        self.gauges = {}
        self.histograms = {}
    
    def write_events(self, events: List[TelemetryEvent]) -> None:
        """Process events and update Prometheus metrics."""
        if not self.enabled or not events:
            return
        
        current_time = time.time()
        
        with self.lock:
            try:
                # Process events to update metrics
                for event in events:
                    self._process_event_for_metrics(event)
                
                # Export metrics if enough time has passed
                if current_time - self.last_export_time >= self.export_interval:
                    self._export_prometheus_metrics()
                    self.last_export_time = current_time
                
            except Exception as e:
                logger.error(f"Error processing events for Prometheus: {e}")
    
    def _process_event_for_metrics(self, event: TelemetryEvent) -> None:
        """Process a single event to update Prometheus metrics."""
        labels = {
            'subsystem': event.subsystem,
            'component': event.component,
            'level': event.level
        }
        
        # Event counter
        counter_name = f'dawn_telemetry_events_total'
        self._increment_counter(counter_name, labels)
        
        # Level-specific counters
        level_counter = f'dawn_telemetry_{event.level.lower()}_events_total'
        self._increment_counter(level_counter, labels)
        
        # Performance metrics
        if event.event_type == 'performance_metric' and 'duration_ms' in event.data:
            duration = event.data['duration_ms']
            duration_gauge = f'dawn_operation_duration_ms'
            operation_labels = {**labels, 'operation': event.data.get('operation', 'unknown')}
            self._set_gauge(duration_gauge, duration, operation_labels)
            
            # Success rate
            if 'success' in event.data:
                success_counter = f'dawn_operation_success_total'
                success_labels = {**operation_labels, 'success': str(event.data['success']).lower()}
                self._increment_counter(success_counter, success_labels)
        
        # System metrics
        if event.subsystem == 'system':
            if event.component == 'cpu' and 'cpu_percent' in event.data:
                self._set_gauge('dawn_system_cpu_usage_percent', event.data['cpu_percent'], {})
            elif event.component == 'memory':
                if 'memory_percent' in event.data:
                    self._set_gauge('dawn_system_memory_usage_percent', event.data['memory_percent'], {})
                if 'memory_used_mb' in event.data:
                    self._set_gauge('dawn_system_memory_used_mb', event.data['memory_used_mb'], {})
    
    def _increment_counter(self, name: str, labels: Dict[str, str]) -> None:
        """Increment a Prometheus counter."""
        key = (name, tuple(sorted(labels.items())))
        if key not in self.counters:
            self.counters[key] = 0
        self.counters[key] += 1
    
    def _set_gauge(self, name: str, value: float, labels: Dict[str, str]) -> None:
        """Set a Prometheus gauge value."""
        key = (name, tuple(sorted(labels.items())))
        self.gauges[key] = value
    
    def _export_prometheus_metrics(self) -> None:
        """Export metrics to Prometheus format file."""
        try:
            output_file = self.output_path if self.output_path.suffix else self.output_path / "metrics.prom"
            
            with open(output_file, 'w') as f:
                # Write timestamp
                f.write(f"# HELP dawn_telemetry_export_timestamp_seconds Timestamp of last export\n")
                f.write(f"# TYPE dawn_telemetry_export_timestamp_seconds gauge\n")
                f.write(f"dawn_telemetry_export_timestamp_seconds {time.time()}\n\n")
                
                # Write counters
                written_metrics = set()
                for (metric_name, labels_tuple), value in self.counters.items():
                    if metric_name not in written_metrics:
                        f.write(f"# HELP {metric_name} Total number of events\n")
                        f.write(f"# TYPE {metric_name} counter\n")
                        written_metrics.add(metric_name)
                    
                    labels_str = self._format_prometheus_labels(dict(labels_tuple))
                    f.write(f"{metric_name}{labels_str} {value}\n")
                
                f.write("\n")
                
                # Write gauges
                for (metric_name, labels_tuple), value in self.gauges.items():
                    if metric_name not in written_metrics:
                        f.write(f"# HELP {metric_name} Current gauge value\n")
                        f.write(f"# TYPE {metric_name} gauge\n")
                        written_metrics.add(metric_name)
                    
                    labels_str = self._format_prometheus_labels(dict(labels_tuple))
                    f.write(f"{metric_name}{labels_str} {value}\n")
            
            logger.debug(f"Exported Prometheus metrics to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting Prometheus metrics: {e}")
    
    def _format_prometheus_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus format."""
        if not labels:
            return ""
        
        label_pairs = []
        for key, value in labels.items():
            # Escape quotes in values
            escaped_value = value.replace('"', '\\"')
            label_pairs.append(f'{key}="{escaped_value}"')
        
        return "{" + ",".join(label_pairs) + "}"
    
    def close(self) -> None:
        """Close the Prometheus exporter."""
        with self.lock:
            # Final export
            self._export_prometheus_metrics()

class InfluxDBExporter(TelemetryExporter):
    """Export telemetry events to InfluxDB line protocol format."""
    
    def __init__(self, output_path: str, config: Dict[str, Any] = None):
        super().__init__(output_path, config)
        
        self.measurement_prefix = self.config.get('measurement_prefix', 'dawn_telemetry')
        self.batch_size = self.config.get('batch_size', 1000)
        self.current_batch = []
    
    def write_events(self, events: List[TelemetryEvent]) -> None:
        """Write events to InfluxDB line protocol format."""
        if not self.enabled or not events:
            return
        
        with self.lock:
            try:
                for event in events:
                    line = self._event_to_line_protocol(event)
                    if line:
                        self.current_batch.append(line)
                
                # Write batch if it's large enough
                if len(self.current_batch) >= self.batch_size:
                    self._write_batch()
                
            except Exception as e:
                logger.error(f"Error writing events to InfluxDB format: {e}")
    
    def _event_to_line_protocol(self, event: TelemetryEvent) -> Optional[str]:
        """Convert telemetry event to InfluxDB line protocol format."""
        try:
            # Measurement name
            measurement = f"{self.measurement_prefix}_{event.event_type}"
            
            # Tags (indexed fields)
            tags = {
                'subsystem': event.subsystem,
                'component': event.component,
                'level': event.level,
                'session_id': event.session_id
            }
            
            if event.tick_id is not None:
                tags['tick_id'] = str(event.tick_id)
            
            # Fields (non-indexed data)
            fields = {}
            
            # Add data fields
            for key, value in event.data.items():
                if isinstance(value, (int, float)):
                    fields[f"data_{key}"] = value
                elif isinstance(value, bool):
                    fields[f"data_{key}"] = value
                elif isinstance(value, str) and len(value) < 100:  # Limit string field length
                    fields[f'data_{key}'] = f'"{value}"'
            
            # Add metadata fields
            for key, value in event.metadata.items():
                if isinstance(value, (int, float)):
                    fields[f"meta_{key}"] = value
                elif isinstance(value, bool):
                    fields[f"meta_{key}"] = value
            
            # Always include an event count field
            fields['count'] = 1
            
            if not fields:
                return None
            
            # Parse timestamp
            timestamp_ns = self._parse_timestamp_to_nanoseconds(event.timestamp)
            
            # Format tags
            tag_str = ",".join([f"{k}={v}" for k, v in tags.items()])
            
            # Format fields
            field_str = ",".join([f"{k}={v}" for k, v in fields.items()])
            
            # Construct line protocol
            line = f"{measurement},{tag_str} {field_str} {timestamp_ns}"
            
            return line
            
        except Exception as e:
            logger.error(f"Error converting event to line protocol: {e}")
            return None
    
    def _parse_timestamp_to_nanoseconds(self, timestamp_str: str) -> int:
        """Parse timestamp string to nanoseconds since epoch."""
        try:
            if timestamp_str.endswith('Z'):
                dt = datetime.fromisoformat(timestamp_str[:-1] + '+00:00')
            else:
                dt = datetime.fromisoformat(timestamp_str)
            
            return int(dt.timestamp() * 1_000_000_000)
        except:
            # Fallback to current time
            return int(time.time() * 1_000_000_000)
    
    def _write_batch(self) -> None:
        """Write current batch to file."""
        if not self.current_batch:
            return
        
        try:
            output_file = self.output_path if self.output_path.suffix else self.output_path / "telemetry.influx"
            
            with open(output_file, 'a', encoding='utf-8') as f:
                for line in self.current_batch:
                    f.write(line + '\n')
            
            logger.debug(f"Wrote {len(self.current_batch)} lines to InfluxDB file")
            self.current_batch.clear()
            
        except Exception as e:
            logger.error(f"Error writing InfluxDB batch: {e}")
    
    def close(self) -> None:
        """Close the InfluxDB exporter."""
        with self.lock:
            # Write any remaining batch
            if self.current_batch:
                self._write_batch()

# Factory function to create exporters
def create_exporter(format_type: str, output_path: str, config: Dict[str, Any] = None) -> TelemetryExporter:
    """
    Create a telemetry exporter for the specified format.
    
    Args:
        format_type: Type of exporter ('json', 'csv', 'prometheus', 'influxdb')
        output_path: Output path for the exporter
        config: Exporter-specific configuration
        
    Returns:
        TelemetryExporter instance
        
    Raises:
        ValueError: If format_type is not supported
    """
    exporters = {
        'json': JSONExporter,
        'csv': CSVExporter,
        'prometheus': PrometheusExporter,
        'influxdb': InfluxDBExporter
    }
    
    if format_type not in exporters:
        raise ValueError(f"Unsupported exporter format: {format_type}. Supported: {list(exporters.keys())}")
    
    return exporters[format_type](output_path, config)
