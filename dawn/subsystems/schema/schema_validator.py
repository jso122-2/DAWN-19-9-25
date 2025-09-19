"""
DAWN Schema Validator
====================
Multi-layer validation framework for schema compliance and integrity checking.

Implements comprehensive validation across syntax, semantic, coherence, and 
performance layers to ensure schema reliability and consistency.

Author: DAWN Development Team
Generated: 2025-09-18
"""

import time
import json
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading

# Import torch with fallback for systems without PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ValidationLevel(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationLayer(Enum):
    """Validation layer types"""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    COHERENCE = "coherence"
    PERFORMANCE = "performance"


@dataclass
class ValidationResult:
    """Individual validation result"""
    layer: ValidationLayer
    level: ValidationLevel
    valid: bool
    message: str
    details: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'layer': self.layer.value,
            'level': self.level.value,
            'valid': self.valid,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp
        }


@dataclass
class ValidationSummary:
    """Overall validation summary"""
    overall_valid: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    errors: int
    critical_issues: int
    validation_time: float
    results: List[ValidationResult]
    
    def get_pass_rate(self) -> float:
        if self.total_checks == 0:
            return 1.0
        return self.passed_checks / self.total_checks


class SyntaxValidator:
    """Syntax validation layer"""
    
    def __init__(self):
        self.required_fields = ['nodes', 'edges', 'signals', 'shi', 'scup']
        self.node_required = ['id', 'health']
        self.edge_required = ['id', 'source', 'target', 'weight']
        
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate syntax structure"""
        try:
            # Check required top-level fields
            missing_fields = [field for field in self.required_fields if field not in data]
            if missing_fields:
                return ValidationResult(
                    layer=ValidationLayer.SYNTAX,
                    level=ValidationLevel.ERROR,
                    valid=False,
                    message=f"Missing required fields: {missing_fields}",
                    details={'missing_fields': missing_fields},
                    timestamp=time.time()
                )
            
            # Validate nodes structure
            if 'nodes' in data and isinstance(data['nodes'], list):
                for i, node in enumerate(data['nodes']):
                    missing_node_fields = [field for field in self.node_required if field not in node]
                    if missing_node_fields:
                        return ValidationResult(
                            layer=ValidationLayer.SYNTAX,
                            level=ValidationLevel.ERROR,
                            valid=False,
                            message=f"Node {i} missing fields: {missing_node_fields}",
                            details={'node_index': i, 'missing_fields': missing_node_fields},
                            timestamp=time.time()
                        )
            
            # Validate edges structure
            if 'edges' in data and isinstance(data['edges'], list):
                for i, edge in enumerate(data['edges']):
                    missing_edge_fields = [field for field in self.edge_required if field not in edge]
                    if missing_edge_fields:
                        return ValidationResult(
                            layer=ValidationLayer.SYNTAX,
                            level=ValidationLevel.ERROR,
                            valid=False,
                            message=f"Edge {i} missing fields: {missing_edge_fields}",
                            details={'edge_index': i, 'missing_fields': missing_edge_fields},
                            timestamp=time.time()
                        )
            
            return ValidationResult(
                layer=ValidationLayer.SYNTAX,
                level=ValidationLevel.INFO,
                valid=True,
                message="Syntax validation passed",
                details={'checked_fields': self.required_fields},
                timestamp=time.time()
            )
            
        except Exception as e:
            return ValidationResult(
                layer=ValidationLayer.SYNTAX,
                level=ValidationLevel.CRITICAL,
                valid=False,
                message=f"Syntax validation error: {str(e)}",
                details={'error': str(e)},
                timestamp=time.time()
            )


class SemanticValidator:
    """Semantic validation layer"""
    
    def __init__(self):
        self.health_range = (0.0, 1.0)
        self.weight_range = (0.0, 1.0)
        self.shi_range = (0.0, 1.0)
        self.scup_range = (0.0, 1.0)
        
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate semantic consistency"""
        try:
            issues = []
            
            # Validate SHI range
            if 'shi' in data:
                shi = data['shi']
                if not (self.shi_range[0] <= shi <= self.shi_range[1]):
                    issues.append(f"SHI value {shi} outside valid range {self.shi_range}")
            
            # Validate SCUP range
            if 'scup' in data:
                scup = data['scup']
                if not (self.scup_range[0] <= scup <= self.scup_range[1]):
                    issues.append(f"SCUP value {scup} outside valid range {self.scup_range}")
            
            # Validate node health values
            if 'nodes' in data:
                for i, node in enumerate(data['nodes']):
                    if 'health' in node:
                        health = node['health']
                        if not (self.health_range[0] <= health <= self.health_range[1]):
                            issues.append(f"Node {i} health {health} outside valid range {self.health_range}")
            
            # Validate edge weights
            if 'edges' in data:
                for i, edge in enumerate(data['edges']):
                    if 'weight' in edge:
                        weight = edge['weight']
                        if not (self.weight_range[0] <= weight <= self.weight_range[1]):
                            issues.append(f"Edge {i} weight {weight} outside valid range {self.weight_range}")
            
            # Check for edge connectivity issues
            node_ids = set()
            if 'nodes' in data:
                node_ids = {node.get('id') for node in data['nodes'] if 'id' in node}
            
            if 'edges' in data:
                for i, edge in enumerate(data['edges']):
                    source = edge.get('source')
                    target = edge.get('target')
                    if source and source not in node_ids:
                        issues.append(f"Edge {i} source '{source}' references non-existent node")
                    if target and target not in node_ids:
                        issues.append(f"Edge {i} target '{target}' references non-existent node")
            
            if issues:
                return ValidationResult(
                    layer=ValidationLayer.SEMANTIC,
                    level=ValidationLevel.WARNING if len(issues) < 3 else ValidationLevel.ERROR,
                    valid=len(issues) == 0,
                    message=f"Semantic validation found {len(issues)} issues",
                    details={'issues': issues},
                    timestamp=time.time()
                )
            
            return ValidationResult(
                layer=ValidationLayer.SEMANTIC,
                level=ValidationLevel.INFO,
                valid=True,
                message="Semantic validation passed",
                details={'checks_performed': ['ranges', 'connectivity']},
                timestamp=time.time()
            )
            
        except Exception as e:
            return ValidationResult(
                layer=ValidationLayer.SEMANTIC,
                level=ValidationLevel.CRITICAL,
                valid=False,
                message=f"Semantic validation error: {str(e)}",
                details={'error': str(e)},
                timestamp=time.time()
            )


class CoherenceValidator:
    """Coherence validation layer"""
    
    def __init__(self):
        self.coherence_thresholds = {
            'critical': 0.3,
            'warning': 0.5,
            'good': 0.7
        }
        
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate schema coherence"""
        try:
            shi = data.get('shi', 0.5)
            scup = data.get('scup', 0.5)
            
            # Calculate coherence score
            coherence_score = (shi + scup) / 2.0
            
            # Assess coherence level
            if coherence_score < self.coherence_thresholds['critical']:
                level = ValidationLevel.CRITICAL
                message = f"Critical coherence issue: score {coherence_score:.3f}"
            elif coherence_score < self.coherence_thresholds['warning']:
                level = ValidationLevel.WARNING
                message = f"Low coherence warning: score {coherence_score:.3f}"
            elif coherence_score < self.coherence_thresholds['good']:
                level = ValidationLevel.INFO
                message = f"Moderate coherence: score {coherence_score:.3f}"
            else:
                level = ValidationLevel.INFO
                message = f"Good coherence: score {coherence_score:.3f}"
            
            # Check for coherence mismatches
            shi_scup_diff = abs(shi - scup)
            if shi_scup_diff > 0.3:
                level = ValidationLevel.WARNING
                message += f". SHI-SCUP mismatch: {shi_scup_diff:.3f}"
            
            return ValidationResult(
                layer=ValidationLayer.COHERENCE,
                level=level,
                valid=coherence_score >= self.coherence_thresholds['critical'],
                message=message,
                details={
                    'coherence_score': coherence_score,
                    'shi': shi,
                    'scup': scup,
                    'shi_scup_diff': shi_scup_diff
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return ValidationResult(
                layer=ValidationLayer.COHERENCE,
                level=ValidationLevel.CRITICAL,
                valid=False,
                message=f"Coherence validation error: {str(e)}",
                details={'error': str(e)},
                timestamp=time.time()
            )


class PerformanceValidator:
    """Performance validation layer"""
    
    def __init__(self):
        self.max_nodes = 10000
        self.max_edges = 50000
        self.max_processing_time = 1.0  # seconds
        
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate performance characteristics"""
        try:
            issues = []
            
            # Check node count
            node_count = len(data.get('nodes', []))
            if node_count > self.max_nodes:
                issues.append(f"Node count {node_count} exceeds maximum {self.max_nodes}")
            
            # Check edge count
            edge_count = len(data.get('edges', []))
            if edge_count > self.max_edges:
                issues.append(f"Edge count {edge_count} exceeds maximum {self.max_edges}")
            
            # Check for performance warnings
            if node_count > self.max_nodes * 0.8:
                issues.append(f"Node count {node_count} approaching limit")
            
            if edge_count > self.max_edges * 0.8:
                issues.append(f"Edge count {edge_count} approaching limit")
            
            # Calculate complexity score
            complexity = (node_count / self.max_nodes) + (edge_count / self.max_edges)
            
            if issues:
                level = ValidationLevel.ERROR if complexity > 1.0 else ValidationLevel.WARNING
                return ValidationResult(
                    layer=ValidationLayer.PERFORMANCE,
                    level=level,
                    valid=complexity <= 1.0,
                    message=f"Performance validation found {len(issues)} issues",
                    details={
                        'issues': issues,
                        'node_count': node_count,
                        'edge_count': edge_count,
                        'complexity_score': complexity
                    },
                    timestamp=time.time()
                )
            
            return ValidationResult(
                layer=ValidationLayer.PERFORMANCE,
                level=ValidationLevel.INFO,
                valid=True,
                message="Performance validation passed",
                details={
                    'node_count': node_count,
                    'edge_count': edge_count,
                    'complexity_score': complexity
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return ValidationResult(
                layer=ValidationLayer.PERFORMANCE,
                level=ValidationLevel.CRITICAL,
                valid=False,
                message=f"Performance validation error: {str(e)}",
                details={'error': str(e)},
                timestamp=time.time()
            )


class SchemaValidator:
    """
    Multi-layer schema validation framework
    
    Provides comprehensive validation across syntax, semantic, coherence,
    and performance layers for complete schema integrity checking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize validation layers
        self.validators = {
            'syntax': SyntaxValidator(),
            'semantic': SemanticValidator(),
            'coherence': CoherenceValidator(),
            'performance': PerformanceValidator()
        }
        
        # Validation settings
        self.fail_fast = self.config.get('fail_fast', False)
        self.layers_enabled = self.config.get('layers_enabled', list(self.validators.keys()))
        
        # Performance tracking
        self.validation_count = 0
        self.total_validation_time = 0.0
        self.error_count = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        print("[SchemaValidator] 🔍 Multi-layer validation framework initialized")
        
    def validate_schema_snapshot(self, data: Dict[str, Any]) -> ValidationSummary:
        """
        Perform complete multi-layer validation of schema snapshot
        
        Args:
            data: Schema snapshot data to validate
            
        Returns:
            ValidationSummary: Complete validation results across all layers
        """
        start_time = time.time()
        results = []
        
        try:
            with self.lock:
                # Perform validation across enabled layers
                for layer_name in self.layers_enabled:
                    if layer_name in self.validators:
                        validator = self.validators[layer_name]
                        
                        # Run validation
                        result = validator.validate(data)
                        results.append(result)
                        
                        # Check if we should fail fast
                        if self.fail_fast and not result.valid and result.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]:
                            break
                
                # Calculate summary statistics
                total_checks = len(results)
                passed_checks = sum(1 for r in results if r.valid)
                failed_checks = total_checks - passed_checks
                warnings = sum(1 for r in results if r.level == ValidationLevel.WARNING)
                errors = sum(1 for r in results if r.level == ValidationLevel.ERROR)
                critical_issues = sum(1 for r in results if r.level == ValidationLevel.CRITICAL)
                
                # Overall validity
                overall_valid = all(r.valid for r in results) and critical_issues == 0
                
                # Update performance tracking
                validation_time = time.time() - start_time
                self.validation_count += 1
                self.total_validation_time += validation_time
                
                return ValidationSummary(
                    overall_valid=overall_valid,
                    total_checks=total_checks,
                    passed_checks=passed_checks,
                    failed_checks=failed_checks,
                    warnings=warnings,
                    errors=errors,
                    critical_issues=critical_issues,
                    validation_time=validation_time,
                    results=results
                )
                
        except Exception as e:
            self.error_count += 1
            print(f"[SchemaValidator] ❌ Validation error: {str(e)}")
            
            # Return error summary
            error_result = ValidationResult(
                layer=ValidationLayer.SYNTAX,
                level=ValidationLevel.CRITICAL,
                valid=False,
                message=f"Validation framework error: {str(e)}",
                details={'error': str(e)},
                timestamp=time.time()
            )
            
            return ValidationSummary(
                overall_valid=False,
                total_checks=1,
                passed_checks=0,
                failed_checks=1,
                warnings=0,
                errors=0,
                critical_issues=1,
                validation_time=time.time() - start_time,
                results=[error_result]
            )
    
    def validate_layer(self, data: Dict[str, Any], layer: str) -> ValidationResult:
        """Validate specific layer only"""
        if layer not in self.validators:
            return ValidationResult(
                layer=ValidationLayer.SYNTAX,
                level=ValidationLevel.ERROR,
                valid=False,
                message=f"Unknown validation layer: {layer}",
                details={'available_layers': list(self.validators.keys())},
                timestamp=time.time()
            )
        
        return self.validators[layer].validate(data)
    
    def add_custom_validator(self, layer_name: str, validator: Any):
        """Add custom validation layer"""
        with self.lock:
            self.validators[layer_name] = validator
            if layer_name not in self.layers_enabled:
                self.layers_enabled.append(layer_name)
            print(f"[SchemaValidator] Added custom validator: {layer_name}")
    
    def enable_layer(self, layer_name: str):
        """Enable specific validation layer"""
        with self.lock:
            if layer_name in self.validators and layer_name not in self.layers_enabled:
                self.layers_enabled.append(layer_name)
    
    def disable_layer(self, layer_name: str):
        """Disable specific validation layer"""
        with self.lock:
            if layer_name in self.layers_enabled:
                self.layers_enabled.remove(layer_name)
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation performance statistics"""
        with self.lock:
            avg_time = self.total_validation_time / max(1, self.validation_count)
            
            return {
                'validation_count': self.validation_count,
                'total_time': self.total_validation_time,
                'average_time': avg_time,
                'error_count': self.error_count,
                'enabled_layers': self.layers_enabled.copy(),
                'available_layers': list(self.validators.keys())
            }


def create_schema_validator(config: Optional[Dict[str, Any]] = None) -> SchemaValidator:
    """Create configured schema validator"""
    return SchemaValidator(config)


if __name__ == "__main__":
    # Example usage
    validator = create_schema_validator()
    
    # Test data
    test_data = {
        'nodes': [
            {'id': 'node1', 'health': 0.8},
            {'id': 'node2', 'health': 0.6}
        ],
        'edges': [
            {'id': 'edge1', 'source': 'node1', 'target': 'node2', 'weight': 0.7}
        ],
        'signals': {'pressure': 0.3, 'drift': 0.2},
        'shi': 0.75,
        'scup': 0.68
    }
    
    # Validate
    summary = validator.validate_schema_snapshot(test_data)
    
    print(f"Validation Summary:")
    print(f"Overall Valid: {summary.overall_valid}")
    print(f"Pass Rate: {summary.get_pass_rate():.2%}")
    print(f"Warnings: {summary.warnings}, Errors: {summary.errors}")
    
    for result in summary.results:
        print(f"  {result.layer.value}: {result.message}")
