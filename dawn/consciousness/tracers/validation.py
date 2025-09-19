"""
Tracer System Validation

Test vectors and validation framework for the DAWN tracer ecosystem.
Provides functionality to validate tracer behavior, test spawn conditions,
and verify ecosystem balance.
"""

from typing import Dict, Any, List, Tuple
import logging
from . import create_tracer_ecosystem, TracerType, AlertSeverity
from .integration import create_standard_integration

logger = logging.getLogger(__name__)


class TracerValidationSuite:
    """
    Comprehensive validation suite for the tracer ecosystem.
    """
    
    def __init__(self):
        self.test_results = []
        self.ecosystem = create_tracer_ecosystem(nutrient_budget=50.0)  # Smaller budget for testing
        self.integration = create_standard_integration(self.ecosystem)
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests and return comprehensive results"""
        logger.info("Starting tracer ecosystem validation...")
        
        test_suites = [
            self.test_spawn_conditions(),
            self.test_tracer_lifecycle(),
            self.test_ecosystem_balance(),
            self.test_alert_generation(),
            self.test_nutrient_management(),
            self.test_integration_functionality()
        ]
        
        passed_tests = sum(1 for suite in test_suites if suite['passed'])
        total_tests = len(test_suites)
        
        summary = {
            'total_test_suites': total_tests,
            'passed_test_suites': passed_tests,
            'success_rate': passed_tests / total_tests,
            'test_results': test_suites,
            'overall_status': 'PASS' if passed_tests == total_tests else 'FAIL'
        }
        
        logger.info(f"Validation complete: {passed_tests}/{total_tests} test suites passed")
        return summary
        
    def test_spawn_conditions(self) -> Dict[str, Any]:
        """Test tracer spawn condition logic"""
        test_name = "Spawn Conditions Test"
        test_cases = []
        
        # Test 1: Crow spawn on entropy spike
        context = {'entropy': 0.8, 'tick_id': 1, 'timestamp': 1.0}
        spawn_result = self.ecosystem.evaluate_spawning(1, context)
        crow_spawned = any(tracer.tracer_type == TracerType.CROW for tracer in spawn_result)
        test_cases.append({
            'name': 'Crow entropy spike spawn',
            'passed': crow_spawned,
            'details': f"Spawned {len(spawn_result)} tracers, Crow present: {crow_spawned}"
        })
        
        # Test 2: Spider spawn on schema tension
        context = {
            'schema_edges': [{'tension': 0.7, 'id': 'test-edge'}],
            'tick_id': 2, 'timestamp': 2.0
        }
        spawn_result = self.ecosystem.evaluate_spawning(2, context)
        spider_spawned = any(tracer.tracer_type == TracerType.SPIDER for tracer in spawn_result)
        test_cases.append({
            'name': 'Spider tension spawn',
            'passed': spider_spawned,
            'details': f"Spawned {len(spawn_result)} tracers, Spider present: {spider_spawned}"
        })
        
        # Test 3: Beetle spawn on soot accumulation
        context = {'soot_ratio': 0.5, 'tick_id': 3, 'timestamp': 3.0}
        spawn_result = self.ecosystem.evaluate_spawning(3, context)
        beetle_spawned = any(tracer.tracer_type == TracerType.BEETLE for tracer in spawn_result)
        test_cases.append({
            'name': 'Beetle soot spawn',
            'passed': beetle_spawned,
            'details': f"Spawned {len(spawn_result)} tracers, Beetle present: {beetle_spawned}"
        })
        
        passed_cases = sum(1 for case in test_cases if case['passed'])
        
        return {
            'test_name': test_name,
            'passed': passed_cases == len(test_cases),
            'passed_cases': passed_cases,
            'total_cases': len(test_cases),
            'test_cases': test_cases
        }
        
    def test_tracer_lifecycle(self) -> Dict[str, Any]:
        """Test complete tracer lifecycle"""
        test_name = "Tracer Lifecycle Test"
        test_cases = []
        
        # Create a Crow tracer and test its lifecycle
        from .crow_tracer import CrowTracer
        crow = CrowTracer()
        
        # Test spawn
        context = {'entropy': 0.8, 'tick_id': 1, 'timestamp': 1.0}
        crow.spawn(1, context)
        spawn_success = crow.status.value == 'active' and crow.spawn_tick == 1
        test_cases.append({
            'name': 'Tracer spawn',
            'passed': spawn_success,
            'details': f"Status: {crow.status.value}, Spawn tick: {crow.spawn_tick}"
        })
        
        # Test observation
        reports = crow.tick(2, context)
        observation_success = isinstance(reports, list)
        test_cases.append({
            'name': 'Tracer observation',
            'passed': observation_success,
            'details': f"Generated {len(reports)} reports"
        })
        
        # Test retirement conditions
        retirement_context = {'tick_id': 10, 'timestamp': 10.0}
        should_retire = crow.should_retire(retirement_context)
        retirement_success = should_retire  # Crow should retire after several ticks
        test_cases.append({
            'name': 'Tracer retirement',
            'passed': retirement_success,
            'details': f"Should retire: {should_retire}, Age: {crow.get_age(10)}"
        })
        
        passed_cases = sum(1 for case in test_cases if case['passed'])
        
        return {
            'test_name': test_name,
            'passed': passed_cases == len(test_cases),
            'passed_cases': passed_cases,
            'total_cases': len(test_cases),
            'test_cases': test_cases
        }
        
    def test_ecosystem_balance(self) -> Dict[str, Any]:
        """Test ecosystem balance and resource management"""
        test_name = "Ecosystem Balance Test"
        test_cases = []
        
        # Test nutrient budget management
        initial_budget = self.ecosystem.get_available_budget()
        budget_test = initial_budget > 0
        test_cases.append({
            'name': 'Initial budget allocation',
            'passed': budget_test,
            'details': f"Available budget: {initial_budget}"
        })
        
        # Test rate limiting
        high_spawn_context = {
            'entropy': 0.9, 'soot_ratio': 0.6, 'schema_edges': [{'tension': 0.8}],
            'tick_id': 5, 'timestamp': 5.0
        }
        
        spawn_result = self.ecosystem.evaluate_spawning(5, high_spawn_context)
        rate_limit_test = len(spawn_result) <= 20  # Should not exceed reasonable limits
        test_cases.append({
            'name': 'Rate limiting',
            'passed': rate_limit_test,
            'details': f"Spawned {len(spawn_result)} tracers under high demand"
        })
        
        # Test ecosystem tick processing
        tick_summary = self.ecosystem.tick(6, high_spawn_context)
        tick_test = 'ecosystem_state' in tick_summary
        test_cases.append({
            'name': 'Ecosystem tick processing',
            'passed': tick_test,
            'details': f"Tick summary keys: {list(tick_summary.keys())}"
        })
        
        passed_cases = sum(1 for case in test_cases if case['passed'])
        
        return {
            'test_name': test_name,
            'passed': passed_cases == len(test_cases),
            'passed_cases': passed_cases,
            'total_cases': len(test_cases),
            'test_cases': test_cases
        }
        
    def test_alert_generation(self) -> Dict[str, Any]:
        """Test alert generation and severity levels"""
        test_name = "Alert Generation Test"
        test_cases = []
        
        # Test critical alert generation
        critical_context = {
            'active_blooms': [{'entropy': 0.9, 'id': 'critical-bloom', 'intensity': 0.95}],
            'soot_fragments': [{'volatility': 0.8, 'id': f'volatile-{i}'} for i in range(5)],
            'tick_id': 7, 'timestamp': 7.0
        }
        
        tick_summary = self.ecosystem.tick(7, critical_context)
        reports = tick_summary.get('reports', [])
        critical_alerts = [r for r in reports if r.get('severity') == 'critical']
        
        critical_test = len(critical_alerts) > 0
        test_cases.append({
            'name': 'Critical alert generation',
            'passed': critical_test,
            'details': f"Generated {len(critical_alerts)} critical alerts from {len(reports)} total reports"
        })
        
        # Test alert metadata completeness
        if reports:
            sample_report = reports[0]
            metadata_test = all(key in sample_report for key in ['tracer_id', 'tracer_type', 'tick_id', 'metadata'])
            test_cases.append({
                'name': 'Alert metadata completeness',
                'passed': metadata_test,
                'details': f"Report keys: {list(sample_report.keys())}"
            })
        else:
            test_cases.append({
                'name': 'Alert metadata completeness',
                'passed': False,
                'details': "No reports generated to test"
            })
        
        passed_cases = sum(1 for case in test_cases if case['passed'])
        
        return {
            'test_name': test_name,
            'passed': passed_cases == len(test_cases),
            'passed_cases': passed_cases,
            'total_cases': len(test_cases),
            'test_cases': test_cases
        }
        
    def test_nutrient_management(self) -> Dict[str, Any]:
        """Test nutrient allocation and recycling"""
        test_name = "Nutrient Management Test"
        test_cases = []
        
        # Test nutrient consumption
        initial_usage = self.ecosystem.current_nutrient_usage
        consumption_context = {'entropy': 0.7, 'tick_id': 8, 'timestamp': 8.0}
        
        self.ecosystem.tick(8, consumption_context)
        final_usage = self.ecosystem.current_nutrient_usage
        
        consumption_test = final_usage >= initial_usage  # Should consume or maintain
        test_cases.append({
            'name': 'Nutrient consumption tracking',
            'passed': consumption_test,
            'details': f"Usage: {initial_usage} -> {final_usage}"
        })
        
        # Test budget limits
        over_budget_context = {
            'entropy': 0.95, 'soot_ratio': 0.8, 'pressure': 0.9,
            'schema_edges': [{'tension': 0.9} for _ in range(10)],
            'tick_id': 9, 'timestamp': 9.0
        }
        
        for _ in range(5):  # Try to spawn many expensive tracers
            self.ecosystem.tick(_, over_budget_context)
        
        budget_usage = self.ecosystem.current_nutrient_usage / self.ecosystem.nutrient_budget
        budget_test = budget_usage <= 1.1  # Allow small overflow but not excessive
        test_cases.append({
            'name': 'Budget enforcement',
            'passed': budget_test,
            'details': f"Budget utilization: {budget_usage:.1%}"
        })
        
        passed_cases = sum(1 for case in test_cases if case['passed'])
        
        return {
            'test_name': test_name,
            'passed': passed_cases == len(test_cases),
            'passed_cases': passed_cases,
            'total_cases': len(test_cases),
            'test_cases': test_cases
        }
        
    def test_integration_functionality(self) -> Dict[str, Any]:
        """Test integration with DAWN subsystems"""
        test_name = "Integration Functionality Test"
        test_cases = []
        
        # Test context building
        dawn_state = {
            'current_tick': 10,
            'timestamp': 10.0,
            'consciousness': {'entropy_level': 0.6, 'cognitive_pressure': 0.4},
            'memory': {'pressure': 0.3, 'active_blooms': []},
            'schema': {'avg_coherence': 0.8, 'drift_magnitude': 0.1}
        }
        
        tracer_context = self.integration.build_tracer_context(dawn_state)
        context_test = all(key in tracer_context for key in ['tick_id', 'entropy', 'pressure'])
        test_cases.append({
            'name': 'Context building',
            'passed': context_test,
            'details': f"Context keys: {list(tracer_context.keys())[:10]}..."  # Show first 10 keys
        })
        
        # Test report processing
        from .base_tracer import TracerReport, TracerType
        sample_report = TracerReport(
            tracer_id='test-crow',
            tracer_type=TracerType.CROW,
            tick_id=10,
            timestamp=10.0,
            severity=AlertSeverity.WARN,
            report_type='bloom_anomaly',
            metadata={'bloom_id': 'test-bloom', 'entropy_level': 0.8}
        )
        
        action_summary = self.integration.process_tracer_reports([sample_report], dawn_state)
        processing_test = 'alerts_processed' in action_summary and action_summary['alerts_processed'] > 0
        test_cases.append({
            'name': 'Report processing',
            'passed': processing_test,
            'details': f"Action summary: {action_summary}"
        })
        
        passed_cases = sum(1 for case in test_cases if case['passed'])
        
        return {
            'test_name': test_name,
            'passed': passed_cases == len(test_cases),
            'passed_cases': passed_cases,
            'total_cases': len(test_cases),
            'test_cases': test_cases
        }


def validate_tracer_implementation() -> Dict[str, Any]:
    """
    Run comprehensive validation of the tracer system implementation.
    
    Returns:
        dict: Validation results
    """
    validator = TracerValidationSuite()
    return validator.run_all_tests()


def run_quick_tracer_test() -> bool:
    """
    Run a quick smoke test of the tracer system.
    
    Returns:
        bool: True if basic functionality works
    """
    try:
        # Create ecosystem
        ecosystem = create_tracer_ecosystem(nutrient_budget=20.0)
        
        # Test basic spawning
        context = {'entropy': 0.8, 'tick_id': 1, 'timestamp': 1.0}
        spawned = ecosystem.evaluate_spawning(1, context)
        
        # Test tick processing
        tick_summary = ecosystem.tick(2, context)
        
        # Basic checks
        return (len(spawned) > 0 and 
                'ecosystem_state' in tick_summary and
                tick_summary['ecosystem_state']['active_tracers'] >= 0)
                
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        return False


if __name__ == "__main__":
    # Run validation when executed directly
    results = validate_tracer_implementation()
    print(f"\nTracer System Validation Results:")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Passed: {results['passed_test_suites']}/{results['total_test_suites']} test suites")
    
    for test_result in results['test_results']:
        status = "✓" if test_result['passed'] else "✗"
        print(f"{status} {test_result['test_name']}: {test_result['passed_cases']}/{test_result['total_cases']} cases passed")
