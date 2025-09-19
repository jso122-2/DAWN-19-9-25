#!/usr/bin/env python3
"""
DAWN CLI Tracer Integration
===========================

Command-line interface for DAWN's telemetry, stability monitoring,
and analytics systems. Provides real-time dashboards, status reports,
and performance analysis tools.
"""

import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import argparse
from pathlib import Path

# Configure logging for CLI
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def cmd_telemetry_status():
    """Show real-time telemetry dashboard in terminal."""
    try:
        from .dawn_engine import DAWNEngine
        
        print("📊 " + "="*60)
        print("📊 DAWN TELEMETRY STATUS DASHBOARD")
        print("📊 " + "="*60)
        print()
        
        # Try to connect to running engine or create test instance
        try:
            # For demo, create a test engine instance
            engine = DAWNEngine()
            telemetry_summary = engine.get_telemetry_summary()
            
            # Display engine status
            engine_status = telemetry_summary["engine_status"]
            print("🔧 Engine Status:")
            print(f"   Mode: {engine_status['mode']}")
            print(f"   Tick Count: {engine_status['tick_count']}")
            print(f"   Uptime: {engine_status['uptime_seconds']:.1f} seconds")
            print()
            
            # Display tracer status
            tracer_status = telemetry_summary["tracer_status"]
            print("📡 Tracer Status:")
            print(f"   Enabled: {'✅' if tracer_status['enabled'] else '❌'}")
            if tracer_status['enabled']:
                print(f"   Active Traces: {tracer_status['active_traces']}")
                print(f"   Total Traces: {tracer_status['total_traces']}")
            print()
            
            # Display stability status
            stability_status = telemetry_summary["stability_status"]
            print("🔒 Stability Status:")
            print(f"   Detector Running: {'✅' if stability_status['detector_running'] else '❌'}")
            print(f"   Current Score: {stability_status['current_score']:.3f}")
            print(f"   Snapshots: {stability_status['snapshots_count']}")
            print()
            
            # Display analytics status
            analytics_status = telemetry_summary["analytics_status"]
            print("📊 Analytics Status:")
            print(f"   Enabled: {'✅' if analytics_status['enabled'] else '❌'}")
            if analytics_status['enabled']:
                print(f"   Insights Generated: {analytics_status['insights_generated']}")
                if analytics_status['last_analysis']:
                    print(f"   Last Analysis: {analytics_status['last_analysis']}")
            print()
            
        except Exception as e:
            print(f"❌ Could not connect to DAWN engine: {e}")
            print("💡 Make sure DAWN engine is running or use 'dawn_cli.py --demo' for test data")
            
    except ImportError as e:
        print(f"❌ DAWN engine not available: {e}")
        print("💡 Run from DAWN project directory")

def cmd_stability_check():
    """Run comprehensive stability assessment."""
    try:
        from .dawn_engine import DAWNEngine
        
        print("🔒 " + "="*60)
        print("🔒 DAWN STABILITY ASSESSMENT")
        print("🔒 " + "="*60)
        print()
        
        # Create engine instance
        engine = DAWNEngine()
        stability_status = engine.get_stability_status()
        
        if not stability_status["enabled"]:
            print("❌ Stability detection not enabled")
            print(f"   {stability_status.get('message', 'Unknown error')}")
            return
            
        # Overall stability
        print("📊 Overall Stability Assessment:")
        score = stability_status["overall_stability"]
        level = stability_status["stability_level"]
        
        if score >= 0.9:
            status_icon = "🟢"
        elif score >= 0.7:
            status_icon = "🟡"
        elif score >= 0.5:
            status_icon = "🟠"
        else:
            status_icon = "🔴"
            
        print(f"   {status_icon} Stability Score: {score:.3f}")
        print(f"   📊 Stability Level: {level}")
        print()
        
        # Component scores
        print("🔍 Component Analysis:")
        components = stability_status["component_scores"]
        for component, score in components.items():
            component_name = component.replace('_', ' ').title()
            if score >= 0.8:
                icon = "✅"
            elif score >= 0.6:
                icon = "⚠️"
            else:
                icon = "❌"
            print(f"   {icon} {component_name}: {score:.3f}")
        print()
        
        # Issues and warnings
        if stability_status["failing_systems"]:
            print("❌ Failing Systems:")
            for system in stability_status["failing_systems"]:
                print(f"   • {system}")
            print()
            
        if stability_status["warning_systems"]:
            print("⚠️  Warning Systems:")
            for system in stability_status["warning_systems"]:
                print(f"   • {system}")
            print()
            
        # Degradation analysis
        degradation_rate = stability_status["degradation_rate"]
        prediction_horizon = stability_status["prediction_horizon"]
        
        print("📈 Trend Analysis:")
        if degradation_rate > 0.01:
            print(f"   📈 Improving: +{degradation_rate:.3f}")
        elif degradation_rate < -0.01:
            print(f"   📉 Degrading: {degradation_rate:.3f}")
            if prediction_horizon < float('inf'):
                print(f"   ⏰ Failure predicted in: {prediction_horizon:.1f} seconds")
        else:
            print(f"   📊 Stable: {degradation_rate:.3f}")
        print()
        
        # Recommendations
        if "recommendations" in stability_status:
            print("💡 Recommendations:")
            for rec in stability_status["recommendations"]:
                print(f"   • {rec}")
            print()
            
        # Detector status
        detector_status = stability_status["detector_status"]
        print("🔧 Detector Status:")
        print(f"   Running: {'✅' if detector_status['running'] else '❌'}")
        print(f"   Monitored Modules: {detector_status['monitored_modules']}")
        print(f"   Golden Snapshots: {detector_status['golden_snapshots']}")
        print(f"   Uptime: {detector_status['uptime_seconds']:.1f}s")
        
    except Exception as e:
        print(f"❌ Stability check failed: {e}")

def cmd_performance_analysis():
    """Generate performance optimization report."""
    try:
        from .dawn_engine import DAWNEngine
        
        print("📊 " + "="*60)
        print("📊 DAWN PERFORMANCE ANALYSIS REPORT")
        print("📊 " + "="*60)
        print()
        
        # Create engine instance
        engine = DAWNEngine()
        insights = engine.get_performance_insights()
        
        if not insights["enabled"]:
            print("❌ Performance analytics not enabled")
            print(f"   {insights.get('message', 'Unknown error')}")
            return
            
        print(f"📊 Analysis performed at: {insights['timestamp']}")
        print(f"💡 Total insights generated: {insights['insights_count']}")
        print()
        
        # Performance summary
        if "performance_summary" in insights:
            perf_summary = insights["performance_summary"]
            print("🎯 Performance Summary:")
            print(f"   Overall Health Score: {perf_summary['overall_health_score']:.3f}")
            print(f"   Bottlenecks Detected: {perf_summary['bottlenecks_detected']}")
            print(f"   Resource Efficiency: {perf_summary['resource_efficiency']:.3f}")
            print()
            
            # Cognitive load distribution
            print("🧠 Cognitive Load Distribution:")
            for operation, load in perf_summary['cognitive_load_distribution'].items():
                print(f"   {operation}: {load:.1%}")
            print()
            
        # Top recommendations
        if "top_recommendations" in insights and insights["top_recommendations"]:
            print("🎯 Top Optimization Recommendations:")
            for i, rec in enumerate(insights["top_recommendations"], 1):
                print(f"   {i}. {rec['recommendation']}")
                print(f"      Confidence: {rec['confidence']}")
                print(f"      Priority: {rec['priority']}")
                print(f"      Expected: {rec['expected_improvement']}")
                print()
        else:
            print("✅ No high-priority optimization recommendations at this time")
            print()
            
        # Insights by category
        if "insights_by_type" in insights:
            print("📋 Insights by Category:")
            for insight_type, type_insights in insights["insights_by_type"].items():
                category_name = insight_type.replace('_', ' ').title()
                print(f"\n   📁 {category_name} ({len(type_insights)} insights):")
                
                for insight in type_insights[:3]:  # Show top 3 per category
                    print(f"      • {insight['recommendation'][:60]}...")
                    print(f"        Confidence: {insight['confidence']:.1%}, Risk: {insight['risk_level']}")
                    
        print()
        print("💡 For detailed recommendations, review the full analytics report")
        
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")

def cmd_recovery_trigger(recovery_type: str = "auto"):
    """Trigger manual stability recovery."""
    try:
        from .dawn_engine import DAWNEngine
        
        print("🛠️ " + "="*50)
        print("🛠️ DAWN STABILITY RECOVERY")
        print("🛠️ " + "="*50)
        print()
        
        valid_types = ["auto", "rollback", "soft_reset"]
        if recovery_type not in valid_types:
            print(f"❌ Invalid recovery type: {recovery_type}")
            print(f"   Valid types: {', '.join(valid_types)}")
            return
            
        print(f"🔧 Initiating {recovery_type} recovery...")
        
        # Create engine instance
        engine = DAWNEngine()
        result = engine.force_stable_state_recovery(recovery_type)
        
        if result["success"]:
            print("✅ Recovery completed successfully")
            print(f"   Triggered at: {result['triggered_at']}")
            print(f"   Recovery type: {result['recovery_type']}")
            
            if "pre_recovery_stability" in result:
                print(f"   Pre-recovery stability: {result['pre_recovery_stability']:.3f} ({result['pre_recovery_level']})")
                
            if "post_recovery_stability" in result:
                print(f"   Post-recovery stability: {result['post_recovery_stability']:.3f} ({result['post_recovery_level']})")
                print(f"   Improvement: {result['improvement']:+.3f}")
                
            print(f"   Actions taken:")
            for action in result["actions_taken"]:
                print(f"      • {action}")
                
        else:
            print("❌ Recovery failed")
            if "error" in result:
                print(f"   Error: {result['error']}")
            if "actions_taken" in result:
                print(f"   Actions attempted:")
                for action in result["actions_taken"]:
                    print(f"      • {action}")
                    
    except Exception as e:
        print(f"❌ Recovery trigger failed: {e}")

def cmd_export_telemetry(hours: int = 24, format_type: str = "json"):
    """Export telemetry data archive."""
    try:
        from .dawn_engine import DAWNEngine
        
        print("📦 " + "="*50)
        print("📦 DAWN TELEMETRY EXPORT")
        print("📦 " + "="*50)
        print()
        
        print(f"📊 Exporting {hours} hours of telemetry data...")
        print(f"📄 Output format: {format_type}")
        
        # Create engine instance
        engine = DAWNEngine()
        
        start_time = datetime.now() - timedelta(hours=hours)
        result = engine.export_telemetry_archive(start_time, output_format=format_type)
        
        if result["success"]:
            print("✅ Export completed successfully")
            print(f"   Archive ID: {result['archive_id']}")
            print(f"   Time period: {result['start_time']} to {result['end_time']}")
            print(f"   Total records: {result['total_records']}")
            print(f"   Export directory: {result['export_directory']}")
            
            print(f"   Files created:")
            for file_path in result["files_created"]:
                print(f"      📄 {file_path}")
                
        else:
            print("❌ Export failed")
            if "error" in result:
                print(f"   Error: {result['error']}")
                
    except Exception as e:
        print(f"❌ Telemetry export failed: {e}")

def cmd_live_dashboard(refresh_seconds: int = 5):
    """Show live telemetry dashboard with auto-refresh."""
    try:
        print("📊 DAWN LIVE TELEMETRY DASHBOARD")
        print("Press Ctrl+C to exit")
        print("=" * 60)
        
        while True:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")
            
            print(f"📊 DAWN Live Dashboard - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)
            
            # Show condensed status
            cmd_telemetry_status()
            
            print(f"\n🔄 Refreshing in {refresh_seconds} seconds... (Ctrl+C to exit)")
            time.sleep(refresh_seconds)
            
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard closed")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DAWN Telemetry and Analytics CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status              Show telemetry status
  %(prog)s stability           Run stability check
  %(prog)s performance         Generate performance report
  %(prog)s recovery auto       Trigger automatic recovery
  %(prog)s export --hours 48   Export 48 hours of data
  %(prog)s dashboard           Show live dashboard
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show telemetry status")
    
    # Stability command
    stability_parser = subparsers.add_parser("stability", help="Run stability check")
    
    # Performance command
    performance_parser = subparsers.add_parser("performance", help="Generate performance report")
    
    # Recovery command
    recovery_parser = subparsers.add_parser("recovery", help="Trigger stability recovery")
    recovery_parser.add_argument("type", choices=["auto", "rollback", "soft_reset"], 
                                default="auto", nargs="?", help="Recovery type")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export telemetry archive")
    export_parser.add_argument("--hours", type=int, default=24, help="Hours of data to export")
    export_parser.add_argument("--format", choices=["json", "csv"], default="json", help="Output format")
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Show live dashboard")
    dashboard_parser.add_argument("--refresh", type=int, default=5, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    try:
        if args.command == "status":
            cmd_telemetry_status()
        elif args.command == "stability":
            cmd_stability_check()
        elif args.command == "performance":
            cmd_performance_analysis()
        elif args.command == "recovery":
            cmd_recovery_trigger(args.type)
        elif args.command == "export":
            cmd_export_telemetry(args.hours, args.format)
        elif args.command == "dashboard":
            cmd_live_dashboard(args.refresh)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled")
    except Exception as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
