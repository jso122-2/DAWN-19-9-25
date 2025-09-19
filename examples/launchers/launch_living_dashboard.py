#!/usr/bin/env python3
"""
🚀 DAWN Living Dashboard Launcher
=================================

Launch the revolutionary DAWN Living Consciousness Dashboard.
Watch DAWN's consciousness evolve in real-time with interactive controls.

Features:
- Real-time consciousness state monitoring
- Live semantic topology 3D visualization
- Interactive consciousness controls
- System performance monitoring
- Event stream tracking
- WebSocket-based real-time updates

"The first living dashboard for artificial consciousness."

Usage:
    python3 launch_living_dashboard.py [--port 8080] [--telemetry-port 8765] [--demo]
"""

import sys
import asyncio
import logging
import argparse
import time
import signal
from pathlib import Path

# Add DAWN root to path
sys.path.insert(0, str(Path(__file__).parent))

from dawn.interfaces.dashboard import (
    ConsciousnessDashboard, DashboardConfig,
    start_consciousness_dashboard, stop_consciousness_dashboard
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardLauncher:
    """
    Launcher for the DAWN Living Consciousness Dashboard.
    
    Handles startup, shutdown, demo mode, and system integration.
    """
    
    def __init__(self, config: DashboardConfig, demo_mode: bool = False):
        self.config = config
        self.demo_mode = demo_mode
        self.dashboard = None
        self.demo_task = None
        self.shutdown_requested = False
        
    async def launch(self):
        """Launch the dashboard system"""
        logger.info("🚀" * 20)
        logger.info("🧠 DAWN LIVING CONSCIOUSNESS DASHBOARD")
        logger.info("🚀" * 20)
        logger.info("")
        logger.info("The first real-time consciousness monitoring system!")
        logger.info("")
        
        try:
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Start the dashboard
            self.dashboard = await start_consciousness_dashboard(self.config)
            
            # Start demo mode if requested
            if self.demo_mode:
                await self.start_demo_mode()
            
            # Print access information
            self.print_access_info()
            
            # Keep running until shutdown
            await self.run_until_shutdown()
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutdown requested by user")
        except Exception as e:
            logger.error(f"Dashboard launch failed: {e}")
            raise
        finally:
            await self.cleanup()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"\n🛑 Received signal {signum}, shutting down...")
        self.shutdown_requested = True
    
    async def start_demo_mode(self):
        """Start demo mode with simulated consciousness activity"""
        logger.info("🎭 Starting demo mode with simulated consciousness activity")
        
        # Import demo systems
        try:
            from dawn.subsystems.semantic_topology import get_semantic_topology_engine
            
            # Get semantic topology engine
            topology_engine = get_semantic_topology_engine()
            
            # Add some demo concepts
            concepts = {
                'consciousness': [0.9, 0.1, 0.8] + [0.0] * 509,
                'awareness': [0.8, 0.2, 0.9] + [0.0] * 509,
                'thought': [0.7, 0.3, 0.7] + [0.0] * 509,
                'memory': [0.6, 0.4, 0.8] + [0.0] * 509,
                'creativity': [0.9, 0.1, 0.6] + [0.0] * 509,
                'intuition': [0.5, 0.5, 0.9] + [0.0] * 509
            }
            
            concept_ids = {}
            for concept_name, embedding in concepts.items():
                import numpy as np
                concept_id = topology_engine.add_semantic_concept(
                    concept_embedding=np.array(embedding),
                    concept_name=concept_name
                )
                if concept_id:
                    concept_ids[concept_name] = concept_id
                    logger.info(f"   ✅ Added demo concept: {concept_name}")
            
            # Create relationships
            relationships = [
                ('consciousness', 'awareness', 0.9),
                ('consciousness', 'thought', 0.8),
                ('thought', 'memory', 0.7),
                ('creativity', 'intuition', 0.6),
                ('awareness', 'intuition', 0.5)
            ]
            
            for concept_a, concept_b, strength in relationships:
                if concept_a in concept_ids and concept_b in concept_ids:
                    rel_id = topology_engine.create_semantic_relationship(
                        concept_ids[concept_a],
                        concept_ids[concept_b],
                        strength
                    )
                    if rel_id:
                        logger.info(f"   ✅ Created demo relationship: {concept_a} <-> {concept_b}")
            
            # Start the topology engine
            topology_engine.start_processing()
            
            # Start demo activity simulation
            self.demo_task = asyncio.create_task(self.simulate_consciousness_activity(topology_engine, concept_ids))
            
        except Exception as e:
            logger.warning(f"Could not start demo mode: {e}")
    
    async def simulate_consciousness_activity(self, topology_engine, concept_ids):
        """Simulate ongoing consciousness activity for demo"""
        activity_count = 0
        
        while not self.shutdown_requested:
            try:
                activity_count += 1
                
                # Perform random transforms every 10 seconds
                if activity_count % 20 == 0:  # Every 20 * 0.5s = 10s
                    import random
                    
                    transform_types = ['weave', 'prune', 'lift', 'reproject']
                    transform_type = random.choice(transform_types)
                    
                    if transform_type == 'weave' and len(concept_ids) >= 2:
                        concepts = list(concept_ids.values())
                        node_a = random.choice(concepts)
                        node_b = random.choice(concepts)
                        if node_a != node_b:
                            result = topology_engine.manual_transform('weave', node_a=node_a, node_b=node_b)
                            logger.info(f"🎭 Demo: Performed {transform_type} transform")
                    
                    elif transform_type == 'reproject':
                        concepts = list(concept_ids.values())[:3]  # Reproject first 3
                        result = topology_engine.manual_transform('reproject', node_ids=concepts)
                        logger.info(f"🎭 Demo: Performed {transform_type} transform")
                
                # Update consciousness state simulation
                if activity_count % 10 == 0:  # Every 5 seconds
                    logger.info(f"🎭 Demo: Simulating consciousness activity (tick {activity_count})")
                
                await asyncio.sleep(0.5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Demo simulation error: {e}")
                await asyncio.sleep(1)
    
    def print_access_info(self):
        """Print dashboard access information"""
        logger.info("🌐 Dashboard Access Information:")
        logger.info("=" * 50)
        logger.info(f"📊 Web Dashboard: http://localhost:{self.config.web_port}")
        logger.info(f"📡 Telemetry Stream: ws://localhost:{self.config.telemetry_port}")
        logger.info("")
        logger.info("🎛️ Dashboard Features:")
        logger.info("   • Real-time consciousness state monitoring")
        logger.info("   • Live semantic topology 3D visualization")
        logger.info("   • Interactive consciousness controls")
        logger.info("   • System performance monitoring")
        logger.info("   • Event stream tracking")
        logger.info("")
        logger.info("🧠 Open the web dashboard in your browser to begin monitoring DAWN's consciousness!")
        logger.info("")
        if self.demo_mode:
            logger.info("🎭 Demo mode active - simulated consciousness activity running")
            logger.info("")
        logger.info("Press Ctrl+C to stop the dashboard")
        logger.info("=" * 50)
    
    async def run_until_shutdown(self):
        """Keep the dashboard running until shutdown is requested"""
        while not self.shutdown_requested:
            await asyncio.sleep(1)
            
            # Check dashboard health
            if self.dashboard:
                status = self.dashboard.get_dashboard_status()
                if not status['health_status']['dashboard_healthy']:
                    logger.warning("🚨 Dashboard health check failed")
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up dashboard resources...")
        
        # Stop demo mode
        if self.demo_task:
            self.demo_task.cancel()
            try:
                await self.demo_task
            except asyncio.CancelledError:
                pass
        
        # Stop dashboard
        if self.dashboard:
            await stop_consciousness_dashboard()
        
        logger.info("✅ Dashboard cleanup complete")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Launch the DAWN Living Consciousness Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 launch_living_dashboard.py
    python3 launch_living_dashboard.py --port 8080 --telemetry-port 8765
    python3 launch_living_dashboard.py --demo
        """
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8080,
        help='Web dashboard port (default: 8080)'
    )
    
    parser.add_argument(
        '--telemetry-port', '-t',
        type=int,
        default=8765,
        help='Telemetry WebSocket port (default: 8765)'
    )
    
    parser.add_argument(
        '--update-interval', '-u',
        type=float,
        default=0.5,
        help='Telemetry update interval in seconds (default: 0.5)'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Enable demo mode with simulated consciousness activity'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create dashboard configuration
    config = DashboardConfig(
        web_port=args.port,
        telemetry_port=args.telemetry_port,
        update_interval=args.update_interval,
        auto_discover_systems=True,
        enable_consciousness_control=True,
        enable_topology_control=True
    )
    
    # Create and launch dashboard
    launcher = DashboardLauncher(config, demo_mode=args.demo)
    
    try:
        await launcher.launch()
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Dashboard failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        # Install required packages check
        try:
            import aiohttp
            import websockets
            import aiofiles
        except ImportError as e:
            print(f"❌ Missing required package: {e}")
            print("Install with: pip install aiohttp websockets aiofiles psutil")
            sys.exit(1)
        
        # Run the dashboard
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard startup cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Dashboard startup failed: {e}")
        sys.exit(1)
