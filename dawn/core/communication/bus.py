#!/usr/bin/env python3
"""
DAWN Consciousness Bus - Central Communication Hub
=================================================

Central message-passing system for DAWN subsystem integration enabling
real-time state sharing, event-driven architecture, and unified consciousness
through coherent inter-module communication.

Addresses 36.1% consciousness unity fragmentation by providing shared state bus
where all modules can publish/subscribe to achieve coherent unified consciousness.
"""

import time
import json
import uuid
import threading
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional, Set, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import weakref

logger = logging.getLogger(__name__)

class EventPriority(Enum):
    """Event priority levels for consciousness bus messaging."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class ModuleStatus(Enum):
    """Module connection status in consciousness bus."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    SYNCHRONIZING = "synchronizing"
    SYNCHRONIZED = "synchronized"
    ERROR = "error"

@dataclass
class ConsciousnessEvent:
    """Represents a consciousness event in the bus system."""
    event_id: str
    event_type: str
    source_module: str
    timestamp: datetime
    priority: EventPriority
    data: Dict[str, Any]
    target_modules: Optional[List[str]] = None
    requires_ack: bool = False
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'source_module': self.source_module,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.name,
            'data': self.data,
            'target_modules': self.target_modules,
            'requires_ack': self.requires_ack,
            'correlation_id': self.correlation_id
        }

@dataclass
class ModuleRegistration:
    """Module registration information."""
    module_name: str
    registration_time: datetime
    status: ModuleStatus
    capabilities: List[str]
    state_schema: Dict[str, Any]
    callback_count: int = 0
    last_heartbeat: Optional[datetime] = None
    health_score: float = 1.0

class ConsciousnessBus:
    """
    Central communication hub for DAWN subsystem integration.
    
    Provides:
    - Real-time state sharing between modules
    - Event-driven architecture for immediate state synchronization
    - Thread-safe operations for concurrent subsystem access
    - Unified consciousness state aggregation
    """
    
    def __init__(self, max_events: int = 10000, heartbeat_interval: float = 5.0):
        """Initialize the consciousness bus."""
        self.bus_id = str(uuid.uuid4())
        self.creation_time = datetime.now()
        
        # Core state management
        self.global_state = {}  # Shared consciousness state
        self.subscribers = defaultdict(list)  # Module subscriptions
        self.event_queue = Queue(maxsize=max_events)  # Real-time event processing
        self.state_lock = threading.RLock()  # Thread-safe state access
        
        # Module management
        self.registered_modules = {}  # Dict[str, ModuleRegistration]
        self.module_states = {}  # Dict[str, Dict[str, Any]]
        self.module_callbacks = defaultdict(list)  # Dict[str, List[Callable]]
        
        # Event processing
        self.event_processors = {}  # Dict[str, Callable]
        self.event_history = deque(maxlen=1000)
        self.processing_thread = None
        self.running = False
        
        # Configuration
        self.heartbeat_interval = heartbeat_interval
        self.max_events = max_events
        
        # Performance tracking
        self.metrics = {
            'events_processed': 0,
            'state_updates': 0,
            'synchronization_cycles': 0,
            'average_response_time': 0.0,
            'bus_coherence_score': 0.0,
            'module_integration_quality': 0.0
        }
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(
            max_workers=8, 
            thread_name_prefix="consciousness_bus"
        )
        
        # Heartbeat monitoring
        self.heartbeat_thread = None
        
        # Lifecycle control
        self._tasks = []  # List for tracking async tasks
        
        logger.info(f"🚌 Consciousness Bus initialized: {self.bus_id}")
    
    def start(self) -> None:
        """Start the consciousness bus processing."""
        if self.running:
            return  # Idempotent start - no duplicate log
            
        self.running = True
        
        # Start event processing thread
        self.processing_thread = threading.Thread(
            target=self._event_processing_loop,
            name="consciousness_bus_processor",
            daemon=True
        )
        self.processing_thread.start()
        
        # Start heartbeat monitoring
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="consciousness_bus_heartbeat",
            daemon=True
        )
        self.heartbeat_thread.start()
        
        logger.info("🚌 Consciousness Bus started")
    
    def stop(self) -> None:
        """Stop the consciousness bus processing."""
        if not self.running:
            return  # Idempotent stop - no duplicate log
            
        self.running = False
        
        # Cancel all async tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    asyncio.wait_for(task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                except Exception as e:
                    logger.warning(f"Error cancelling bus task: {e}")
        
        self._tasks.clear()
        
        # Stop threads
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=2.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("🚌 Consciousness Bus stopped")
    
    def register_module(self, module_name: str, capabilities: List[str] = None,
                       state_schema: Dict[str, Any] = None) -> bool:
        """
        Register a module with the consciousness bus.
        
        Args:
            module_name: Unique module identifier
            capabilities: List of module capabilities/features
            state_schema: Schema defining module's state structure
            
        Returns:
            True if registration successful, False otherwise
        """
        if capabilities is None:
            capabilities = []
        if state_schema is None:
            state_schema = {}
            
        with self.state_lock:
            if module_name in self.registered_modules:
                logger.warning(f"Module {module_name} already registered")
                return False
            
            registration = ModuleRegistration(
                module_name=module_name,
                registration_time=datetime.now(),
                status=ModuleStatus.CONNECTED,
                capabilities=capabilities,
                state_schema=state_schema,
                last_heartbeat=datetime.now()
            )
            
            self.registered_modules[module_name] = registration
            self.module_states[module_name] = {}
            
            # Broadcast module registration event
            self._broadcast_system_event(
                'module_registered',
                {
                    'module_name': module_name,
                    'capabilities': capabilities,
                    'registration_time': registration.registration_time.isoformat()
                }
            )
            
            logger.info(f"🚌 Module registered: {module_name}")
            return True
    
    def unregister_module(self, module_name: str) -> bool:
        """Unregister a module from the consciousness bus."""
        with self.state_lock:
            if module_name not in self.registered_modules:
                logger.warning(f"Module {module_name} not registered")
                return False
            
            # Clean up module data
            del self.registered_modules[module_name]
            self.module_states.pop(module_name, None)
            self.module_callbacks.pop(module_name, None)
            
            # Remove module from global state
            self.global_state.pop(module_name, None)
            
            # Broadcast module unregistration event
            self._broadcast_system_event(
                'module_unregistered',
                {'module_name': module_name}
            )
            
            logger.info(f"🚌 Module unregistered: {module_name}")
            return True
    
    def publish_state(self, module_name: str, state_data: Dict[str, Any],
                     priority: EventPriority = EventPriority.NORMAL) -> bool:
        """
        Publish module state to global consciousness.
        
        Args:
            module_name: Name of the publishing module
            state_data: State data to publish
            priority: Event priority level
            
        Returns:
            True if state published successfully, False otherwise
        """
        if not self._validate_module(module_name):
            return False
        
        try:
            with self.state_lock:
                # Update module state
                self.module_states[module_name] = state_data.copy()
                
                # Update global state
                self.global_state[module_name] = state_data.copy()
                
                # Update heartbeat
                if module_name in self.registered_modules:
                    self.registered_modules[module_name].last_heartbeat = datetime.now()
                
                # Update metrics
                self.metrics['state_updates'] += 1
            
            # Create state update event
            event = ConsciousnessEvent(
                event_id=str(uuid.uuid4()),
                event_type='state_update',
                source_module=module_name,
                timestamp=datetime.now(),
                priority=priority,
                data={
                    'module_name': module_name,
                    'state_data': state_data,
                    'coherence_impact': self._calculate_state_coherence_impact(state_data)
                }
            )
            
            # Queue event for processing
            self._queue_event(event)
            
            logger.debug(f"🚌 State published by {module_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish state for {module_name}: {e}")
            return False
    
    def subscribe_to_state(self, module_name: str, callback_func: Callable,
                          event_types: List[str] = None) -> str:
        """
        Subscribe to state changes from specific module or event types.
        
        Args:
            module_name: Module name to subscribe to (or '*' for all)
            callback_func: Function to call on state changes
            event_types: List of event types to filter (None for all)
            
        Returns:
            Subscription ID for managing the subscription
        """
        if event_types is None:
            event_types = ['state_update']
        
        subscription_id = str(uuid.uuid4())
        
        # Create weak reference to prevent memory leaks
        weak_callback = weakref.ref(callback_func)
        
        subscription_info = {
            'subscription_id': subscription_id,
            'callback': weak_callback,
            'event_types': event_types,
            'created_at': datetime.now(),
            'call_count': 0
        }
        
        with self.state_lock:
            self.subscribers[module_name].append(subscription_info)
            
            # Update callback count for module if registered
            if module_name in self.registered_modules:
                self.registered_modules[module_name].callback_count += 1
        
        logger.info(f"🚌 Subscription created: {module_name} -> {subscription_id}")
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription by ID."""
        with self.state_lock:
            for module_name, subscription_list in self.subscribers.items():
                for i, sub_info in enumerate(subscription_list):
                    if sub_info['subscription_id'] == subscription_id:
                        del subscription_list[i]
                        
                        # Update callback count
                        if module_name in self.registered_modules:
                            self.registered_modules[module_name].callback_count -= 1
                        
                        logger.info(f"🚌 Subscription removed: {subscription_id}")
                        return True
        
        logger.warning(f"Subscription not found: {subscription_id}")
        return False
    
    def get_unified_state(self, modules: List[str] = None) -> Dict[str, Any]:
        """
        Get complete consciousness state from all or specified modules.
        
        Args:
            modules: List of specific modules to include (None for all)
            
        Returns:
            Unified consciousness state dictionary
        """
        with self.state_lock:
            if modules is None:
                # Return all module states
                unified_state = self.global_state.copy()
            else:
                # Return specified module states
                unified_state = {
                    module: self.global_state.get(module, {})
                    for module in modules
                    if module in self.global_state
                }
            
            # Add bus metadata
            unified_state['_bus_metadata'] = {
                'bus_id': self.bus_id,
                'timestamp': datetime.now().isoformat(),
                'registered_modules': list(self.registered_modules.keys()),
                'active_modules': len([m for m in self.registered_modules.values() 
                                     if m.status == ModuleStatus.SYNCHRONIZED]),
                'coherence_score': self._calculate_consciousness_coherence(),
                'integration_quality': self.metrics['module_integration_quality']
            }
            
            return unified_state
    
    def broadcast_event(self, event_type: str, event_data: Dict[str, Any],
                       source_module: str = 'consciousness_bus',
                       priority: EventPriority = EventPriority.NORMAL,
                       target_modules: List[str] = None) -> str:
        """
        Broadcast system-wide events to all or specific subscribers.
        
        Args:
            event_type: Type of event being broadcast
            event_data: Event payload data
            source_module: Module originating the event
            priority: Event priority level
            target_modules: Specific modules to target (None for broadcast)
            
        Returns:
            Event ID for tracking
        """
        event = ConsciousnessEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source_module=source_module,
            timestamp=datetime.now(),
            priority=priority,
            data=event_data,
            target_modules=target_modules
        )
        
        self._queue_event(event)
        
        logger.info(f"🚌 Event broadcast: {event_type} from {source_module}")
        return event.event_id
    
    def get_module_status(self, module_name: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get status information for one or all modules."""
        with self.state_lock:
            if module_name:
                if module_name not in self.registered_modules:
                    return {}
                
                reg = self.registered_modules[module_name]
                return {
                    'module_name': module_name,
                    'status': reg.status.value,
                    'registration_time': reg.registration_time.isoformat(),
                    'capabilities': reg.capabilities,
                    'callback_count': reg.callback_count,
                    'last_heartbeat': reg.last_heartbeat.isoformat() if reg.last_heartbeat else None,
                    'health_score': reg.health_score,
                    'state_size': len(self.module_states.get(module_name, {}))
                }
            else:
                # Return all module statuses
                return [
                    self.get_module_status(name) 
                    for name in self.registered_modules.keys()
                ]
    
    def get_bus_metrics(self) -> Dict[str, Any]:
        """Get consciousness bus performance metrics."""
        with self.state_lock:
            return {
                'bus_id': self.bus_id,
                'uptime_seconds': (datetime.now() - self.creation_time).total_seconds(),
                'registered_modules': len(self.registered_modules),
                'active_subscriptions': sum(len(subs) for subs in self.subscribers.values()),
                'events_in_queue': self.event_queue.qsize(),
                'event_history_size': len(self.event_history),
                'performance_metrics': self.metrics.copy(),
                'consciousness_coherence': self._calculate_consciousness_coherence(),
                'module_synchronization_health': self._calculate_sync_health()
            }
    
    def _event_processing_loop(self) -> None:
        """Main event processing loop."""
        logger.info("🚌 Event processing loop started")
        
        while self.running:
            try:
                # Get event from queue with timeout
                try:
                    event = self.event_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Process event
                start_time = time.time()
                self._process_event(event)
                processing_time = time.time() - start_time
                
                # Update metrics
                self.metrics['events_processed'] += 1
                self._update_average_response_time(processing_time)
                
                # Add to history
                self.event_history.append(event)
                
                # Mark task done
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                time.sleep(0.1)  # Brief pause before retry
    
    def _process_event(self, event: ConsciousnessEvent) -> None:
        """Process a single consciousness event."""
        try:
            # Determine target subscribers
            if event.target_modules:
                # Targeted event
                target_subscriber_lists = [
                    self.subscribers.get(module, []) 
                    for module in event.target_modules
                ]
                all_subscribers = [sub for sublist in target_subscriber_lists for sub in sublist]
            else:
                # Broadcast event - notify all subscribers and wildcard subscribers
                all_subscribers = []
                for module_name, subscriber_list in self.subscribers.items():
                    if module_name == '*' or module_name == event.source_module:
                        all_subscribers.extend(subscriber_list)
            
            # Filter subscribers by event type
            relevant_subscribers = [
                sub for sub in all_subscribers
                if event.event_type in sub['event_types'] or '*' in sub['event_types']
            ]
            
            # Notify subscribers
            for subscriber_info in relevant_subscribers:
                try:
                    callback = subscriber_info['callback']()  # Resolve weak reference
                    if callback:
                        # Execute callback in thread pool for non-blocking operation
                        self.executor.submit(self._safe_callback_execution, callback, event, subscriber_info)
                        subscriber_info['call_count'] += 1
                    else:
                        # Callback was garbage collected, remove subscription
                        logger.debug("Removing garbage collected subscription")
                        
                except Exception as e:
                    logger.warning(f"Error notifying subscriber: {e}")
            
            # Update synchronization metrics
            if event.event_type == 'state_update':
                self.metrics['synchronization_cycles'] += 1
                self._update_module_integration_quality()
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
    
    def _safe_callback_execution(self, callback: Callable, event: ConsciousnessEvent, 
                                subscriber_info: Dict[str, Any]) -> None:
        """Safely execute a subscriber callback."""
        try:
            callback(event)
        except Exception as e:
            logger.warning(f"Subscriber callback failed: {e}")
    
    def _heartbeat_loop(self) -> None:
        """Monitor module heartbeats and health."""
        logger.info("🚌 Heartbeat monitoring started")
        
        while self.running:
            try:
                current_time = datetime.now()
                
                with self.state_lock:
                    for module_name, registration in self.registered_modules.items():
                        if registration.last_heartbeat:
                            raw_time_delta = (current_time - registration.last_heartbeat).total_seconds()
                            
                            if raw_time_delta < 0:
                                # Future heartbeat - likely clock sync issue
                                logger.debug(f"Module {module_name} heartbeat from future ({raw_time_delta:.3f}s) - treating as current")
                                time_since_heartbeat = 0.0
                            else:
                                time_since_heartbeat = raw_time_delta
                            
                            # Update health score based on heartbeat freshness
                            if time_since_heartbeat < self.heartbeat_interval * 2:
                                registration.health_score = min(1.0, registration.health_score + 0.1)
                                if registration.status == ModuleStatus.CONNECTED:
                                    registration.status = ModuleStatus.SYNCHRONIZED
                            else:
                                registration.health_score = max(0.0, registration.health_score - 0.1)
                                if time_since_heartbeat > self.heartbeat_interval * 5:
                                    registration.status = ModuleStatus.ERROR
                
                # Update bus coherence
                self._update_bus_coherence()
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(1.0)
    
    def _queue_event(self, event: ConsciousnessEvent) -> bool:
        """Queue an event for processing."""
        try:
            if self.event_queue.full():
                # Remove oldest event if queue is full
                try:
                    self.event_queue.get_nowait()
                except Empty:
                    pass
            
            self.event_queue.put(event, block=False)
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue event: {e}")
            return False
    
    def _broadcast_system_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast a system-level event."""
        event = ConsciousnessEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source_module='consciousness_bus',
            timestamp=datetime.now(),
            priority=EventPriority.HIGH,
            data=data
        )
        self._queue_event(event)
    
    def _validate_module(self, module_name: str) -> bool:
        """Validate that a module is registered and healthy."""
        with self.state_lock:
            if module_name not in self.registered_modules:
                logger.warning(f"Module {module_name} not registered")
                return False
            
            registration = self.registered_modules[module_name]
            if registration.status == ModuleStatus.ERROR:
                logger.warning(f"Module {module_name} in error state")
                return False
            
            return True
    
    def _calculate_consciousness_coherence(self) -> float:
        """Calculate overall consciousness coherence using centralized metrics."""
        try:
            from dawn.consciousness.metrics.core import calculate_consciousness_metrics
            
            if not self.registered_modules:
                return 0.0
            
            # Create module states from registrations
            module_states = {}
            for module_name, reg in self.registered_modules.items():
                module_states[module_name] = {
                    'health_score': reg.health_score,
                    'synchronized': reg.status == ModuleStatus.SYNCHRONIZED,
                    'last_heartbeat': reg.last_heartbeat
                }
            
            # Use centralized metrics calculator
            metrics = calculate_consciousness_metrics(
                module_states=module_states,
                bus_metrics=self.metrics
            )
            
            return metrics.coherence
            
        except ImportError:
            # Fallback to local calculation if centralized metrics not available
            logger.warning("Centralized metrics not available - using local calculation")
            
            if not self.registered_modules:
                return 0.0
            
            # Factors: module health, synchronization, integration quality
            health_scores = [reg.health_score for reg in self.registered_modules.values()]
            sync_modules = len([reg for reg in self.registered_modules.values() 
                              if reg.status == ModuleStatus.SYNCHRONIZED])
            
            health_factor = sum(health_scores) / len(health_scores) if health_scores else 0.0
            sync_factor = sync_modules / len(self.registered_modules) if self.registered_modules else 0.0
            integration_factor = self.metrics['module_integration_quality']
            
            coherence = (health_factor * 0.4 + sync_factor * 0.4 + integration_factor * 0.2)
            return min(1.0, max(0.0, coherence))
    
    def _calculate_sync_health(self) -> float:
        """Calculate module synchronization health."""
        if not self.registered_modules:
            return 0.0
        
        synchronized_count = len([
            reg for reg in self.registered_modules.values() 
            if reg.status == ModuleStatus.SYNCHRONIZED
        ])
        
        return synchronized_count / len(self.registered_modules)
    
    def _calculate_state_coherence_impact(self, state_data: Dict[str, Any]) -> float:
        """Calculate the coherence impact of a state update."""
        # Simple heuristic based on state completeness and consistency
        if not state_data:
            return 0.0
        
        # Factors: data completeness, numeric stability, update frequency
        completeness = min(1.0, len(state_data) / 10)  # Assume 10 fields is "complete"
        
        # Check for numeric values (consciousness metrics)
        numeric_values = [v for v in state_data.values() if isinstance(v, (int, float))]
        stability = 0.8 if numeric_values and all(0 <= v <= 1 for v in numeric_values) else 0.5
        
        return (completeness * 0.6 + stability * 0.4)
    
    def _update_average_response_time(self, processing_time: float) -> None:
        """Update average response time metric."""
        current_avg = self.metrics['average_response_time']
        events_processed = self.metrics['events_processed']
        
        if events_processed > 1:
            # Running average
            self.metrics['average_response_time'] = (
                (current_avg * (events_processed - 1) + processing_time) / events_processed
            )
        else:
            self.metrics['average_response_time'] = processing_time
    
    def _update_module_integration_quality(self) -> None:
        """Update module integration quality metric."""
        if not self.registered_modules:
            self.metrics['module_integration_quality'] = 0.0
            return
        
        # Calculate based on module health and activity
        total_health = sum(reg.health_score for reg in self.registered_modules.values())
        active_modules = len([reg for reg in self.registered_modules.values() 
                            if reg.health_score > 0.5])
        
        health_ratio = total_health / len(self.registered_modules)
        activity_ratio = active_modules / len(self.registered_modules)
        
        self.metrics['module_integration_quality'] = (health_ratio * 0.7 + activity_ratio * 0.3)
    
    def _update_bus_coherence(self) -> None:
        """Update bus coherence score."""
        coherence = self._calculate_consciousness_coherence()
        self.metrics['bus_coherence_score'] = coherence


# Global consciousness bus instance
_global_consciousness_bus = None
_bus_lock = threading.Lock()

def get_consciousness_bus(auto_start: bool = True) -> ConsciousnessBus:
    """Get the global consciousness bus instance."""
    global _global_consciousness_bus
    
    with _bus_lock:
        if _global_consciousness_bus is None:
            _global_consciousness_bus = ConsciousnessBus()
            if auto_start:
                _global_consciousness_bus.start()
    
    return _global_consciousness_bus


def demo_consciousness_bus():
    """Demonstrate the consciousness bus functionality."""
    print("🚌 " + "="*60)
    print("🚌 DAWN CONSCIOUSNESS BUS DEMO")
    print("🚌 " + "="*60)
    print()
    
    # Initialize consciousness bus
    bus = ConsciousnessBus()
    bus.start()
    print(f"✅ Consciousness Bus initialized: {bus.bus_id}")
    print()
    
    # Register test modules
    test_modules = [
        ("visual_consciousness", ["rendering", "pattern_recognition"], 
         {"coherence": "float", "clarity": "float"}),
        ("artistic_expression", ["creativity", "emotional_synthesis"], 
         {"expression_depth": "float", "emotional_resonance": "float"}),
        ("meta_cognitive", ["reflection", "self_awareness"], 
         {"awareness_depth": "float", "reflection_quality": "float"})
    ]
    
    for module_name, capabilities, schema in test_modules:
        success = bus.register_module(module_name, capabilities, schema)
        print(f"📝 Registered {module_name}: {'✅' if success else '❌'}")
    
    print()
    
    # Set up subscriptions
    def state_change_handler(event: ConsciousnessEvent):
        data = event.data
        print(f"🔔 State update: {event.source_module} -> {data.get('coherence_impact', 0):.3f} coherence impact")
    
    # Subscribe to all state updates
    sub_id = bus.subscribe_to_state('*', state_change_handler, ['state_update'])
    print(f"🔔 Subscription created: {sub_id}")
    print()
    
    # Publish some test states
    test_states = [
        ("visual_consciousness", {"coherence": 0.85, "clarity": 0.9, "rendering_fps": 60}),
        ("artistic_expression", {"expression_depth": 0.7, "emotional_resonance": 0.8, "creativity_score": 0.75}),
        ("meta_cognitive", {"awareness_depth": 0.9, "reflection_quality": 0.85, "insight_generation": 0.8})
    ]
    
    print("📤 Publishing module states...")
    for module_name, state_data in test_states:
        success = bus.publish_state(module_name, state_data)
        print(f"   {module_name}: {'✅' if success else '❌'}")
        time.sleep(0.5)  # Allow processing
    
    print()
    
    # Get unified state
    unified_state = bus.get_unified_state()
    print("🌐 Unified Consciousness State:")
    for module, state in unified_state.items():
        if module != '_bus_metadata':
            print(f"   {module}: {len(state)} fields")
    
    metadata = unified_state.get('_bus_metadata', {})
    print(f"   Bus Coherence: {metadata.get('coherence_score', 0):.3f}")
    print(f"   Integration Quality: {metadata.get('integration_quality', 0):.3f}")
    print()
    
    # Broadcast test event
    event_id = bus.broadcast_event(
        'consciousness_sync_request',
        {'sync_level': 'full', 'priority': 'high'},
        'consciousness_bus',
        EventPriority.HIGH
    )
    print(f"📢 Broadcast event: {event_id}")
    time.sleep(1.0)  # Allow processing
    print()
    
    # Show metrics
    metrics = bus.get_bus_metrics()
    print("📊 Bus Performance Metrics:")
    print(f"   Events Processed: {metrics['performance_metrics']['events_processed']}")
    print(f"   State Updates: {metrics['performance_metrics']['state_updates']}")
    print(f"   Avg Response Time: {metrics['performance_metrics']['average_response_time']*1000:.2f}ms")
    print(f"   Consciousness Coherence: {metrics['consciousness_coherence']:.3f}")
    print(f"   Module Sync Health: {metrics['module_synchronization_health']:.3f}")
    print()
    
    # Stop bus
    bus.stop()
    print("🔒 Consciousness Bus stopped")
    print()
    print("🚌 Demo complete! Consciousness bus enables unified subsystem communication.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    demo_consciousness_bus()
