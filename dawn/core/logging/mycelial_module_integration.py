#!/usr/bin/env python3
"""
🍄🔗 DAWN Mycelial Module Integration System
============================================

Wires the mycelial semantic hash map into all DAWN modules, creating a living
network that travels and propagates semantic meaning as code executes throughout
the entire codebase. Every module operation becomes a spore that spreads through
the semantic network.

This system:
1. Auto-integrates mycelial hash map into all discovered modules
2. Maps DAWN folder structure to mycelial network topology
3. Injects semantic context into module operations
4. Enables cross-module semantic spore propagation
5. Creates living semantic telemetry that follows code execution paths

Architecture:
- MycelialModuleIntegrator: Main integration orchestrator
- ModuleSemanticWrapper: Wraps modules with mycelial behavior
- SemanticPathMapper: Maps code paths to semantic concepts
- CrossModulePropagator: Manages inter-module spore propagation
- FolderTopologyMapper: Maps folder structure to network topology
"""

import sys
import os
import time
import threading
import importlib
import inspect
import types
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

# Import mycelial hash map system
try:
    from .mycelial_semantic_hashmap import (
        get_mycelial_hashmap, store_semantic_data, touch_semantic_concept,
        ping_semantic_network, SemanticSpore, SporeType, PropagationMode
    )
    MYCELIAL_AVAILABLE = True
except ImportError:
    MYCELIAL_AVAILABLE = False

# Import DAWN core systems
try:
    from dawn.core.singleton import get_dawn
    DAWN_SINGLETON_AVAILABLE = True
except ImportError:
    DAWN_SINGLETON_AVAILABLE = False

try:
    from dawn.core.telemetry import log_event, TelemetryLevel
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ModuleSemanticContext:
    """Semantic context for a DAWN module"""
    module_path: str
    module_name: str
    namespace: str
    folder_depth: int
    semantic_concepts: Set[str] = field(default_factory=set)
    parent_concepts: Set[str] = field(default_factory=set)
    child_concepts: Set[str] = field(default_factory=set)
    execution_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    spore_generation_count: int = 0

@dataclass
class SemanticExecutionTrace:
    """Trace of semantic execution through modules"""
    trace_id: str
    start_module: str
    current_module: str
    execution_path: List[str] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    spores_generated: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    active: bool = True

class SemanticPathMapper:
    """Maps DAWN code paths to semantic concepts"""
    
    def __init__(self):
        self.path_concept_map: Dict[str, Set[str]] = {}
        self.concept_hierarchy: Dict[str, Dict[str, Set[str]]] = {}
        
        # Initialize with DAWN folder structure concepts
        self._initialize_dawn_concepts()
    
    def _initialize_dawn_concepts(self):
        """Initialize semantic concepts based on DAWN folder structure"""
        
        # Core concepts by namespace
        namespace_concepts = {
            'core': {'foundation', 'singleton', 'communication', 'telemetry', 'logging'},
            'consciousness': {'awareness', 'engines', 'models', 'states', 'perception'},
            'processing': {'engines', 'pipelines', 'tick', 'orchestration', 'computation'},
            'memory': {'storage', 'retrieval', 'patterns', 'associations', 'persistence'},
            'subsystems': {'specialized', 'domain', 'integration', 'coordination'},
            'interfaces': {'interaction', 'communication', 'user', 'external', 'api'},
            'capabilities': {'dynamic', 'skills', 'abilities', 'functions', 'features'},
            'extensions': {'plugins', 'addons', 'expansion', 'modularity'},
            'tools': {'development', 'analysis', 'utilities', 'debugging'},
            'research': {'experimentation', 'exploration', 'innovation', 'discovery'}
        }
        
        # Build concept hierarchy
        for namespace, concepts in namespace_concepts.items():
            self.concept_hierarchy[namespace] = {
                'core_concepts': concepts,
                'derived_concepts': set(),
                'related_concepts': set()
            }
    
    def extract_semantic_concepts(self, module_path: str) -> Set[str]:
        """Extract semantic concepts from module path"""
        if module_path in self.path_concept_map:
            return self.path_concept_map[module_path]
        
        concepts = set()
        
        # Split path into components
        path_parts = module_path.split('.')
        
        # Add namespace concepts
        if len(path_parts) > 0:
            namespace = path_parts[0]
            if namespace in self.concept_hierarchy:
                concepts.update(self.concept_hierarchy[namespace]['core_concepts'])
        
        # Add path component concepts
        for part in path_parts:
            # Convert snake_case to concepts
            part_concepts = part.replace('_', ' ').split()
            concepts.update(part_concepts)
        
        # Add derived concepts based on common patterns
        if 'engine' in module_path.lower():
            concepts.update({'processing', 'execution', 'computation'})
        if 'manager' in module_path.lower():
            concepts.update({'coordination', 'orchestration', 'control'})
        if 'system' in module_path.lower():
            concepts.update({'integration', 'architecture', 'framework'})
        
        # Filter out common words
        filtered_concepts = {c for c in concepts if len(c) > 2 and c not in {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }}
        
        self.path_concept_map[module_path] = filtered_concepts
        return filtered_concepts
    
    def get_concept_relationships(self, concept: str) -> Dict[str, Set[str]]:
        """Get relationships for a concept"""
        relationships = {
            'parents': set(),
            'children': set(),
            'siblings': set()
        }
        
        # Find relationships in concept hierarchy
        for namespace, hierarchy in self.concept_hierarchy.items():
            if concept in hierarchy['core_concepts']:
                # Add namespace as parent
                relationships['parents'].add(namespace)
                # Add other concepts in namespace as siblings
                relationships['siblings'].update(hierarchy['core_concepts'] - {concept})
        
        return relationships

class ModuleSemanticWrapper:
    """Wraps DAWN modules with mycelial semantic behavior"""
    
    def __init__(self, module, module_context: ModuleSemanticContext, 
                 semantic_mapper: SemanticPathMapper):
        self.original_module = module
        self.context = module_context
        self.semantic_mapper = semantic_mapper
        self.wrapped_functions: Dict[str, Callable] = {}
        self.active_traces: Dict[str, SemanticExecutionTrace] = {}
        
        # Get mycelial hash map
        if MYCELIAL_AVAILABLE:
            self.hashmap = get_mycelial_hashmap()
        else:
            self.hashmap = None
    
    def wrap_module(self):
        """Wrap all functions in the module with semantic behavior"""
        if not hasattr(self.original_module, '__dict__'):
            return self.original_module
        
        # Create wrapped module
        wrapped_module = types.ModuleType(self.original_module.__name__)
        
        # Copy all attributes
        for name, attr in self.original_module.__dict__.items():
            if callable(attr) and not name.startswith('_'):
                # Wrap callable with semantic behavior
                wrapped_attr = self._wrap_callable(attr, name)
                setattr(wrapped_module, name, wrapped_attr)
                self.wrapped_functions[name] = wrapped_attr
            else:
                # Copy non-callable attributes
                setattr(wrapped_module, name, attr)
        
        # Store module context in mycelial hash map
        if self.hashmap:
            self._store_module_semantic_context()
        
        return wrapped_module
    
    def _wrap_callable(self, func: Callable, func_name: str) -> Callable:
        """Wrap a callable with semantic spore propagation"""
        
        def semantic_wrapper(*args, **kwargs):
            # Start semantic execution trace
            trace_id = f"{self.context.module_path}.{func_name}_{int(time.time() * 1000000)}"
            
            # Create execution trace
            trace = SemanticExecutionTrace(
                trace_id=trace_id,
                start_module=self.context.module_path,
                current_module=self.context.module_path,
                semantic_context={
                    'function_name': func_name,
                    'module_concepts': list(self.context.semantic_concepts),
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
            )
            
            self.active_traces[trace_id] = trace
            
            try:
                # Generate semantic spores before execution
                self._generate_execution_spores(func_name, trace)
                
                # Execute original function
                result = func(*args, **kwargs)
                
                # Generate completion spores
                self._generate_completion_spores(func_name, trace, result)
                
                # Update context
                self.context.execution_count += 1
                self.context.last_accessed = time.time()
                
                return result
                
            except Exception as e:
                # Generate error spores
                self._generate_error_spores(func_name, trace, e)
                raise
            
            finally:
                # Clean up trace
                trace.active = False
                if trace_id in self.active_traces:
                    del self.active_traces[trace_id]
        
        # Preserve function metadata
        semantic_wrapper.__name__ = func.__name__
        semantic_wrapper.__doc__ = func.__doc__
        semantic_wrapper.__module__ = func.__module__
        
        return semantic_wrapper
    
    def _store_module_semantic_context(self):
        """Store module semantic context in mycelial hash map"""
        if not self.hashmap:
            return
        
        # Store module context
        context_key = f"module_context_{self.context.module_path}"
        context_data = {
            'module_path': self.context.module_path,
            'namespace': self.context.namespace,
            'semantic_concepts': list(self.context.semantic_concepts),
            'folder_depth': self.context.folder_depth,
            'wrapped_functions': list(self.wrapped_functions.keys())
        }
        
        store_semantic_data(context_key, context_data)
        
        # Store individual concepts
        for concept in self.context.semantic_concepts:
            concept_key = f"concept_{concept}_{self.context.module_path}"
            store_semantic_data(concept_key, {
                'concept': concept,
                'module': self.context.module_path,
                'context': 'module_semantic_context'
            })
    
    def _generate_execution_spores(self, func_name: str, trace: SemanticExecutionTrace):
        """Generate spores when function execution starts"""
        if not self.hashmap:
            return
        
        # Touch module concepts to trigger propagation
        for concept in self.context.semantic_concepts:
            ping_semantic_network(f"concept_{concept}_{self.context.module_path}")
        
        # Generate execution spore
        execution_key = f"execution_{self.context.module_path}_{func_name}"
        execution_data = {
            'module': self.context.module_path,
            'function': func_name,
            'trace_id': trace.trace_id,
            'execution_type': 'function_start',
            'timestamp': time.time()
        }
        
        node_id = store_semantic_data(execution_key, execution_data)
        trace.spores_generated.append(node_id)
        self.context.spore_generation_count += 1
    
    def _generate_completion_spores(self, func_name: str, trace: SemanticExecutionTrace, result: Any):
        """Generate spores when function completes successfully"""
        if not self.hashmap:
            return
        
        completion_key = f"completion_{self.context.module_path}_{func_name}"
        completion_data = {
            'module': self.context.module_path,
            'function': func_name,
            'trace_id': trace.trace_id,
            'execution_type': 'function_complete',
            'result_type': type(result).__name__,
            'timestamp': time.time()
        }
        
        node_id = store_semantic_data(completion_key, completion_data)
        trace.spores_generated.append(node_id)
    
    def _generate_error_spores(self, func_name: str, trace: SemanticExecutionTrace, error: Exception):
        """Generate spores when function encounters an error"""
        if not self.hashmap:
            return
        
        error_key = f"error_{self.context.module_path}_{func_name}"
        error_data = {
            'module': self.context.module_path,
            'function': func_name,
            'trace_id': trace.trace_id,
            'execution_type': 'function_error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': time.time()
        }
        
        node_id = store_semantic_data(error_key, error_data)
        trace.spores_generated.append(node_id)
        
        # Generate error propagation
        touch_semantic_concept('error', energy=0.8)

class FolderTopologyMapper:
    """Maps DAWN folder structure to mycelial network topology"""
    
    def __init__(self, dawn_root_path: str):
        self.dawn_root = Path(dawn_root_path)
        self.folder_topology: Dict[str, Dict[str, Any]] = {}
        self.depth_mapping: Dict[int, List[str]] = defaultdict(list)
        
        self._build_topology_map()
    
    def _build_topology_map(self):
        """Build topology map from DAWN folder structure"""
        if not self.dawn_root.exists():
            logger.warning(f"DAWN root path does not exist: {self.dawn_root}")
            return
        
        # Walk through DAWN directory structure
        for root, dirs, files in os.walk(self.dawn_root):
            root_path = Path(root)
            relative_path = root_path.relative_to(self.dawn_root)
            
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            # Calculate depth
            depth = len(relative_path.parts) if relative_path != Path('.') else 0
            
            # Create folder entry
            folder_key = str(relative_path).replace('/', '.') if relative_path != Path('.') else 'dawn'
            
            self.folder_topology[folder_key] = {
                'path': str(relative_path),
                'depth': depth,
                'parent': str(relative_path.parent).replace('/', '.') if relative_path.parent != Path('.') else None,
                'children': [],
                'python_files': [f for f in files if f.endswith('.py') and f != '__init__.py'],
                'semantic_concepts': self._extract_folder_concepts(folder_key)
            }
            
            self.depth_mapping[depth].append(folder_key)
        
        # Build parent-child relationships
        for folder_key, folder_data in self.folder_topology.items():
            parent_key = folder_data['parent']
            if parent_key and parent_key in self.folder_topology:
                self.folder_topology[parent_key]['children'].append(folder_key)
    
    def _extract_folder_concepts(self, folder_key: str) -> Set[str]:
        """Extract semantic concepts from folder name and path"""
        concepts = set()
        
        # Split folder key into parts
        parts = folder_key.split('.')
        
        # Add each part as a concept
        for part in parts:
            if part and part != 'dawn':
                # Convert snake_case to separate concepts
                part_concepts = part.replace('_', ' ').split()
                concepts.update(part_concepts)
        
        return concepts
    
    def get_folder_hierarchy(self) -> Dict[int, List[str]]:
        """Get folder hierarchy by depth"""
        return dict(self.depth_mapping)
    
    def get_folder_relationships(self, folder_key: str) -> Dict[str, Any]:
        """Get relationships for a folder"""
        if folder_key not in self.folder_topology:
            return {}
        
        folder_data = self.folder_topology[folder_key]
        return {
            'parent': folder_data['parent'],
            'children': folder_data['children'],
            'siblings': self._get_siblings(folder_key),
            'depth': folder_data['depth'],
            'concepts': folder_data['semantic_concepts']
        }
    
    def _get_siblings(self, folder_key: str) -> List[str]:
        """Get sibling folders at the same level"""
        folder_data = self.folder_topology[folder_key]
        parent_key = folder_data['parent']
        
        if not parent_key:
            # Root level siblings
            return [k for k, v in self.folder_topology.items() 
                   if v['depth'] == 1 and k != folder_key]
        
        # Same parent siblings
        parent_data = self.folder_topology.get(parent_key, {})
        return [child for child in parent_data.get('children', []) if child != folder_key]

class CrossModulePropagator:
    """Manages cross-module semantic spore propagation"""
    
    def __init__(self, module_wrappers: Dict[str, ModuleSemanticWrapper]):
        self.module_wrappers = module_wrappers
        self.cross_module_traces: Dict[str, List[str]] = defaultdict(list)
        self.propagation_stats = {
            'cross_module_propagations': 0,
            'concept_bridges': 0,
            'namespace_crossings': 0
        }
        
        if MYCELIAL_AVAILABLE:
            self.hashmap = get_mycelial_hashmap()
        else:
            self.hashmap = None
    
    def propagate_across_modules(self, source_module: str, target_concepts: Set[str]):
        """Propagate semantic spores across related modules"""
        if not self.hashmap:
            return
        
        # Find modules with related concepts
        related_modules = self._find_related_modules(target_concepts)
        
        for module_path in related_modules:
            if module_path != source_module and module_path in self.module_wrappers:
                # Create cross-module spore
                cross_module_key = f"cross_module_{source_module}_to_{module_path}"
                cross_module_data = {
                    'source_module': source_module,
                    'target_module': module_path,
                    'propagation_concepts': list(target_concepts),
                    'propagation_type': 'cross_module',
                    'timestamp': time.time()
                }
                
                store_semantic_data(cross_module_key, cross_module_data)
                
                # Touch concepts in target module
                wrapper = self.module_wrappers[module_path]
                for concept in target_concepts & wrapper.context.semantic_concepts:
                    ping_semantic_network(f"concept_{concept}_{module_path}")
                
                self.propagation_stats['cross_module_propagations'] += 1
                
                # Track cross-module trace
                trace_key = f"{source_module}→{module_path}"
                self.cross_module_traces[trace_key].append(time.time())
    
    def _find_related_modules(self, concepts: Set[str]) -> List[str]:
        """Find modules that contain related concepts"""
        related_modules = []
        
        for module_path, wrapper in self.module_wrappers.items():
            # Check for concept overlap
            overlap = concepts & wrapper.context.semantic_concepts
            if overlap:
                related_modules.append(module_path)
        
        return related_modules
    
    def bridge_namespaces(self, source_namespace: str, target_namespace: str, bridge_concepts: Set[str]):
        """Create semantic bridges between different namespaces"""
        if not self.hashmap:
            return
        
        bridge_key = f"namespace_bridge_{source_namespace}_to_{target_namespace}"
        bridge_data = {
            'source_namespace': source_namespace,
            'target_namespace': target_namespace,
            'bridge_concepts': list(bridge_concepts),
            'bridge_type': 'namespace_crossing',
            'timestamp': time.time()
        }
        
        store_semantic_data(bridge_key, bridge_data)
        
        # Propagate concepts across namespace boundary
        for concept in bridge_concepts:
            touch_semantic_concept(concept, energy=0.9)
        
        self.propagation_stats['namespace_crossings'] += 1

class MycelialModuleIntegrator:
    """Main orchestrator for mycelial module integration"""
    
    def __init__(self, dawn_root_path: Optional[str] = None):
        self.dawn_root = dawn_root_path or str(Path(__file__).parent.parent.parent)
        self.semantic_mapper = SemanticPathMapper()
        self.topology_mapper = FolderTopologyMapper(self.dawn_root)
        self.module_contexts: Dict[str, ModuleSemanticContext] = {}
        self.module_wrappers: Dict[str, ModuleSemanticWrapper] = {}
        self.cross_module_propagator: Optional[CrossModulePropagator] = None
        
        self.integration_stats = {
            'modules_wrapped': 0,
            'concepts_mapped': 0,
            'spores_generated': 0,
            'cross_module_propagations': 0
        }
        
        self.running = False
        self.integration_thread: Optional[threading.Thread] = None
        
        logger.info("🍄🔗 Mycelial Module Integrator initialized")
    
    def start_integration(self):
        """Start the mycelial module integration system"""
        if self.running or not MYCELIAL_AVAILABLE:
            return
        
        self.running = True
        
        # Initialize cross-module propagator
        self.cross_module_propagator = CrossModulePropagator(self.module_wrappers)
        
        # Start integration thread
        self.integration_thread = threading.Thread(
            target=self._integration_loop,
            name="mycelial_module_integrator",
            daemon=True
        )
        self.integration_thread.start()
        
        logger.info("🍄🔗 Mycelial module integration started")
    
    def stop_integration(self):
        """Stop the mycelial module integration system"""
        if not self.running:
            return
        
        self.running = False
        
        if self.integration_thread and self.integration_thread.is_alive():
            self.integration_thread.join(timeout=2.0)
        
        logger.info("🍄🔗 Mycelial module integration stopped")
    
    def integrate_module(self, module_path: str, module_obj: Any = None) -> Any:
        """Integrate a single module with mycelial behavior"""
        if not MYCELIAL_AVAILABLE:
            return module_obj
        
        # Load module if not provided
        if module_obj is None:
            try:
                module_obj = importlib.import_module(module_path)
            except ImportError as e:
                logger.warning(f"Could not load module {module_path}: {e}")
                return None
        
        # Create module context
        context = self._create_module_context(module_path)
        self.module_contexts[module_path] = context
        
        # Create wrapper
        wrapper = ModuleSemanticWrapper(module_obj, context, self.semantic_mapper)
        wrapped_module = wrapper.wrap_module()
        
        self.module_wrappers[module_path] = wrapper
        self.integration_stats['modules_wrapped'] += 1
        self.integration_stats['concepts_mapped'] += len(context.semantic_concepts)
        
        logger.debug(f"🍄🔗 Integrated module: {module_path}")
        
        return wrapped_module
    
    def integrate_discovered_modules(self, discovered_modules: Dict[str, List[str]]):
        """Integrate all discovered DAWN modules"""
        total_modules = sum(len(modules) for modules in discovered_modules.values())
        logger.info(f"🍄🔗 Integrating {total_modules} discovered modules...")
        
        integrated_count = 0
        
        for namespace, modules in discovered_modules.items():
            for module_path in modules:
                try:
                    full_module_path = f"dawn.{module_path}"
                    self.integrate_module(full_module_path)
                    integrated_count += 1
                    
                    # Small delay to prevent overwhelming
                    if integrated_count % 10 == 0:
                        time.sleep(0.01)
                        
                except Exception as e:
                    logger.debug(f"Could not integrate module {module_path}: {e}")
        
        logger.info(f"🍄🔗 Successfully integrated {integrated_count}/{total_modules} modules")
        
        # Start cross-module propagation
        if self.cross_module_propagator:
            self._initialize_cross_module_propagation()
    
    def _create_module_context(self, module_path: str) -> ModuleSemanticContext:
        """Create semantic context for a module"""
        # Extract namespace and depth
        path_parts = module_path.split('.')
        namespace = path_parts[1] if len(path_parts) > 1 else 'unknown'
        folder_depth = len(path_parts) - 1  # Subtract 'dawn' prefix
        
        # Extract semantic concepts
        semantic_concepts = self.semantic_mapper.extract_semantic_concepts(module_path)
        
        # Get folder relationships
        folder_key = '.'.join(path_parts[1:]) if len(path_parts) > 1 else ''
        folder_relationships = self.topology_mapper.get_folder_relationships(folder_key)
        
        # Create context
        context = ModuleSemanticContext(
            module_path=module_path,
            module_name=path_parts[-1] if path_parts else 'unknown',
            namespace=namespace,
            folder_depth=folder_depth,
            semantic_concepts=semantic_concepts
        )
        
        # Add parent and child concepts from folder structure
        if folder_relationships:
            parent_key = folder_relationships.get('parent')
            if parent_key:
                context.parent_concepts.update(
                    self.topology_mapper.folder_topology.get(parent_key, {}).get('semantic_concepts', set())
                )
            
            for child_key in folder_relationships.get('children', []):
                context.child_concepts.update(
                    self.topology_mapper.folder_topology.get(child_key, {}).get('semantic_concepts', set())
                )
        
        return context
    
    def _initialize_cross_module_propagation(self):
        """Initialize cross-module semantic propagation"""
        if not self.cross_module_propagator:
            return
        
        # Create namespace bridges
        namespaces = set(context.namespace for context in self.module_contexts.values())
        
        for source_ns in namespaces:
            for target_ns in namespaces:
                if source_ns != target_ns:
                    # Find bridge concepts
                    source_concepts = set()
                    target_concepts = set()
                    
                    for context in self.module_contexts.values():
                        if context.namespace == source_ns:
                            source_concepts.update(context.semantic_concepts)
                        elif context.namespace == target_ns:
                            target_concepts.update(context.semantic_concepts)
                    
                    # Create bridges for overlapping concepts
                    bridge_concepts = source_concepts & target_concepts
                    if bridge_concepts:
                        self.cross_module_propagator.bridge_namespaces(
                            source_ns, target_ns, bridge_concepts
                        )
    
    def _integration_loop(self):
        """Main integration monitoring loop"""
        while self.running:
            try:
                # Update integration statistics
                self._update_integration_stats()
                
                # Trigger periodic cross-module propagation
                if self.cross_module_propagator:
                    self._trigger_periodic_propagation()
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in integration loop: {e}")
                time.sleep(1.0)
    
    def _update_integration_stats(self):
        """Update integration statistics"""
        total_spores = sum(
            wrapper.context.spore_generation_count 
            for wrapper in self.module_wrappers.values()
        )
        
        self.integration_stats['spores_generated'] = total_spores
        
        if self.cross_module_propagator:
            self.integration_stats['cross_module_propagations'] = (
                self.cross_module_propagator.propagation_stats['cross_module_propagations']
            )
    
    def _trigger_periodic_propagation(self):
        """Trigger periodic cross-module propagation"""
        if not self.cross_module_propagator:
            return
        
        # Find active modules (recently accessed)
        current_time = time.time()
        active_modules = [
            context.module_path for context in self.module_contexts.values()
            if current_time - context.last_accessed < 60.0  # Active in last minute
        ]
        
        # Propagate concepts from active modules
        for module_path in active_modules[:5]:  # Limit to prevent spam
            context = self.module_contexts[module_path]
            if context.semantic_concepts:
                self.cross_module_propagator.propagate_across_modules(
                    module_path, context.semantic_concepts
                )
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        stats = self.integration_stats.copy()
        
        # Add detailed breakdown
        stats['modules_by_namespace'] = defaultdict(int)
        stats['concepts_by_namespace'] = defaultdict(set)
        
        for context in self.module_contexts.values():
            stats['modules_by_namespace'][context.namespace] += 1
            stats['concepts_by_namespace'][context.namespace].update(context.semantic_concepts)
        
        # Convert sets to counts for JSON serialization
        stats['concepts_by_namespace'] = {
            ns: len(concepts) for ns, concepts in stats['concepts_by_namespace'].items()
        }
        
        if self.cross_module_propagator:
            stats['cross_module_stats'] = self.cross_module_propagator.propagation_stats
        
        return stats

# Global mycelial module integrator
_mycelial_integrator: Optional[MycelialModuleIntegrator] = None
_integrator_lock = threading.Lock()

def get_mycelial_integrator() -> MycelialModuleIntegrator:
    """Get the global mycelial module integrator"""
    global _mycelial_integrator
    
    with _integrator_lock:
        if _mycelial_integrator is None:
            _mycelial_integrator = MycelialModuleIntegrator()
        return _mycelial_integrator

def start_mycelial_integration():
    """Start mycelial integration for all DAWN modules"""
    if not MYCELIAL_AVAILABLE:
        logger.warning("🍄❌ Mycelial integration not available - mycelial hash map not found")
        return False
    
    integrator = get_mycelial_integrator()
    integrator.start_integration()
    
    # Integrate with DAWN module discovery
    try:
        from dawn import discover_capabilities
        discovered_modules = discover_capabilities()
        integrator.integrate_discovered_modules(discovered_modules)
        
        logger.info("🍄✅ Mycelial integration started for all DAWN modules")
        return True
        
    except ImportError as e:
        logger.warning(f"Could not access DAWN module discovery: {e}")
        return False

def stop_mycelial_integration():
    """Stop mycelial integration"""
    global _mycelial_integrator
    
    with _integrator_lock:
        if _mycelial_integrator:
            _mycelial_integrator.stop_integration()

def integrate_module_with_mycelial(module_path: str, module_obj: Any = None) -> Any:
    """Integrate a specific module with mycelial behavior"""
    integrator = get_mycelial_integrator()
    return integrator.integrate_module(module_path, module_obj)

def get_mycelial_integration_stats() -> Dict[str, Any]:
    """Get mycelial integration statistics"""
    integrator = get_mycelial_integrator()
    return integrator.get_integration_stats()

if __name__ == "__main__":
    # Test mycelial module integration
    logging.basicConfig(level=logging.INFO)
    
    print("🍄🔗 Testing Mycelial Module Integration")
    print("=" * 50)
    
    success = start_mycelial_integration()
    
    if success:
        print("✅ Mycelial integration started successfully")
        
        # Show integration stats
        time.sleep(2)
        stats = get_mycelial_integration_stats()
        
        print("\n📊 Integration Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    else:
        print("❌ Mycelial integration failed")
    
    print("\n🍄🔗 Test complete!")
