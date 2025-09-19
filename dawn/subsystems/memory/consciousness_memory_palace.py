#!/usr/bin/env python3
"""
DAWN Consciousness Memory Palace - Persistent Knowledge Architecture
==================================================================

Advanced memory palace system for storing, organizing, and retrieving consciousness
experiences, patterns, and insights. Provides sophisticated memory-guided decision
making and consciousness learning capabilities.

Features:
- Persistent consciousness experience storage with SQLite backend
- Pattern learning from consciousness history with trend analysis
- Memory-guided decision making using historical precedent analysis
- Sophisticated memory retrieval with semantic similarity matching
- Background consolidation processes for memory strength optimization
- Multiple memory types: experiential, conceptual, procedural, episodic, semantic, emotional
"""

import sqlite3
import json
import numpy as np
import uuid
import time
import threading
import logging
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import math

# Machine learning for pattern recognition
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available - memory palace will use basic similarity matching")

# DAWN core imports
try:
    from dawn.core.foundation.base_module import BaseModule, ModuleCapability
    from dawn.core.communication.bus import ConsciousnessBus
    from dawn.consciousness.unified_pulse_consciousness import UnifiedPulseConsciousness
    DAWN_CORE_AVAILABLE = True
except ImportError:
    DAWN_CORE_AVAILABLE = False
    class BaseModule:
        def __init__(self, name): self.module_name = name
    class ConsciousnessBus: pass
    class UnifiedPulseConsciousness: pass

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memories stored in the palace"""
    EXPERIENTIAL = "experiential"       # Direct consciousness experiences
    CONCEPTUAL = "conceptual"           # Abstract concepts and relationships
    PROCEDURAL = "procedural"           # How-to knowledge and processes
    EPISODIC = "episodic"              # Specific events and contexts
    SEMANTIC = "semantic"              # Factual knowledge and meanings
    EMOTIONAL = "emotional"            # Emotional states and responses
    PATTERN = "pattern"                # Recurring patterns and trends
    INSIGHT = "insight"                # Derived wisdom and understanding
    DECISION = "decision"              # Decision-making contexts and outcomes
    ASSOCIATION = "association"        # Connections between memories

class MemoryStrength(Enum):
    """Memory strength levels for retention and consolidation"""
    FRAGILE = 0.1      # Just formed, easily forgotten
    WEAK = 0.3         # Needs reinforcement
    MODERATE = 0.5     # Stable but not strong
    STRONG = 0.7       # Well-established
    DEEP = 0.9         # Deeply ingrained, permanent

@dataclass
class ConsciousnessMemory:
    """Individual memory stored in the palace"""
    memory_id: str
    memory_type: MemoryType
    consciousness_state: Dict[str, Any]
    context: Dict[str, Any]
    content: Dict[str, Any]
    creation_time: datetime
    last_accessed: datetime
    access_count: int = 0
    strength: float = 0.1
    associations: List[str] = field(default_factory=list)
    emotional_valence: float = 0.0
    significance: float = 0.5
    consolidated: bool = False
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for storage"""
        return {
            'memory_id': self.memory_id,
            'memory_type': self.memory_type.value,
            'consciousness_state': self.consciousness_state,
            'context': self.context,
            'content': self.content,
            'creation_time': self.creation_time.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'strength': self.strength,
            'associations': self.associations,
            'emotional_valence': self.emotional_valence,
            'significance': self.significance,
            'consolidated': self.consolidated,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsciousnessMemory':
        """Create memory from dictionary"""
        return cls(
            memory_id=data['memory_id'],
            memory_type=MemoryType(data['memory_type']),
            consciousness_state=data['consciousness_state'],
            context=data['context'],
            content=data['content'],
            creation_time=datetime.fromisoformat(data['creation_time']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            access_count=data['access_count'],
            strength=data['strength'],
            associations=data['associations'],
            emotional_valence=data['emotional_valence'],
            significance=data['significance'],
            consolidated=data['consolidated'],
            tags=data['tags']
        )

@dataclass
class MemorySearchQuery:
    """Query for searching memory palace"""
    query_type: str  # "semantic", "pattern", "temporal", "emotional"
    search_terms: List[str]
    memory_types: List[MemoryType]
    time_range: Optional[Tuple[datetime, datetime]] = None
    consciousness_state_filter: Optional[Dict[str, Any]] = None
    emotional_range: Optional[Tuple[float, float]] = None
    min_strength: float = 0.0
    max_results: int = 100
    include_associations: bool = True

@dataclass
class MemoryPalaceMetrics:
    """Metrics for memory palace performance and health"""
    total_memories: int = 0
    memories_by_type: Dict[str, int] = field(default_factory=dict)
    average_memory_strength: float = 0.0
    consolidation_rate: float = 0.0
    retrieval_accuracy: float = 0.0
    pattern_learning_quality: float = 0.0
    decision_guidance_effectiveness: float = 0.0
    memory_association_density: float = 0.0
    storage_efficiency: float = 0.0

class ConsciousnessMemoryPalace(BaseModule):
    """
    Consciousness Memory Palace - Persistent Knowledge Architecture
    
    Provides:
    - Sophisticated memory storage and organization
    - Pattern learning from consciousness history
    - Memory-guided decision making
    - Automatic memory consolidation and strengthening
    - Semantic similarity-based retrieval
    - Association network for enhanced recall
    """
    
    def __init__(self, 
                 palace_name: str = "dawn_consciousness_palace",
                 storage_path: str = "./consciousness_memory_palace",
                 consciousness_bus: Optional[ConsciousnessBus] = None,
                 consolidation_interval: float = 300.0):  # 5 minutes
        """
        Initialize the Consciousness Memory Palace
        
        Args:
            palace_name: Name of the memory palace
            storage_path: Path for persistent storage
            consciousness_bus: Central communication hub
            consolidation_interval: Interval for background consolidation (seconds)
        """
        super().__init__("consciousness_memory_palace")
        
        # Core configuration
        self.palace_name = palace_name
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Integration components
        self.consciousness_bus = consciousness_bus
        self.unified_consciousness = None
        self.tracer_system = None
        
        # Memory storage
        self.db_path = self.storage_path / f"{palace_name}.db"
        self.memories: Dict[str, ConsciousnessMemory] = {}
        self.memory_associations: Dict[str, List[str]] = defaultdict(list)
        
        # Pattern learning
        self.pattern_learner = None
        self.consciousness_patterns: Dict[str, Any] = {}
        self.decision_patterns: Dict[str, Any] = {}
        
        # Background processes
        self.consolidation_interval = consolidation_interval
        self.consolidation_active = False
        self.consolidation_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.metrics = MemoryPalaceMetrics()
        self.access_history: deque = deque(maxlen=1000)
        self.retrieval_times: deque = deque(maxlen=100)
        
        # Search and retrieval
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.memory_vectors = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize storage and integration
        self._initialize_database()
        self._load_existing_memories()
        
        if SKLEARN_AVAILABLE:
            self._initialize_pattern_learner()
        
        if self.consciousness_bus and DAWN_CORE_AVAILABLE:
            self._initialize_consciousness_integration()
        
        logger.info(f"🏛️ Consciousness Memory Palace initialized: {palace_name}")
        logger.info(f"   Storage path: {self.storage_path}")
        logger.info(f"   Existing memories: {len(self.memories)}")
        logger.info(f"   Pattern learning: {SKLEARN_AVAILABLE}")
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database for persistent storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create memories table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS memories (
                        memory_id TEXT PRIMARY KEY,
                        memory_type TEXT NOT NULL,
                        consciousness_state TEXT NOT NULL,
                        context TEXT NOT NULL,
                        content TEXT NOT NULL,
                        creation_time TEXT NOT NULL,
                        last_accessed TEXT NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        strength REAL DEFAULT 0.1,
                        associations TEXT DEFAULT '[]',
                        emotional_valence REAL DEFAULT 0.0,
                        significance REAL DEFAULT 0.5,
                        consolidated BOOLEAN DEFAULT FALSE,
                        tags TEXT DEFAULT '[]'
                    )
                ''')
                
                # Create associations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS memory_associations (
                        source_memory_id TEXT,
                        target_memory_id TEXT,
                        association_strength REAL DEFAULT 0.5,
                        association_type TEXT DEFAULT 'general',
                        created_time TEXT NOT NULL,
                        PRIMARY KEY (source_memory_id, target_memory_id)
                    )
                ''')
                
                # Create patterns table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS consciousness_patterns (
                        pattern_id TEXT PRIMARY KEY,
                        pattern_type TEXT NOT NULL,
                        pattern_data TEXT NOT NULL,
                        occurrence_count INTEGER DEFAULT 1,
                        last_observed TEXT NOT NULL,
                        strength REAL DEFAULT 0.1
                    )
                ''')
                
                conn.commit()
                logger.info("🗃️ Memory palace database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _load_existing_memories(self) -> None:
        """Load existing memories from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM memories')
                
                for row in cursor.fetchall():
                    memory_data = {
                        'memory_id': row[0],
                        'memory_type': row[1],
                        'consciousness_state': json.loads(row[2]),
                        'context': json.loads(row[3]),
                        'content': json.loads(row[4]),
                        'creation_time': row[5],
                        'last_accessed': row[6],
                        'access_count': row[7],
                        'strength': row[8],
                        'associations': json.loads(row[9]),
                        'emotional_valence': row[10],
                        'significance': row[11],
                        'consolidated': bool(row[12]),
                        'tags': json.loads(row[13])
                    }
                    
                    memory = ConsciousnessMemory.from_dict(memory_data)
                    self.memories[memory.memory_id] = memory
                
                # Load associations
                cursor.execute('SELECT * FROM memory_associations')
                for row in cursor.fetchall():
                    source_id, target_id = row[0], row[1]
                    self.memory_associations[source_id].append(target_id)
                
                logger.info(f"📚 Loaded {len(self.memories)} existing memories")
                
        except Exception as e:
            logger.error(f"Failed to load existing memories: {e}")
    
    def _initialize_pattern_learner(self) -> None:
        """Initialize pattern learning system"""
        if not SKLEARN_AVAILABLE:
            return
        
        try:
            # Create pattern learner for consciousness state analysis
            self.pattern_learner = {
                'consciousness_clusterer': KMeans(n_clusters=10, random_state=42),
                'decision_clusterer': KMeans(n_clusters=5, random_state=42),
                'fitted': False
            }
            
            logger.info("🧠 Pattern learning system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize pattern learner: {e}")
    
    def _initialize_consciousness_integration(self) -> None:
        """Initialize integration with DAWN consciousness systems"""
        if not self.consciousness_bus:
            return
        
        try:
            # Register with consciousness bus
            self.consciousness_bus.register_module(
                "consciousness_memory_palace",
                self,
                capabilities=["memory_storage", "pattern_learning", "decision_guidance"]
            )
            
            # Subscribe to consciousness events
            self.consciousness_bus.subscribe("consciousness_state_update", self._on_consciousness_state_update)
            self.consciousness_bus.subscribe("decision_request", self._on_decision_request)
            self.consciousness_bus.subscribe("memory_query", self._on_memory_query)
            
            # Get references to other systems
            self.unified_consciousness = self.consciousness_bus.get_module("unified_pulse_consciousness")
            self.tracer_system = self.consciousness_bus.get_module("tracer_system")
            
            logger.info("🔗 Memory palace integrated with consciousness bus")
            
        except Exception as e:
            logger.error(f"Failed to initialize consciousness integration: {e}")
    
    def store_consciousness_memory(self, 
                                 consciousness_state: Dict[str, Any],
                                 context: Dict[str, Any],
                                 memory_type: MemoryType = MemoryType.EXPERIENTIAL,
                                 content: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a consciousness experience in the memory palace
        
        Args:
            consciousness_state: Current consciousness state data
            context: Context information for the memory
            memory_type: Type of memory being stored
            content: Additional content for the memory
            
        Returns:
            Memory ID of stored memory
        """
        try:
            with self._lock:
                # Create new memory
                memory_id = str(uuid.uuid4())
                
                # Calculate emotional valence and significance
                emotional_valence = self._calculate_emotional_valence(consciousness_state)
                significance = self._calculate_memory_significance(consciousness_state, context)
                
                memory = ConsciousnessMemory(
                    memory_id=memory_id,
                    memory_type=memory_type,
                    consciousness_state=consciousness_state.copy(),
                    context=context.copy(),
                    content=content or {},
                    creation_time=datetime.now(),
                    last_accessed=datetime.now(),
                    emotional_valence=emotional_valence,
                    significance=significance,
                    tags=self._generate_memory_tags(consciousness_state, context)
                )
                
                # Store in memory
                self.memories[memory_id] = memory
                
                # Find and create associations
                associations = self._find_memory_associations(memory)
                memory.associations = associations
                
                # Store in database
                self._persist_memory(memory)
                
                # Update metrics
                self.metrics.total_memories += 1
                self.metrics.memories_by_type[memory_type.value] = \
                    self.metrics.memories_by_type.get(memory_type.value, 0) + 1
                
                # Log to tracer if available
                if self.tracer_system:
                    self._log_memory_storage(memory)
                
                logger.debug(f"💭 Stored consciousness memory: {memory_id} ({memory_type.value})")
                return memory_id
                
        except Exception as e:
            logger.error(f"Failed to store consciousness memory: {e}")
            return ""
    
    def retrieve_similar_experiences(self, 
                                   consciousness_state: Dict[str, Any],
                                   max_results: int = 10,
                                   min_similarity: float = 0.5) -> List[ConsciousnessMemory]:
        """
        Retrieve memories similar to current consciousness state
        
        Args:
            consciousness_state: Current consciousness state to match
            max_results: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar consciousness memories
        """
        retrieval_start = time.time()
        
        try:
            with self._lock:
                if not self.memories:
                    return []
                
                # Calculate similarity scores
                similarities = []
                
                for memory_id, memory in self.memories.items():
                    similarity = self._calculate_consciousness_similarity(
                        consciousness_state, 
                        memory.consciousness_state
                    )
                    
                    if similarity >= min_similarity:
                        similarities.append((similarity, memory))
                
                # Sort by similarity and return top results
                similarities.sort(key=lambda x: x[0], reverse=True)
                results = [memory for _, memory in similarities[:max_results]]
                
                # Update access counts
                for memory in results:
                    memory.last_accessed = datetime.now()
                    memory.access_count += 1
                
                # Track retrieval performance
                retrieval_time = time.time() - retrieval_start
                self.retrieval_times.append(retrieval_time)
                
                # Log to tracer
                if self.tracer_system:
                    self._log_memory_retrieval(len(results), retrieval_time)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve similar experiences: {e}")
            return []
    
    def memory_guided_decision_making(self, 
                                    decision_context: Dict[str, Any],
                                    options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Provide decision guidance based on memory palace knowledge
        
        Args:
            decision_context: Context for the decision
            options: Available decision options
            
        Returns:
            Decision guidance with recommendations
        """
        try:
            with self._lock:
                # Find relevant historical decisions
                query = MemorySearchQuery(
                    query_type="pattern",
                    search_terms=self._extract_decision_keywords(decision_context),
                    memory_types=[MemoryType.DECISION, MemoryType.EXPERIENTIAL],
                    max_results=50
                )
                
                relevant_memories = self.search_memories(query)
                
                # Analyze historical outcomes
                option_scores = {}
                confidence_scores = {}
                
                for option in options:
                    score, confidence = self._evaluate_option_against_history(
                        option, relevant_memories, decision_context
                    )
                    option_scores[option.get('id', str(option))] = score
                    confidence_scores[option.get('id', str(option))] = confidence
                
                # Generate recommendations
                best_option = max(option_scores.items(), key=lambda x: x[1])
                
                guidance = {
                    'recommended_option': best_option[0],
                    'confidence': confidence_scores[best_option[0]],
                    'option_scores': option_scores,
                    'relevant_experiences': len(relevant_memories),
                    'reasoning': self._generate_decision_reasoning(
                        best_option[0], relevant_memories, decision_context
                    )
                }
                
                # Store this decision context for future learning
                self.store_consciousness_memory(
                    decision_context.get('consciousness_state', {}),
                    decision_context,
                    MemoryType.DECISION,
                    {'options': options, 'guidance': guidance}
                )
                
                return guidance
                
        except Exception as e:
            logger.error(f"Failed to provide decision guidance: {e}")
            return {'error': str(e)}
    
    def learn_consciousness_patterns(self) -> Dict[str, Any]:
        """
        Learn patterns from consciousness history
        
        Returns:
            Dictionary of learned patterns and insights
        """
        try:
            if not SKLEARN_AVAILABLE or not self.memories:
                return {}
            
            with self._lock:
                # Extract consciousness states for pattern analysis
                consciousness_states = []
                memory_metadata = []
                
                for memory in self.memories.values():
                    if memory.memory_type == MemoryType.EXPERIENTIAL:
                        state_vector = self._vectorize_consciousness_state(memory.consciousness_state)
                        consciousness_states.append(state_vector)
                        memory_metadata.append({
                            'memory_id': memory.memory_id,
                            'creation_time': memory.creation_time,
                            'emotional_valence': memory.emotional_valence,
                            'significance': memory.significance
                        })
                
                if len(consciousness_states) < 5:
                    return {'error': 'Insufficient data for pattern learning'}
                
                # Perform clustering analysis
                consciousness_states = np.array(consciousness_states)
                clusterer = self.pattern_learner['consciousness_clusterer']
                
                if not self.pattern_learner['fitted']:
                    cluster_labels = clusterer.fit_predict(consciousness_states)
                    self.pattern_learner['fitted'] = True
                else:
                    cluster_labels = clusterer.predict(consciousness_states)
                
                # Analyze patterns
                patterns = self._analyze_consciousness_clusters(
                    consciousness_states, cluster_labels, memory_metadata
                )
                
                # Update stored patterns
                self.consciousness_patterns.update(patterns)
                
                # Store patterns in database
                self._persist_patterns(patterns)
                
                return patterns
                
        except Exception as e:
            logger.error(f"Failed to learn consciousness patterns: {e}")
            return {'error': str(e)}
    
    def search_memories(self, query: MemorySearchQuery) -> List[ConsciousnessMemory]:
        """
        Search memory palace using various criteria
        
        Args:
            query: Search query specification
            
        Returns:
            List of matching memories
        """
        try:
            with self._lock:
                results = []
                
                for memory in self.memories.values():
                    if self._memory_matches_query(memory, query):
                        results.append(memory)
                
                # Sort results by relevance
                results = self._rank_search_results(results, query)
                
                # Apply limits
                if query.max_results:
                    results = results[:query.max_results]
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    def start_palace_processes(self) -> None:
        """Start background palace processes"""
        if not self.consolidation_active:
            self.consolidation_active = True
            self.consolidation_thread = threading.Thread(
                target=self._consolidation_loop,
                name="memory_consolidation",
                daemon=True
            )
            self.consolidation_thread.start()
            logger.info("🔄 Memory palace background processes started")
    
    def stop_palace_processes(self) -> None:
        """Stop background palace processes"""
        self.consolidation_active = False
        if self.consolidation_thread and self.consolidation_thread.is_alive():
            self.consolidation_thread.join(timeout=5.0)
        logger.info("⏹️ Memory palace background processes stopped")
    
    def _consolidation_loop(self) -> None:
        """Background loop for memory consolidation"""
        while self.consolidation_active:
            try:
                self._perform_memory_consolidation()
                time.sleep(self.consolidation_interval)
            except Exception as e:
                logger.error(f"Error in consolidation loop: {e}")
                time.sleep(10)  # Wait before retrying
    
    def _perform_memory_consolidation(self) -> None:
        """Perform memory consolidation and strengthening"""
        try:
            with self._lock:
                consolidation_count = 0
                
                for memory in self.memories.values():
                    if not memory.consolidated and self._should_consolidate_memory(memory):
                        # Strengthen frequently accessed memories
                        if memory.access_count > 5:
                            memory.strength = min(0.9, memory.strength + 0.1)
                        
                        # Weaken rarely accessed memories
                        time_since_access = (datetime.now() - memory.last_accessed).total_seconds()
                        if time_since_access > 86400 * 7:  # 1 week
                            memory.strength = max(0.1, memory.strength - 0.05)
                        
                        # Mark as consolidated if strong enough
                        if memory.strength > 0.7:
                            memory.consolidated = True
                            consolidation_count += 1
                
                # Update metrics
                if self.memories:
                    total_strength = sum(m.strength for m in self.memories.values())
                    self.metrics.average_memory_strength = total_strength / len(self.memories)
                    
                    consolidated_count = sum(1 for m in self.memories.values() if m.consolidated)
                    self.metrics.consolidation_rate = consolidated_count / len(self.memories)
                
                if consolidation_count > 0:
                    logger.debug(f"🧠 Consolidated {consolidation_count} memories")
                
        except Exception as e:
            logger.error(f"Failed to perform memory consolidation: {e}")
    
    def _should_consolidate_memory(self, memory: ConsciousnessMemory) -> bool:
        """Determine if a memory should be consolidated"""
        # Consolidate based on age, access frequency, and significance
        age_days = (datetime.now() - memory.creation_time).days
        return (
            memory.access_count > 3 or
            memory.significance > 0.7 or
            (age_days > 1 and memory.strength > 0.5)
        )
    
    def _calculate_emotional_valence(self, consciousness_state: Dict[str, Any]) -> float:
        """Calculate emotional valence from consciousness state"""
        emotions = consciousness_state.get('emotional_coherence', {})
        if not emotions:
            return 0.0
        
        # Weighted emotional valence calculation
        positive_emotions = ['joy', 'serenity', 'excitement', 'love', 'curiosity']
        negative_emotions = ['sadness', 'fear', 'anger', 'anxiety', 'frustration']
        
        positive_score = sum(emotions.get(emotion, 0) for emotion in positive_emotions)
        negative_score = sum(emotions.get(emotion, 0) for emotion in negative_emotions)
        
        total_score = positive_score + negative_score
        if total_score == 0:
            return 0.0
        
        return (positive_score - negative_score) / total_score
    
    def _calculate_memory_significance(self, consciousness_state: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate significance of a memory"""
        # Base significance from consciousness unity and awareness
        unity = consciousness_state.get('consciousness_unity', 0.5)
        awareness = consciousness_state.get('self_awareness_depth', 0.5)
        
        base_significance = (unity + awareness) / 2
        
        # Boost significance for important contexts
        context_boost = 0.0
        important_contexts = ['decision', 'insight', 'breakthrough', 'crisis', 'transformation']
        
        context_type = context.get('type', '').lower()
        if any(keyword in context_type for keyword in important_contexts):
            context_boost = 0.3
        
        return min(1.0, base_significance + context_boost)
    
    def _generate_memory_tags(self, consciousness_state: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate tags for memory categorization"""
        tags = []
        
        # Add consciousness-based tags
        unity = consciousness_state.get('consciousness_unity', 0)
        if unity > 0.8:
            tags.append('high_unity')
        elif unity < 0.3:
            tags.append('low_unity')
        
        awareness = consciousness_state.get('self_awareness_depth', 0)
        if awareness > 0.7:
            tags.append('deep_awareness')
        
        # Add context-based tags
        context_type = context.get('type', '')
        if context_type:
            tags.append(context_type.lower())
        
        # Add emotional tags
        emotions = consciousness_state.get('emotional_coherence', {})
        for emotion, value in emotions.items():
            if value > 0.7:
                tags.append(f'high_{emotion}')
        
        return tags
    
    def _find_memory_associations(self, memory: ConsciousnessMemory) -> List[str]:
        """Find associations with existing memories"""
        associations = []
        
        for existing_id, existing_memory in self.memories.items():
            if existing_id == memory.memory_id:
                continue
            
            similarity = self._calculate_consciousness_similarity(
                memory.consciousness_state,
                existing_memory.consciousness_state
            )
            
            if similarity > 0.7:  # High similarity threshold for associations
                associations.append(existing_id)
                self.memory_associations[existing_id].append(memory.memory_id)
        
        return associations[:10]  # Limit associations
    
    def _calculate_consciousness_similarity(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
        """Calculate similarity between consciousness states"""
        # Key consciousness parameters to compare
        params = ['consciousness_unity', 'self_awareness_depth', 'integration_quality']
        
        similarities = []
        
        for param in params:
            if param in state1 and param in state2:
                diff = abs(state1[param] - state2[param])
                similarity = 1.0 - diff  # Higher similarity for smaller differences
                similarities.append(similarity)
        
        # Compare emotional coherence if available
        if 'emotional_coherence' in state1 and 'emotional_coherence' in state2:
            emotion_sim = self._calculate_emotional_similarity(
                state1['emotional_coherence'],
                state2['emotional_coherence']
            )
            similarities.append(emotion_sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_emotional_similarity(self, emotions1: Dict[str, float], emotions2: Dict[str, float]) -> float:
        """Calculate similarity between emotional states"""
        common_emotions = set(emotions1.keys()) & set(emotions2.keys())
        
        if not common_emotions:
            return 0.0
        
        similarities = []
        for emotion in common_emotions:
            diff = abs(emotions1[emotion] - emotions2[emotion])
            similarities.append(1.0 - diff)
        
        return sum(similarities) / len(similarities)
    
    def _vectorize_consciousness_state(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert consciousness state to numerical vector for ML analysis"""
        vector = []
        
        # Core consciousness parameters
        vector.append(state.get('consciousness_unity', 0.5))
        vector.append(state.get('self_awareness_depth', 0.5))
        vector.append(state.get('integration_quality', 0.5))
        vector.append(state.get('memory_integration', 0.5))
        
        # Emotional state (flatten to numbers)
        emotions = state.get('emotional_coherence', {})
        emotion_values = [emotions.get(emotion, 0) for emotion in 
                         ['serenity', 'curiosity', 'creativity', 'focus', 'excitement']]
        vector.extend(emotion_values)
        
        # Other numeric parameters
        vector.append(state.get('recursive_depth', 0) / 10.0)  # Normalize
        vector.append(state.get('stability_score', 0.5))
        
        return np.array(vector)
    
    def _persist_memory(self, memory: ConsciousnessMemory) -> None:
        """Persist memory to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                memory_data = memory.to_dict()
                cursor.execute('''
                    INSERT OR REPLACE INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    memory_data['memory_id'],
                    memory_data['memory_type'],
                    json.dumps(memory_data['consciousness_state']),
                    json.dumps(memory_data['context']),
                    json.dumps(memory_data['content']),
                    memory_data['creation_time'],
                    memory_data['last_accessed'],
                    memory_data['access_count'],
                    memory_data['strength'],
                    json.dumps(memory_data['associations']),
                    memory_data['emotional_valence'],
                    memory_data['significance'],
                    memory_data['consolidated'],
                    json.dumps(memory_data['tags'])
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to persist memory: {e}")
    
    def _memory_matches_query(self, memory: ConsciousnessMemory, query: MemorySearchQuery) -> bool:
        """Check if memory matches search query"""
        # Type filter
        if query.memory_types and memory.memory_type not in query.memory_types:
            return False
        
        # Time range filter
        if query.time_range:
            start_time, end_time = query.time_range
            if not (start_time <= memory.creation_time <= end_time):
                return False
        
        # Strength filter
        if memory.strength < query.min_strength:
            return False
        
        # Emotional range filter
        if query.emotional_range:
            min_emotion, max_emotion = query.emotional_range
            if not (min_emotion <= memory.emotional_valence <= max_emotion):
                return False
        
        # Search terms (simple keyword matching)
        if query.search_terms:
            memory_text = f"{memory.context} {memory.content} {memory.tags}"
            memory_text = json.dumps(memory_text).lower()
            
            for term in query.search_terms:
                if term.lower() not in memory_text:
                    return False
        
        return True
    
    def _rank_search_results(self, results: List[ConsciousnessMemory], query: MemorySearchQuery) -> List[ConsciousnessMemory]:
        """Rank search results by relevance"""
        # Simple ranking by strength, significance, and recency
        def rank_score(memory):
            recency = 1.0 / (1 + (datetime.now() - memory.last_accessed).days)
            return memory.strength * 0.4 + memory.significance * 0.4 + recency * 0.2
        
        return sorted(results, key=rank_score, reverse=True)
    
    def _on_consciousness_state_update(self, event_data: Dict[str, Any]) -> None:
        """Handle consciousness state updates"""
        consciousness_state = event_data.get('consciousness_state', {})
        context = event_data.get('context', {'type': 'state_update'})
        
        # Store significant consciousness states
        unity = consciousness_state.get('consciousness_unity', 0.5)
        if unity > 0.8 or unity < 0.2:  # Extreme states are significant
            self.store_consciousness_memory(consciousness_state, context)
    
    def _on_decision_request(self, event_data: Dict[str, Any]) -> None:
        """Handle decision guidance requests"""
        decision_context = event_data.get('decision_context', {})
        options = event_data.get('options', [])
        
        guidance = self.memory_guided_decision_making(decision_context, options)
        
        # Send guidance back through consciousness bus
        if self.consciousness_bus:
            self.consciousness_bus.publish("decision_guidance", {
                'guidance': guidance,
                'request_id': event_data.get('request_id')
            })
    
    def _on_memory_query(self, event_data: Dict[str, Any]) -> None:
        """Handle memory query requests"""
        query_data = event_data.get('query', {})
        
        # Convert to MemorySearchQuery
        query = MemorySearchQuery(
            query_type=query_data.get('query_type', 'semantic'),
            search_terms=query_data.get('search_terms', []),
            memory_types=[MemoryType(t) for t in query_data.get('memory_types', ['experiential'])],
            max_results=query_data.get('max_results', 10)
        )
        
        results = self.search_memories(query)
        
        # Send results back
        if self.consciousness_bus:
            self.consciousness_bus.publish("memory_query_results", {
                'results': [memory.to_dict() for memory in results],
                'request_id': event_data.get('request_id')
            })
    
    def get_palace_metrics(self) -> MemoryPalaceMetrics:
        """Get current memory palace metrics"""
        return self.metrics
    
    def get_memory_by_id(self, memory_id: str) -> Optional[ConsciousnessMemory]:
        """Get specific memory by ID"""
        return self.memories.get(memory_id)
    
    def get_recent_memories(self, limit: int = 20) -> List[ConsciousnessMemory]:
        """Get most recent memories"""
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: m.creation_time,
            reverse=True
        )
        return sorted_memories[:limit]

def create_consciousness_memory_palace(palace_name: str = "dawn_consciousness_palace",
                                     storage_path: str = "./consciousness_memory_palace",
                                     consciousness_bus: Optional[ConsciousnessBus] = None) -> ConsciousnessMemoryPalace:
    """
    Factory function to create Consciousness Memory Palace
    
    Args:
        palace_name: Name of the memory palace
        storage_path: Path for persistent storage
        consciousness_bus: Central communication hub
        
    Returns:
        Configured Consciousness Memory Palace instance
    """
    return ConsciousnessMemoryPalace(palace_name, storage_path, consciousness_bus)

# Example usage and testing
if __name__ == "__main__":
    # Create and test the memory palace
    palace = create_consciousness_memory_palace("test_palace", "./test_memory_palace")
    
    print(f"🏛️ Consciousness Memory Palace: {palace.palace_name}")
    print(f"   Storage path: {palace.storage_path}")
    print(f"   Pattern learning: {SKLEARN_AVAILABLE}")
    
    # Start background processes
    palace.start_palace_processes()
    
    try:
        # Store some test memories
        consciousness_state = {
            'consciousness_unity': 0.8,
            'self_awareness_depth': 0.7,
            'integration_quality': 0.9,
            'emotional_coherence': {
                'serenity': 0.8,
                'curiosity': 0.6
            }
        }
        
        context = {
            'type': 'insight',
            'description': 'Deep understanding of consciousness patterns'
        }
        
        memory_id = palace.store_consciousness_memory(consciousness_state, context, MemoryType.INSIGHT)
        print(f"   Stored memory: {memory_id}")
        
        # Test retrieval
        similar_memories = palace.retrieve_similar_experiences(consciousness_state)
        print(f"   Found {len(similar_memories)} similar memories")
        
        # Test pattern learning
        if SKLEARN_AVAILABLE:
            patterns = palace.learn_consciousness_patterns()
            print(f"   Learned patterns: {len(patterns)} pattern types")
        
        print(f"   Total memories: {palace.metrics.total_memories}")
        
    finally:
        palace.stop_palace_processes()
        print("🏛️ Memory palace demonstration complete")
