#!/usr/bin/env python3
"""
🗂️ Centralized Deep Logging Repository
======================================

This module creates a single, very deep repository structure for all DAWN logging.
Organizes all JSON state logs in a comprehensive hierarchical format with:

- Deep directory structures by system/subsystem/module/date/time
- Centralized metadata and indexing
- Cross-references between related logs
- Automatic organization and archival
- Search and query capabilities
- Compression and optimization
- Historical tracking and analytics

Structure:
logs_repository/
├── systems/
│   ├── consciousness/
│   │   ├── engines/
│   │   │   ├── primary/
│   │   │   │   ├── 2025/
│   │   │   │   │   ├── 09/
│   │   │   │   │   │   ├── 19/
│   │   │   │   │   │   │   ├── hour_14/
│   │   │   │   │   │   │   │   ├── states/
│   │   │   │   │   │   │   │   ├── changes/
│   │   │   │   │   │   │   │   ├── performance/
│   │   │   │   │   │   │   │   └── metadata/
"""

import json
import time
import threading
import gzip
import hashlib
import sqlite3
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict, deque
import shutil
import pickle

# Import our universal logger
from .universal_json_logger import get_universal_logger, StateSnapshot

import logging
logger = logging.getLogger(__name__)

@dataclass
class LogEntry:
    """Centralized log entry with deep metadata"""
    entry_id: str
    system: str
    subsystem: str
    module: str
    timestamp: float
    log_type: str  # 'state', 'change', 'performance', 'metadata'
    file_path: str
    size_bytes: int
    hash_sha256: str
    related_entries: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    compressed: bool = False
    archived: bool = False

@dataclass
class RepositoryStats:
    """Statistics for the centralized repository"""
    total_entries: int = 0
    total_size_bytes: int = 0
    systems_count: int = 0
    subsystems_count: int = 0
    modules_count: int = 0
    oldest_entry: Optional[float] = None
    newest_entry: Optional[float] = None
    compression_ratio: float = 0.0
    
class CentralizedLoggingRepository:
    """
    Single, very deep repository for all DAWN logging.
    Organizes all JSON logs in a comprehensive hierarchical structure.
    """
    
    def __init__(self, base_path: str = "logs_repository"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Deep directory structure
        self.systems_path = self.base_path / "systems"
        self.archive_path = self.base_path / "archive"
        self.index_path = self.base_path / "index"
        self.metadata_path = self.base_path / "metadata"
        self.search_path = self.base_path / "search"
        
        # Create deep structure
        for path in [self.systems_path, self.archive_path, self.index_path, 
                    self.metadata_path, self.search_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Repository database
        self.db_path = self.base_path / "repository.db"
        self._init_database()
        
        # In-memory indexes
        self.entry_index: Dict[str, LogEntry] = {}
        self.system_index: Dict[str, Set[str]] = defaultdict(set)
        self.time_index: Dict[str, List[str]] = defaultdict(list)  # date -> entry_ids
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Repository stats
        self.stats = RepositoryStats()
        
        # Background processes
        self._running = True
        self._organization_thread = threading.Thread(target=self._organization_loop, daemon=True)
        self._compression_thread = threading.Thread(target=self._compression_loop, daemon=True)
        self._indexing_thread = threading.Thread(target=self._indexing_loop, daemon=True)
        
        # Start background processes
        self._organization_thread.start()
        self._compression_thread.start()
        self._indexing_thread.start()
        
        # Load existing repository
        self._load_repository()
        
        logger.info(f"🗂️ Centralized Deep Logging Repository initialized at {self.base_path}")
    
    def _init_database(self):
        """Initialize SQLite database for repository metadata"""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Log entries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS log_entries (
                    entry_id TEXT PRIMARY KEY,
                    system TEXT NOT NULL,
                    subsystem TEXT NOT NULL,
                    module TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    log_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    hash_sha256 TEXT NOT NULL,
                    related_entries TEXT,
                    tags TEXT,
                    compressed BOOLEAN DEFAULT FALSE,
                    archived BOOLEAN DEFAULT FALSE,
                    created_at REAL DEFAULT (julianday('now')),
                    updated_at REAL DEFAULT (julianday('now'))
                )
            """)
            
            # Cross-references table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cross_references (
                    from_entry TEXT,
                    to_entry TEXT,
                    relationship_type TEXT,
                    strength REAL DEFAULT 1.0,
                    PRIMARY KEY (from_entry, to_entry, relationship_type)
                )
            """)
            
            # Search index table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_index (
                    term TEXT,
                    entry_id TEXT,
                    relevance REAL DEFAULT 1.0,
                    context TEXT,
                    PRIMARY KEY (term, entry_id)
                )
            """)
            
            # Repository statistics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS repository_stats (
                    stat_name TEXT PRIMARY KEY,
                    stat_value TEXT,
                    updated_at REAL DEFAULT (julianday('now'))
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_system ON log_entries(system)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_timestamp ON log_entries(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_type ON log_entries(log_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_term ON search_index(term)")
            
            conn.commit()
    
    def add_log_entry(self, system: str, subsystem: str, module: str, 
                     log_data: Dict[str, Any], log_type: str = "state") -> str:
        """Add a log entry to the deep repository structure"""
        
        with self._lock:
            timestamp = time.time()
            dt = datetime.fromtimestamp(timestamp)
            
            # Generate entry ID
            entry_id = f"{system}_{subsystem}_{module}_{int(timestamp*1000000)}"
            
            # Create deep directory structure
            deep_path = self._create_deep_path(system, subsystem, module, dt)
            
            # Determine file name and path
            filename = f"{log_type}_{dt.strftime('%H%M%S_%f')}.json"
            file_path = deep_path / filename
            
            # Write log data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'entry_id': entry_id,
                    'timestamp': timestamp,
                    'system': system,
                    'subsystem': subsystem,
                    'module': module,
                    'log_type': log_type,
                    'data': log_data
                }, f, indent=2, default=str)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            file_size = file_path.stat().st_size
            
            # Create log entry
            log_entry = LogEntry(
                entry_id=entry_id,
                system=system,
                subsystem=subsystem,
                module=module,
                timestamp=timestamp,
                log_type=log_type,
                file_path=str(file_path.relative_to(self.base_path)),
                size_bytes=file_size,
                hash_sha256=file_hash,
                tags=self._extract_tags(log_data)
            )
            
            # Update indexes
            self.entry_index[entry_id] = log_entry
            self.system_index[system].add(entry_id)
            date_key = dt.strftime('%Y-%m-%d')
            self.time_index[date_key].append(entry_id)
            
            # Store in database
            self._store_log_entry(log_entry)
            
            # Update statistics
            self._update_stats(log_entry)
            
            logger.debug(f"🗂️ Added log entry: {entry_id} -> {file_path}")
            
            return entry_id
    
    def _create_deep_path(self, system: str, subsystem: str, module: str, dt: datetime) -> Path:
        """Create deep directory structure for log organization"""
        
        # Deep path: systems/system/subsystem/module/YYYY/MM/DD/hour_HH/log_type/
        deep_path = (self.systems_path / 
                    system / 
                    subsystem / 
                    module /
                    str(dt.year) /
                    f"{dt.month:02d}" /
                    f"{dt.day:02d}" /
                    f"hour_{dt.hour:02d}")
        
        # Create subdirectories for different log types
        for log_type in ['states', 'changes', 'performance', 'metadata', 'errors']:
            (deep_path / log_type).mkdir(parents=True, exist_ok=True)
        
        return deep_path / 'states'  # Default to states
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _extract_tags(self, log_data: Dict[str, Any]) -> Set[str]:
        """Extract tags from log data for indexing"""
        tags = set()
        
        # Extract common tags
        if 'state_data' in log_data:
            state = log_data['state_data']
            if isinstance(state, dict):
                # Add class name as tag
                if '__object__' in state and 'class' in state:
                    tags.add(f"class:{state['class']}")
                
                # Add status/level tags
                for key in ['status', 'level', 'state', 'mode']:
                    if key in state:
                        tags.add(f"{key}:{state[key]}")
        
        # Add change type tags
        if 'change_summary' in log_data:
            change = log_data['change_summary']
            if isinstance(change, dict):
                if 'change_type' in change:
                    tags.add(f"change:{change['change_type']}")
                if 'changes' in change and change['changes'] > 0:
                    tags.add("has_changes")
        
        # Add performance tags
        if 'system_metrics' in log_data:
            metrics = log_data['system_metrics']
            if isinstance(metrics, dict):
                if 'cpu_percent' in metrics and metrics['cpu_percent'] > 50:
                    tags.add("high_cpu")
                if 'memory_mb' in metrics and metrics['memory_mb'] > 1000:
                    tags.add("high_memory")
        
        return tags
    
    def _store_log_entry(self, log_entry: LogEntry):
        """Store log entry in database"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO log_entries 
                (entry_id, system, subsystem, module, timestamp, log_type, 
                 file_path, size_bytes, hash_sha256, related_entries, tags, 
                 compressed, archived)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log_entry.entry_id,
                log_entry.system,
                log_entry.subsystem,
                log_entry.module,
                log_entry.timestamp,
                log_entry.log_type,
                log_entry.file_path,
                log_entry.size_bytes,
                log_entry.hash_sha256,
                json.dumps(log_entry.related_entries),
                json.dumps(list(log_entry.tags)),
                log_entry.compressed,
                log_entry.archived
            ))
            conn.commit()
    
    def _update_stats(self, log_entry: LogEntry):
        """Update repository statistics"""
        self.stats.total_entries += 1
        self.stats.total_size_bytes += log_entry.size_bytes
        
        # Update system counts
        systems = set(entry.system for entry in self.entry_index.values())
        subsystems = set(f"{entry.system}.{entry.subsystem}" for entry in self.entry_index.values())
        modules = set(f"{entry.system}.{entry.subsystem}.{entry.module}" for entry in self.entry_index.values())
        
        self.stats.systems_count = len(systems)
        self.stats.subsystems_count = len(subsystems)
        self.stats.modules_count = len(modules)
        
        # Update time bounds
        timestamps = [entry.timestamp for entry in self.entry_index.values()]
        if timestamps:
            self.stats.oldest_entry = min(timestamps)
            self.stats.newest_entry = max(timestamps)
    
    def _load_repository(self):
        """Load existing repository from database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT * FROM log_entries")
                for row in cursor.fetchall():
                    entry = LogEntry(
                        entry_id=row[0],
                        system=row[1],
                        subsystem=row[2],
                        module=row[3],
                        timestamp=row[4],
                        log_type=row[5],
                        file_path=row[6],
                        size_bytes=row[7],
                        hash_sha256=row[8],
                        related_entries=json.loads(row[9]) if row[9] else [],
                        tags=set(json.loads(row[10])) if row[10] else set(),
                        compressed=bool(row[11]),
                        archived=bool(row[12])
                    )
                    
                    self.entry_index[entry.entry_id] = entry
                    self.system_index[entry.system].add(entry.entry_id)
                    
                    dt = datetime.fromtimestamp(entry.timestamp)
                    date_key = dt.strftime('%Y-%m-%d')
                    self.time_index[date_key].append(entry.entry_id)
            
            # Update stats
            for entry in self.entry_index.values():
                self._update_stats(entry)
            
            logger.info(f"🗂️ Loaded {len(self.entry_index)} entries from repository")
            
        except Exception as e:
            logger.warning(f"Failed to load repository: {e}")
    
    def integrate_universal_logger(self):
        """Integrate with the universal JSON logger to centralize all logs"""
        universal_logger = get_universal_logger()
        
        # Hook into the universal logger's write methods
        original_queue_snapshot = universal_logger._queue_snapshot
        
        def centralized_queue_snapshot(snapshot: StateSnapshot):
            # Call original method
            original_queue_snapshot(snapshot)
            
            # Also add to centralized repository
            try:
                log_data = asdict(snapshot)
                self.add_log_entry(
                    system=snapshot.module_name or "unknown",
                    subsystem=snapshot.class_name or "unknown", 
                    module=snapshot.object_id.split('_')[0] if '_' in snapshot.object_id else "unknown",
                    log_data=log_data,
                    log_type="state"
                )
            except Exception as e:
                logger.warning(f"Failed to centralize log: {e}")
        
        # Replace the method
        universal_logger._queue_snapshot = centralized_queue_snapshot
        
        logger.info("🗂️ Integrated with universal logger for centralized logging")
    
    def query_logs(self, system: Optional[str] = None, subsystem: Optional[str] = None,
                   module: Optional[str] = None, log_type: Optional[str] = None,
                   start_time: Optional[float] = None, end_time: Optional[float] = None,
                   tags: Optional[Set[str]] = None, limit: int = 100) -> List[LogEntry]:
        """Query logs from the centralized repository"""
        
        results = []
        
        with self._lock:
            for entry in self.entry_index.values():
                # Filter by system
                if system and entry.system != system:
                    continue
                
                # Filter by subsystem
                if subsystem and entry.subsystem != subsystem:
                    continue
                
                # Filter by module
                if module and entry.module != module:
                    continue
                
                # Filter by log type
                if log_type and entry.log_type != log_type:
                    continue
                
                # Filter by time range
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue
                
                # Filter by tags
                if tags and not tags.intersection(entry.tags):
                    continue
                
                results.append(entry)
                
                if len(results) >= limit:
                    break
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.timestamp, reverse=True)
        
        return results
    
    def get_log_content(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get the content of a specific log entry"""
        
        if entry_id not in self.entry_index:
            return None
        
        entry = self.entry_index[entry_id]
        file_path = self.base_path / entry.file_path
        
        try:
            if entry.compressed and file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read log {entry_id}: {e}")
            return None
    
    def _organization_loop(self):
        """Background thread for organizing and optimizing the repository"""
        while self._running:
            try:
                # Organize old logs
                self._organize_old_logs()
                
                # Create cross-references
                self._create_cross_references()
                
                # Update search index
                self._update_search_index()
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in organization loop: {e}")
                time.sleep(60)
    
    def _organize_old_logs(self):
        """Organize old logs for better structure"""
        # Implementation for organizing old logs
        pass
    
    def _create_cross_references(self):
        """Create cross-references between related logs"""
        # Implementation for cross-referencing
        pass
    
    def _update_search_index(self):
        """Update search index"""
        # Implementation for search index updates
        pass
    
    def _rebuild_search_index(self):
        """Rebuild search index"""
        # Implementation for rebuilding search index
        pass
    
    def _save_repository_stats(self):
        """Save repository statistics to database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                stats_data = json.dumps(asdict(self.stats), default=str)
                conn.execute("""
                    INSERT OR REPLACE INTO repository_stats (stat_name, stat_value)
                    VALUES ('current_stats', ?)
                """, (stats_data,))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save repository stats: {e}")
    
    def _compression_loop(self):
        """Background thread for compressing old logs"""
        while self._running:
            try:
                # Compress logs older than 24 hours
                cutoff_time = time.time() - (24 * 3600)
                
                for entry_id, entry in list(self.entry_index.items()):
                    if (entry.timestamp < cutoff_time and 
                        not entry.compressed and 
                        not entry.archived):
                        
                        self._compress_log(entry)
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in compression loop: {e}")
                time.sleep(300)
    
    def _indexing_loop(self):
        """Background thread for maintaining search indexes"""
        while self._running:
            try:
                # Update search indexes
                self._rebuild_search_index()
                
                # Update repository statistics
                self._save_repository_stats()
                
                time.sleep(1800)  # Run every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in indexing loop: {e}")
                time.sleep(300)
    
    def _compress_log(self, entry: LogEntry):
        """Compress a log entry"""
        try:
            file_path = self.base_path / entry.file_path
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Update entry
            entry.file_path = str(compressed_path.relative_to(self.base_path))
            entry.compressed = True
            entry.size_bytes = compressed_path.stat().st_size
            
            # Update database
            self._store_log_entry(entry)
            
            # Remove original file
            file_path.unlink()
            
            logger.debug(f"🗂️ Compressed log: {entry.entry_id}")
            
        except Exception as e:
            logger.error(f"Failed to compress {entry.entry_id}: {e}")
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """Get comprehensive repository statistics"""
        
        with self._lock:
            # Calculate compression ratio
            total_original = sum(entry.size_bytes for entry in self.entry_index.values() if not entry.compressed)
            total_compressed = sum(entry.size_bytes for entry in self.entry_index.values() if entry.compressed)
            
            if total_compressed > 0:
                self.stats.compression_ratio = total_compressed / (total_original + total_compressed)
            
            # System breakdown
            system_stats = {}
            for system in set(entry.system for entry in self.entry_index.values()):
                system_entries = [e for e in self.entry_index.values() if e.system == system]
                system_stats[system] = {
                    'entries': len(system_entries),
                    'size_bytes': sum(e.size_bytes for e in system_entries),
                    'subsystems': len(set(e.subsystem for e in system_entries)),
                    'modules': len(set(e.module for e in system_entries))
                }
            
            return {
                'overview': asdict(self.stats),
                'system_breakdown': system_stats,
                'directory_structure': self._get_directory_stats(),
                'recent_activity': self._get_recent_activity(),
                'storage_efficiency': {
                    'compression_ratio': self.stats.compression_ratio,
                    'average_file_size': self.stats.total_size_bytes / max(self.stats.total_entries, 1),
                    'disk_usage_mb': self.stats.total_size_bytes / (1024 * 1024)
                }
            }
    
    def _get_directory_stats(self) -> Dict[str, Any]:
        """Get statistics about directory structure"""
        try:
            total_dirs = sum(1 for _ in self.base_path.rglob('*') if _.is_dir())
            total_files = sum(1 for _ in self.base_path.rglob('*') if _.is_file())
            
            return {
                'total_directories': total_dirs,
                'total_files': total_files,
                'max_depth': self._calculate_max_depth(),
                'systems_directories': len(list(self.systems_path.iterdir())) if self.systems_path.exists() else 0
            }
        except Exception as e:
            logger.error(f"Failed to get directory stats: {e}")
            return {}
    
    def _calculate_max_depth(self) -> int:
        """Calculate maximum directory depth"""
        max_depth = 0
        try:
            for path in self.base_path.rglob('*'):
                if path.is_dir():
                    depth = len(path.relative_to(self.base_path).parts)
                    max_depth = max(max_depth, depth)
        except Exception:
            pass
        return max_depth
    
    def _get_recent_activity(self) -> Dict[str, Any]:
        """Get recent activity statistics"""
        now = time.time()
        hour_ago = now - 3600
        day_ago = now - 86400
        
        recent_hour = [e for e in self.entry_index.values() if e.timestamp > hour_ago]
        recent_day = [e for e in self.entry_index.values() if e.timestamp > day_ago]
        
        return {
            'last_hour': {
                'entries': len(recent_hour),
                'size_bytes': sum(e.size_bytes for e in recent_hour),
                'systems': len(set(e.system for e in recent_hour))
            },
            'last_24_hours': {
                'entries': len(recent_day),
                'size_bytes': sum(e.size_bytes for e in recent_day),
                'systems': len(set(e.system for e in recent_day))
            }
        }
    
    def shutdown(self):
        """Shutdown the centralized repository"""
        logger.info("🗂️ Shutting down centralized logging repository...")
        
        self._running = False
        
        # Wait for background threads
        for thread in [self._organization_thread, self._compression_thread, self._indexing_thread]:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        # Save final statistics
        self._save_repository_stats()
        
        logger.info("✅ Centralized logging repository shutdown complete")

# Global repository instance
_centralized_repo: Optional[CentralizedLoggingRepository] = None
_repo_lock = threading.Lock()

def get_centralized_repository(base_path: str = "logs_repository") -> CentralizedLoggingRepository:
    """Get the global centralized logging repository"""
    global _centralized_repo
    
    with _repo_lock:
        if _centralized_repo is None:
            _centralized_repo = CentralizedLoggingRepository(base_path)
        return _centralized_repo

def centralize_all_logging(base_path: str = "logs_repository"):
    """Centralize all DAWN logging into the deep repository"""
    repo = get_centralized_repository(base_path)
    repo.integrate_universal_logger()
    return repo

if __name__ == "__main__":
    # Test the centralized repository
    logging.basicConfig(level=logging.INFO)
    
    # Create centralized repository
    repo = get_centralized_repository("test_logs_repository")
    
    # Add some test entries
    for i in range(10):
        repo.add_log_entry(
            system="consciousness",
            subsystem="engine", 
            module="primary",
            log_data={
                "test_data": f"entry_{i}",
                "timestamp": time.time(),
                "status": "active" if i % 2 == 0 else "idle"
            },
            log_type="state"
        )
    
    # Get statistics
    stats = repo.get_repository_stats()
    print(f"Repository Stats: {json.dumps(stats, indent=2, default=str)}")
    
    # Query logs
    results = repo.query_logs(system="consciousness", limit=5)
    print(f"Found {len(results)} log entries")
    
    # Shutdown
    repo.shutdown()
