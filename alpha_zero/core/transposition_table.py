import threading
from typing import Any, Tuple, Optional, Dict, List
from enum import Enum
import numpy as np
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

class NodeType(Enum):
    """Enumeration for the type of node in the transposition table."""
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2

class ConcurrentTranspositionTable:
    """
    Lock-free transposition table using striped locking and concurrent data structures.
    Uses multiple sub-tables with separate locks to reduce contention.
    """
    def __init__(self, num_shards: int = 256, shard_size: int = 4096):
        """
        Initialize sharded transposition table.
        
        Args:
            num_shards: Number of separate sub-tables (power of 2 recommended)
            shard_size: Maximum size of each shard
        """
        self.num_shards = num_shards
        self.shard_size = shard_size
        
        # Create sharded tables and locks
        self.shards: List[Dict] = [{} for _ in range(num_shards)]
        self.shard_locks = [threading.RLock() for _ in range(num_shards)]
        
        # LRU tracking per shard
        self.lru_lists = [deque(maxlen=shard_size) for _ in range(num_shards)]
        
        # Statistics tracking
        self.stats_lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.collisions = 0
    
    def _get_shard_index(self, zobrist_hash: Any) -> int:
        """Get shard index from hash value."""
        # Use lower bits for shard index
        return hash(zobrist_hash) & (self.num_shards - 1)
    
    def store(
        self,
        zobrist_hash: Any,
        depth: int,
        value: float,
        flag: NodeType,
        replace_existing: bool = True
    ) -> bool:
        """
        Store entry in appropriate shard. Thread-safe but lock-free for different shards.
        
        Args:
            zobrist_hash: Position hash
            depth: Search depth
            value: Evaluation value
            flag: Node type flag
            replace_existing: Whether to replace existing entries
            
        Returns:
            True if stored successfully, False if skipped
        """
        shard_idx = self._get_shard_index(zobrist_hash)
        
        with self.shard_locks[shard_idx]:
            shard = self.shards[shard_idx]
            lru = self.lru_lists[shard_idx]
            
            # Check if entry exists
            existing = shard.get(zobrist_hash)
            if existing and not replace_existing:
                if existing[0] >= depth:  # Existing entry is deeper
                    return False
                with self.stats_lock:
                    self.collisions += 1
            
            # Add new entry
            shard[zobrist_hash] = (depth, value, flag)
            
            # Update LRU
            if zobrist_hash in lru:
                lru.remove(zobrist_hash)
            lru.append(zobrist_hash)
            
            # Evict oldest if needed
            while len(shard) > self.shard_size:
                oldest = lru.popleft()
                shard.pop(oldest, None)
                
        return True
    
    def lookup(self, zobrist_hash: Any) -> Optional[Tuple[int, float, NodeType]]:
        """
        Thread-safe lookup that only locks the relevant shard.
        
        Args:
            zobrist_hash: Position hash
            
        Returns:
            Tuple of (depth, value, flag) if found, None otherwise
        """
        shard_idx = self._get_shard_index(zobrist_hash)
        
        with self.shard_locks[shard_idx]:
            result = self.shards[shard_idx].get(zobrist_hash)
            
            # Update LRU on hit
            if result:
                lru = self.lru_lists[shard_idx]
                if zobrist_hash in lru:
                    lru.remove(zobrist_hash)
                lru.append(zobrist_hash)
                
                with self.stats_lock:
                    self.hits += 1
            else:
                with self.stats_lock:
                    self.misses += 1
                    
        return result
    
    def prefetch(self, zobrist_hash: Any) -> None:
        """
        Prefetch entry into cache without blocking.
        Useful for speculative loading.
        """
        shard_idx = self._get_shard_index(zobrist_hash)
        # Just touch the shard to bring into cache
        self.shards[shard_idx].get(zobrist_hash, None)
    
    def get_stats(self) -> Dict[str, int]:
        """Get table statistics."""
        with self.stats_lock:
            return {
                'hits': self.hits,
                'misses': self.misses,
                'collisions': self.collisions
            }
    
    def clear_stats(self) -> None:
        """Reset statistics counters."""
        with self.stats_lock:
            self.hits = 0
            self.misses = 0
            self.collisions = 0
            
    def __len__(self) -> int:
        """Get total number of stored positions."""
        return sum(len(shard) for shard in self.shards)


class AsyncTranspositionTable:
    """
    Asynchronous transposition table that supports background operations.
    Combines sharding with async prefetch and background maintenance.
    """
    def __init__(
        self,
        num_shards: int = 256,
        shard_size: int = 4096,
        num_workers: int = 4
    ):
        """
        Initialize async transposition table.
        
        Args:
            num_shards: Number of sub-tables
            shard_size: Size of each shard
            num_workers: Number of background worker threads
        """
        self.table = ConcurrentTranspositionTable(num_shards, shard_size)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.prefetch_queue = deque(maxlen=1000)
        self.prefetch_lock = threading.Lock()
        
    def store(
        self,
        zobrist_hash: Any,
        depth: int,
        value: float,
        flag: NodeType,
        replace_existing: bool = True
    ) -> bool:
        """
        Synchronously store entry.
        
        Args:
            zobrist_hash: Position hash
            depth: Search depth
            value: Evaluation value
            flag: Node type flag
            replace_existing: Whether to replace existing entries
            
        Returns:
            True if stored successfully, False if skipped
        """
        return self.table.store(zobrist_hash, depth, value, flag, replace_existing)
        
    def store_async(
        self,
        zobrist_hash: Any,
        depth: int,
        value: float,
        flag: NodeType
    ) -> None:
        """
        Asynchronously store entry.
        
        Args:
            zobrist_hash: Position hash
            depth: Search depth
            value: Evaluation value
            flag: Node type flag
        """
        self.executor.submit(
            self.table.store,
            zobrist_hash,
            depth,
            value,
            flag
        )
        
    def lookup(self, zobrist_hash: Any) -> Optional[Tuple[int, float, NodeType]]:
        """
        Direct lookup without prefetching.
        
        Args:
            zobrist_hash: Position hash to look up
            
        Returns:
            Entry if found, None otherwise
        """
        return self.table.lookup(zobrist_hash)
        
    def lookup_with_prefetch(
        self,
        zobrist_hash: Any,
        prefetch_hashes: Optional[List[Any]] = None
    ) -> Optional[Tuple[int, float, NodeType]]:
        """
        Lookup entry and optionally prefetch related positions.
        
        Args:
            zobrist_hash: Position hash to look up
            prefetch_hashes: Related positions to prefetch
            
        Returns:
            Entry if found, None otherwise
        """
        # Queue prefetch requests
        if prefetch_hashes:
            with self.prefetch_lock:
                for h in prefetch_hashes:
                    if len(self.prefetch_queue) < self.prefetch_queue.maxlen:
                        self.prefetch_queue.append(h)
                        
        # Do actual lookup
        result = self.table.lookup(zobrist_hash)
        
        # Trigger async prefetch
        self._trigger_prefetch()
        
        return result
        
    def _trigger_prefetch(self) -> None:
        """Process some prefetch requests in background."""
        to_prefetch = []
        with self.prefetch_lock:
            while self.prefetch_queue and len(to_prefetch) < 10:
                to_prefetch.append(self.prefetch_queue.popleft())
                
        if to_prefetch:
            self.executor.submit(self._do_prefetch, to_prefetch)
            
    def _do_prefetch(self, hashes: List[Any]) -> None:
        """Actually perform prefetch operations."""
        for h in hashes:
            self.table.prefetch(h)
            
    def shutdown(self) -> None:
        """Clean shutdown of async operations."""
        self.executor.shutdown(wait=True)
        
    def __len__(self) -> int:
        return len(self.table)
        
    def get_stats(self) -> Dict[str, int]:
        return self.table.get_stats()
        
    def clear_stats(self) -> None:
        self.table.clear_stats()


# For backwards compatibility
TranspositionTable = AsyncTranspositionTable