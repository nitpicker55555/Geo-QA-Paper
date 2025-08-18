"""
Optimization utilities for the Flask application.

This module contains utility functions and classes that help optimize
performance, memory usage, and code organization.
"""

import functools
import hashlib
import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Union, Callable
import threading
from collections import OrderedDict


class LRUCache:
    """
    Thread-safe LRU Cache implementation with size and time-based eviction.
    """
    
    def __init__(self, max_size: int = 128, ttl: Optional[float] = None):
        """
        Initialize LRU Cache.
        
        Args:
            max_size: Maximum number of items to cache
            ttl: Time to live in seconds (None for no expiration)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict = {}
        self.lock = threading.RLock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache with LRU ordering."""
        with self.lock:
            if key not in self.cache:
                return default
            
            # Check TTL
            if self.ttl and self._is_expired(key):
                self._remove_key(key)
                return default
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache, evicting if necessary."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                self.cache[key] = value
                
                # Evict oldest if over capacity
                while len(self.cache) > self.max_size:
                    oldest_key = next(iter(self.cache))
                    self._remove_key(oldest_key)
            
            # Update timestamp
            if self.ttl:
                self.timestamps[key] = time.time()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if not self.ttl or key not in self.timestamps:
            return False
        return time.time() - self.timestamps[key] > self.ttl
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache and timestamps."""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)


class FileCache:
    """
    File-based caching system with automatic invalidation.
    """
    
    def __init__(self, cache_dir: str = '.cache', default_ttl: float = 3600):
        """
        Initialize file cache.
        
        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time to live in seconds
        """
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        os.makedirs(cache_dir, exist_ok=True)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from file cache."""
        cache_file = self._get_cache_file(key)
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            # Check if file is expired
            if self._is_file_expired(cache_file):
                os.remove(cache_file)
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('value')
        
        except (json.JSONDecodeError, OSError):
            # Remove corrupted cache file
            try:
                os.remove(cache_file)
            except OSError:
                pass
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put item in file cache."""
        if ttl is None:
            ttl = self.default_ttl
        
        cache_file = self._get_cache_file(key)
        cache_data = {
            'value': value,
            'timestamp': time.time(),
            'ttl': ttl
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, default=str)
        except (OSError, TypeError) as e:
            print(f"Error writing to cache: {e}")
    
    def _get_cache_file(self, key: str) -> str:
        """Get cache file path for key."""
        # Create hash of key for filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.json")
    
    def _is_file_expired(self, cache_file: str) -> bool:
        """Check if cache file is expired."""
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                timestamp = data.get('timestamp', 0)
                ttl = data.get('ttl', self.default_ttl)
                return time.time() - timestamp > ttl
        except (json.JSONDecodeError, OSError):
            return True
    
    def clear(self) -> None:
        """Clear all cached files."""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
        except OSError as e:
            print(f"Error clearing cache: {e}")


def memoize(max_size: int = 128, ttl: Optional[float] = None):
    """
    Decorator for memoizing function results with LRU cache.
    
    Args:
        max_size: Maximum cache size
        ttl: Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(max_size, ttl)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            key = _create_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        
        # Add cache management methods
        wrapper.cache_info = lambda: {'size': cache.size(), 'max_size': max_size}
        wrapper.cache_clear = cache.clear
        
        return wrapper
    return decorator


def file_cache(cache_dir: str = '.cache', ttl: float = 3600):
    """
    Decorator for file-based caching of function results.
    
    Args:
        cache_dir: Directory for cache files
        ttl: Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        cache = FileCache(cache_dir, ttl)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            key = _create_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.put(key, result, ttl)
            return result
        
        # Add cache management methods
        wrapper.cache_clear = cache.clear
        
        return wrapper
    return decorator


def _create_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Create a unique cache key from function name and arguments."""
    # Convert arguments to hashable representation
    key_parts = [func_name]
    
    # Add positional arguments
    for arg in args:
        if isinstance(arg, (dict, list, set)):
            key_parts.append(json.dumps(arg, sort_keys=True, default=str))
        else:
            key_parts.append(str(arg))
    
    # Add keyword arguments
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (dict, list, set)):
            key_parts.append(f"{k}:{json.dumps(v, sort_keys=True, default=str)}")
        else:
            key_parts.append(f"{k}:{str(v)}")
    
    # Create hash of the key
    key_string = '|'.join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


@contextmanager
def timing_context(operation_name: str) -> Iterator[Dict[str, float]]:
    """
    Context manager for timing operations.
    
    Args:
        operation_name: Name of the operation being timed
        
    Yields:
        Dictionary with timing information
    """
    start_time = time.time()
    timing_info = {'operation': operation_name, 'start_time': start_time}
    
    try:
        yield timing_info
    finally:
        end_time = time.time()
        timing_info.update({
            'end_time': end_time,
            'duration': end_time - start_time
        })
        print(f"Operation '{operation_name}' took {timing_info['duration']:.4f} seconds")


class PerformanceMonitor:
    """
    Performance monitoring utility for tracking application metrics.
    """
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()
    
    def record_timing(self, operation: str, duration: float) -> None:
        """Record timing metric."""
        with self.lock:
            if operation not in self.metrics:
                self.metrics[operation] = {
                    'count': 0,
                    'total_time': 0,
                    'min_time': float('inf'),
                    'max_time': 0,
                    'avg_time': 0
                }
            
            metric = self.metrics[operation]
            metric['count'] += 1
            metric['total_time'] += duration
            metric['min_time'] = min(metric['min_time'], duration)
            metric['max_time'] = max(metric['max_time'], duration)
            metric['avg_time'] = metric['total_time'] / metric['count']
    
    def get_metrics(self, operation: str = None) -> Union[Dict, Dict[str, Dict]]:
        """Get performance metrics."""
        with self.lock:
            if operation:
                return self.metrics.get(operation, {})
            return self.metrics.copy()
    
    def reset_metrics(self, operation: str = None) -> None:
        """Reset performance metrics."""
        with self.lock:
            if operation:
                self.metrics.pop(operation, None)
            else:
                self.metrics.clear()


class MemoryOptimizer:
    """
    Memory optimization utilities for managing large data structures.
    """
    
    @staticmethod
    def chunk_list(data: list, chunk_size: int = 1000) -> Iterator[list]:
        """
        Split large list into smaller chunks for memory efficiency.
        
        Args:
            data: List to chunk
            chunk_size: Size of each chunk
            
        Yields:
            Chunks of the original list
        """
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
    
    @staticmethod
    def lazy_load_json(file_path: str, chunk_key: str = None) -> Iterator[Any]:
        """
        Lazily load JSON data to reduce memory usage.
        
        Args:
            file_path: Path to JSON file
            chunk_key: Key to iterate over for chunked loading
            
        Yields:
            JSON objects or chunks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if chunk_key and isinstance(data, dict) and chunk_key in data:
                    # Yield items from specified key
                    for item in data[chunk_key]:
                        yield item
                else:
                    # Yield entire data structure
                    yield data
        
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error loading JSON file {file_path}: {e}")
    
    @staticmethod
    def memory_efficient_sort(data: list, key_func: Callable = None, 
                            chunk_size: int = 10000) -> list:
        """
        Memory-efficient sorting for large lists.
        
        Args:
            data: List to sort
            key_func: Key function for sorting
            chunk_size: Size of chunks for external sorting
            
        Returns:
            Sorted list
        """
        if len(data) <= chunk_size:
            return sorted(data, key=key_func)
        
        # External merge sort for large data
        chunks = list(MemoryOptimizer.chunk_list(data, chunk_size))
        sorted_chunks = [sorted(chunk, key=key_func) for chunk in chunks]
        
        # Merge sorted chunks
        return MemoryOptimizer._merge_sorted_chunks(sorted_chunks, key_func)
    
    @staticmethod
    def _merge_sorted_chunks(chunks: list, key_func: Callable = None) -> list:
        """Merge multiple sorted chunks into a single sorted list."""
        import heapq
        
        # Create heap entries: (value, chunk_index, item_index)
        heap = []
        chunk_indices = [0] * len(chunks)
        
        # Initialize heap with first item from each chunk
        for chunk_idx, chunk in enumerate(chunks):
            if chunk:
                item = chunk[0]
                heap_item = (key_func(item) if key_func else item, chunk_idx, 0, item)
                heapq.heappush(heap, heap_item)
        
        result = []
        
        while heap:
            key_val, chunk_idx, item_idx, item = heapq.heappop(heap)
            result.append(item)
            
            # Add next item from the same chunk
            next_idx = item_idx + 1
            if next_idx < len(chunks[chunk_idx]):
                next_item = chunks[chunk_idx][next_idx]
                next_heap_item = (
                    key_func(next_item) if key_func else next_item,
                    chunk_idx,
                    next_idx,
                    next_item
                )
                heapq.heappush(heap, next_heap_item)
        
        return result


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile function execution time.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with profiling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            performance_monitor.record_timing(func.__name__, duration)
    
    return wrapper


class ResourceManager:
    """
    Resource manager for handling file operations and cleanup.
    """
    
    def __init__(self):
        self.open_files = []
        self.temp_files = []
    
    @contextmanager
    def managed_file(self, file_path: str, mode: str = 'r', 
                    encoding: str = 'utf-8') -> Iterator[Any]:
        """
        Context manager for safe file handling.
        
        Args:
            file_path: Path to file
            mode: File open mode
            encoding: File encoding
            
        Yields:
            File handle
        """
        file_handle = None
        try:
            file_handle = open(file_path, mode, encoding=encoding)
            self.open_files.append(file_handle)
            yield file_handle
        finally:
            if file_handle:
                try:
                    file_handle.close()
                    self.open_files.remove(file_handle)
                except (OSError, ValueError):
                    pass
    
    def cleanup(self) -> None:
        """Clean up all managed resources."""
        # Close open files
        for file_handle in self.open_files[:]:
            try:
                file_handle.close()
            except OSError:
                pass
        self.open_files.clear()
        
        # Remove temporary files
        for temp_file in self.temp_files[:]:
            try:
                os.remove(temp_file)
            except OSError:
                pass
        self.temp_files.clear()


# Global resource manager instance
resource_manager = ResourceManager()