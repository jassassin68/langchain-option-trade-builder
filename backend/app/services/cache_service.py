"""
Cache Service for managing Redis caching with TTL configuration.

Implements requirements 7.1, 7.4:
- Cache service with TTL configuration for different data types
- Cache key strategies for tickers, quotes, and analysis results
- Cache invalidation and refresh mechanisms
- Performance optimization
"""

import logging
import json
from typing import Optional, Any, Dict
from datetime import timedelta
import redis.asyncio as redis
from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Base exception for cache service errors"""
    pass


class CacheService:
    """
    Service for managing Redis cache with TTL configuration.
    
    Implements different TTL strategies for different data types:
    - Ticker data: 1 hour (relatively static)
    - Market quotes: 5 minutes (frequently changing)
    - Analysis results: 30 minutes (moderate refresh rate)
    """
    
    # Cache key prefixes
    PREFIX_TICKER = "ticker"
    PREFIX_QUOTE = "quote"
    PREFIX_TECHNICAL = "technical"
    PREFIX_FUNDAMENTAL = "fundamental"
    PREFIX_OPTIONS = "options"
    PREFIX_ANALYSIS = "analysis"
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize cache service with Redis connection.
        
        Args:
            redis_url: Redis connection URL (defaults to settings)
        """
        self.redis_url = redis_url or settings.redis_url
        self.redis_client: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self):
        """Establish connection to Redis"""
        try:
            if not self._connected:
                self.redis_client = await redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                # Test connection
                await self.redis_client.ping()
                self._connected = True
                logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            raise CacheError(f"Redis connection failed: {e}")
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self._connected = False
            logger.info("Disconnected from Redis")
    
    def _ensure_connected(self):
        """Ensure Redis connection is established"""
        if not self._connected or not self.redis_client:
            raise CacheError("Redis not connected. Call connect() first.")
    
    def _build_key(self, prefix: str, identifier: str, **kwargs) -> str:
        """
        Build cache key with prefix and identifier.
        
        Args:
            prefix: Key prefix (e.g., 'ticker', 'quote')
            identifier: Main identifier (e.g., ticker symbol)
            **kwargs: Additional key components
            
        Returns:
            Formatted cache key
        """
        key_parts = [prefix, identifier]
        
        # Add additional components if provided
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        
        return ":".join(key_parts)
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            self._ensure_connected()
            
            value = await self.redis_client.get(key)
            
            if value:
                logger.debug(f"Cache hit: {key}")
                return json.loads(value)
            
            logger.debug(f"Cache miss: {key}")
            return None
            
        except Exception as e:
            logger.warning(f"Error getting cache key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (optional)
        """
        try:
            self._ensure_connected()
            
            serialized_value = json.dumps(value)
            
            if ttl:
                await self.redis_client.setex(key, ttl, serialized_value)
            else:
                await self.redis_client.set(key, serialized_value)
            
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            
        except Exception as e:
            logger.warning(f"Error setting cache key {key}: {e}")
    
    async def delete(self, key: str):
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
        """
        try:
            self._ensure_connected()
            
            await self.redis_client.delete(key)
            logger.debug(f"Cache deleted: {key}")
            
        except Exception as e:
            logger.warning(f"Error deleting cache key {key}: {e}")
    
    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: Key pattern (e.g., 'ticker:*')
            
        Returns:
            Number of keys deleted
        """
        try:
            self._ensure_connected()
            
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await self.redis_client.delete(*keys)
                logger.info(f"Deleted {deleted} keys matching pattern: {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.warning(f"Error deleting keys with pattern {pattern}: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            self._ensure_connected()
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.warning(f"Error checking key existence {key}: {e}")
            return False
    
    async def get_ttl(self, key: str) -> int:
        """
        Get remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Remaining TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        try:
            self._ensure_connected()
            return await self.redis_client.ttl(key)
        except Exception as e:
            logger.warning(f"Error getting TTL for key {key}: {e}")
            return -2
    
    # Convenience methods for specific data types
    
    async def get_ticker(self, ticker: str) -> Optional[Dict]:
        """Get cached ticker data"""
        key = self._build_key(self.PREFIX_TICKER, ticker)
        return await self.get(key)
    
    async def set_ticker(self, ticker: str, data: Dict):
        """Cache ticker data with 1 hour TTL"""
        key = self._build_key(self.PREFIX_TICKER, ticker)
        await self.set(key, data, ttl=settings.ticker_cache_ttl)
    
    async def get_quote(self, ticker: str) -> Optional[Dict]:
        """Get cached quote data"""
        key = self._build_key(self.PREFIX_QUOTE, ticker)
        return await self.get(key)
    
    async def set_quote(self, ticker: str, data: Dict):
        """Cache quote data with 5 minutes TTL"""
        key = self._build_key(self.PREFIX_QUOTE, ticker)
        await self.set(key, data, ttl=settings.market_data_cache_ttl)
    
    async def get_technical(self, ticker: str) -> Optional[Dict]:
        """Get cached technical indicators"""
        key = self._build_key(self.PREFIX_TECHNICAL, ticker)
        return await self.get(key)
    
    async def set_technical(self, ticker: str, data: Dict):
        """Cache technical indicators with 5 minutes TTL"""
        key = self._build_key(self.PREFIX_TECHNICAL, ticker)
        await self.set(key, data, ttl=settings.market_data_cache_ttl)
    
    async def get_fundamental(self, ticker: str) -> Optional[Dict]:
        """Get cached fundamental data"""
        key = self._build_key(self.PREFIX_FUNDAMENTAL, ticker)
        return await self.get(key)
    
    async def set_fundamental(self, ticker: str, data: Dict):
        """Cache fundamental data with 1 hour TTL"""
        key = self._build_key(self.PREFIX_FUNDAMENTAL, ticker)
        await self.set(key, data, ttl=settings.ticker_cache_ttl)
    
    async def get_options(self, ticker: str) -> Optional[Dict]:
        """Get cached options chain data"""
        key = self._build_key(self.PREFIX_OPTIONS, ticker)
        return await self.get(key)
    
    async def set_options(self, ticker: str, data: Dict):
        """Cache options chain data with 5 minutes TTL"""
        key = self._build_key(self.PREFIX_OPTIONS, ticker)
        await self.set(key, data, ttl=settings.market_data_cache_ttl)
    
    async def get_analysis(self, ticker: str) -> Optional[Dict]:
        """Get cached analysis results"""
        key = self._build_key(self.PREFIX_ANALYSIS, ticker)
        return await self.get(key)
    
    async def set_analysis(self, ticker: str, data: Dict):
        """Cache analysis results with 30 minutes TTL"""
        key = self._build_key(self.PREFIX_ANALYSIS, ticker)
        await self.set(key, data, ttl=settings.analysis_cache_ttl)
    
    async def invalidate_ticker(self, ticker: str):
        """
        Invalidate all cached data for a ticker.
        
        This removes ticker, quote, technical, fundamental, options, and analysis data.
        
        Args:
            ticker: Ticker symbol to invalidate
        """
        patterns = [
            f"{self.PREFIX_TICKER}:{ticker}",
            f"{self.PREFIX_QUOTE}:{ticker}",
            f"{self.PREFIX_TECHNICAL}:{ticker}",
            f"{self.PREFIX_FUNDAMENTAL}:{ticker}",
            f"{self.PREFIX_OPTIONS}:{ticker}",
            f"{self.PREFIX_ANALYSIS}:{ticker}"
        ]
        
        for pattern in patterns:
            await self.delete(pattern)
        
        logger.info(f"Invalidated all cache for ticker: {ticker}")
    
    async def refresh_ticker(self, ticker: str, data_fetchers: Dict[str, Any]):
        """
        Refresh cached data for a ticker.
        
        Args:
            ticker: Ticker symbol
            data_fetchers: Dictionary of data type to fetcher function
                          e.g., {'quote': fetch_quote_func, 'technical': fetch_technical_func}
        """
        try:
            # Invalidate existing cache
            await self.invalidate_ticker(ticker)
            
            # Fetch and cache new data
            for data_type, fetcher in data_fetchers.items():
                try:
                    data = await fetcher(ticker)
                    
                    if data_type == 'quote':
                        await self.set_quote(ticker, data)
                    elif data_type == 'technical':
                        await self.set_technical(ticker, data)
                    elif data_type == 'fundamental':
                        await self.set_fundamental(ticker, data)
                    elif data_type == 'options':
                        await self.set_options(ticker, data)
                    
                except Exception as e:
                    logger.warning(f"Error refreshing {data_type} for {ticker}: {e}")
            
            logger.info(f"Refreshed cache for ticker: {ticker}")
            
        except Exception as e:
            logger.error(f"Error refreshing ticker cache: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            self._ensure_connected()
            
            info = await self.redis_client.info()
            
            return {
                'connected': self._connected,
                'used_memory': info.get('used_memory_human', 'N/A'),
                'total_keys': await self.redis_client.dbsize(),
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0),
                'hit_rate': self._calculate_hit_rate(
                    info.get('keyspace_hits', 0),
                    info.get('keyspace_misses', 0)
                )
            }
            
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {'connected': False, 'error': str(e)}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage"""
        total = hits + misses
        if total == 0:
            return 0.0
        return round((hits / total) * 100, 2)


# Global cache service instance
_cache_service: Optional[CacheService] = None


async def get_cache_service() -> CacheService:
    """
    Get or create global cache service instance.
    
    Returns:
        CacheService instance
    """
    global _cache_service
    
    if _cache_service is None:
        _cache_service = CacheService()
        await _cache_service.connect()
    
    return _cache_service


async def close_cache_service():
    """Close global cache service instance"""
    global _cache_service
    
    if _cache_service:
        await _cache_service.disconnect()
        _cache_service = None
