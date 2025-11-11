import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
import json

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.cache_service import CacheService, CacheError


class TestCacheService:
    """Test CacheService functionality"""
    
    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client"""
        mock = AsyncMock()
        mock.ping = AsyncMock()
        mock.get = AsyncMock()
        mock.set = AsyncMock()
        mock.setex = AsyncMock()
        mock.delete = AsyncMock()
        mock.exists = AsyncMock()
        mock.ttl = AsyncMock()
        mock.dbsize = AsyncMock()
        mock.info = AsyncMock()
        mock.scan_iter = AsyncMock()
        mock.close = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_from_url(self, mock_redis):
        """Create async mock for redis.from_url"""
        async def _mock_from_url(*args, **kwargs):
            return mock_redis
        return _mock_from_url
    
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_redis, mock_from_url):
        """Test successful Redis connection"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            assert service._connected is True
            mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test Redis connection failure"""
        async def failing_from_url(*args, **kwargs):
            raise Exception("Connection failed")
        
        with patch('redis.asyncio.from_url', side_effect=failing_from_url):
            service = CacheService()
            
            with pytest.raises(CacheError, match="Redis connection failed"):
                await service.connect()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, mock_redis, mock_from_url):
        """Test Redis disconnection"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            await service.disconnect()
            
            assert service._connected is False
            mock_redis.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_build_key(self, mock_redis, mock_from_url):
        """Test cache key building"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            key = service._build_key("ticker", "AAPL")
            assert key == "ticker:AAPL"
            
            key_with_params = service._build_key("quote", "AAPL", date="2024-01-01")
            assert key_with_params == "quote:AAPL:date:2024-01-01"
    
    @pytest.mark.asyncio
    async def test_get_cache_hit(self, mock_redis, mock_from_url):
        """Test getting value from cache (hit)"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            test_data = {"ticker": "AAPL", "price": 150.0}
            mock_redis.get.return_value = json.dumps(test_data)
            
            result = await service.get("test:key")
            
            assert result == test_data
            mock_redis.get.assert_called_once_with("test:key")
    
    @pytest.mark.asyncio
    async def test_get_cache_miss(self, mock_redis, mock_from_url):
        """Test getting value from cache (miss)"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            mock_redis.get.return_value = None
            
            result = await service.get("test:key")
            
            assert result is None
            mock_redis.get.assert_called_once_with("test:key")
    
    @pytest.mark.asyncio
    async def test_set_with_ttl(self, mock_redis, mock_from_url):
        """Test setting value in cache with TTL"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            test_data = {"ticker": "AAPL", "price": 150.0}
            
            await service.set("test:key", test_data, ttl=300)
            
            mock_redis.setex.assert_called_once()
            call_args = mock_redis.setex.call_args
            assert call_args[0][0] == "test:key"
            assert call_args[0][1] == 300
            assert json.loads(call_args[0][2]) == test_data
    
    @pytest.mark.asyncio
    async def test_set_without_ttl(self, mock_redis, mock_from_url):
        """Test setting value in cache without TTL"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            test_data = {"ticker": "AAPL", "price": 150.0}
            
            await service.set("test:key", test_data)
            
            mock_redis.set.assert_called_once()
            call_args = mock_redis.set.call_args
            assert call_args[0][0] == "test:key"
            assert json.loads(call_args[0][1]) == test_data
    
    @pytest.mark.asyncio
    async def test_delete(self, mock_redis, mock_from_url):
        """Test deleting key from cache"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            await service.delete("test:key")
            
            mock_redis.delete.assert_called_once_with("test:key")
    
    @pytest.mark.asyncio
    async def test_delete_pattern(self, mock_redis, mock_from_url):
        """Test deleting keys by pattern"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            # Mock scan_iter to return an async generator
            async def mock_scan():
                for key in ["ticker:AAPL", "ticker:MSFT", "ticker:GOOGL"]:
                    yield key
            
            # Make scan_iter return the async generator directly
            mock_redis.scan_iter = MagicMock(return_value=mock_scan())
            mock_redis.delete.return_value = 3
            
            deleted = await service.delete_pattern("ticker:*")
            
            assert deleted == 3
            mock_redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_exists(self, mock_redis, mock_from_url):
        """Test checking if key exists"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            mock_redis.exists.return_value = 1
            
            exists = await service.exists("test:key")
            
            assert exists is True
            mock_redis.exists.assert_called_once_with("test:key")
    
    @pytest.mark.asyncio
    async def test_get_ttl(self, mock_redis, mock_from_url):
        """Test getting TTL for a key"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            mock_redis.ttl.return_value = 300
            
            ttl = await service.get_ttl("test:key")
            
            assert ttl == 300
            mock_redis.ttl.assert_called_once_with("test:key")
    
    @pytest.mark.asyncio
    async def test_get_ticker(self, mock_redis, mock_from_url):
        """Test getting cached ticker data"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            test_data = {"ticker": "AAPL", "company": "Apple Inc."}
            mock_redis.get.return_value = json.dumps(test_data)
            
            result = await service.get_ticker("AAPL")
            
            assert result == test_data
    
    @pytest.mark.asyncio
    async def test_set_ticker(self, mock_redis, mock_from_url):
        """Test caching ticker data"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            test_data = {"ticker": "AAPL", "company": "Apple Inc."}
            
            await service.set_ticker("AAPL", test_data)
            
            mock_redis.setex.assert_called_once()
            # Verify TTL is set to ticker_cache_ttl
            assert mock_redis.setex.call_args[0][1] == 3600  # Default ticker_cache_ttl
    
    @pytest.mark.asyncio
    async def test_get_quote(self, mock_redis, mock_from_url):
        """Test getting cached quote data"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            test_data = {"ticker": "AAPL", "price": 150.0}
            mock_redis.get.return_value = json.dumps(test_data)
            
            result = await service.get_quote("AAPL")
            
            assert result == test_data
    
    @pytest.mark.asyncio
    async def test_set_quote(self, mock_redis, mock_from_url):
        """Test caching quote data"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            test_data = {"ticker": "AAPL", "price": 150.0}
            
            await service.set_quote("AAPL", test_data)
            
            mock_redis.setex.assert_called_once()
            # Verify TTL is set to market_data_cache_ttl
            assert mock_redis.setex.call_args[0][1] == 300  # Default market_data_cache_ttl
    
    @pytest.mark.asyncio
    async def test_get_analysis(self, mock_redis, mock_from_url):
        """Test getting cached analysis results"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            test_data = {"ticker": "AAPL", "recommendation": "BUY"}
            mock_redis.get.return_value = json.dumps(test_data)
            
            result = await service.get_analysis("AAPL")
            
            assert result == test_data
    
    @pytest.mark.asyncio
    async def test_set_analysis(self, mock_redis, mock_from_url):
        """Test caching analysis results"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            test_data = {"ticker": "AAPL", "recommendation": "BUY"}
            
            await service.set_analysis("AAPL", test_data)
            
            mock_redis.setex.assert_called_once()
            # Verify TTL is set to analysis_cache_ttl
            assert mock_redis.setex.call_args[0][1] == 1800  # Default analysis_cache_ttl
    
    @pytest.mark.asyncio
    async def test_invalidate_ticker(self, mock_redis, mock_from_url):
        """Test invalidating all cache for a ticker"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            await service.invalidate_ticker("AAPL")
            
            # Should delete 6 different cache keys
            assert mock_redis.delete.call_count == 6
    
    @pytest.mark.asyncio
    async def test_get_cache_stats(self, mock_redis, mock_from_url):
        """Test getting cache statistics"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            mock_redis.info.return_value = {
                'used_memory_human': '1.5M',
                'keyspace_hits': 100,
                'keyspace_misses': 20
            }
            mock_redis.dbsize.return_value = 50
            
            stats = await service.get_cache_stats()
            
            assert stats['connected'] is True
            assert stats['used_memory'] == '1.5M'
            assert stats['total_keys'] == 50
            assert stats['hits'] == 100
            assert stats['misses'] == 20
            assert stats['hit_rate'] == 83.33  # 100/(100+20) * 100
    
    @pytest.mark.asyncio
    async def test_calculate_hit_rate(self, mock_redis, mock_from_url):
        """Test hit rate calculation"""
        with patch('redis.asyncio.from_url', side_effect=mock_from_url):
            service = CacheService()
            await service.connect()
            
            assert service._calculate_hit_rate(100, 20) == 83.33
            assert service._calculate_hit_rate(0, 0) == 0.0
            assert service._calculate_hit_rate(50, 50) == 50.0
    
    @pytest.mark.asyncio
    async def test_ensure_connected_raises_error(self):
        """Test that operations fail when not connected"""
        service = CacheService()
        
        with pytest.raises(CacheError, match="Redis not connected"):
            service._ensure_connected()
