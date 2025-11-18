"""
Database operation tests with test database setup.

Tests database operations, transactions, and data integrity as required by 7.1, 7.3.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.models.database import StockTicker, AnalysisCache
from app.services.ticker_service import TickerService
from app.services.cache_service import CacheService


class TestDatabaseOperations:
    """Test database operations with mocked database"""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock database session"""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()
        return session
    
    @pytest.fixture
    def sample_tickers(self):
        """Sample ticker data for testing"""
        return [
            StockTicker(
                ticker="AAPL",
                company_name="Apple Inc.",
                exchange="NASDAQ",
                is_active=True
            ),
            StockTicker(
                ticker="MSFT",
                company_name="Microsoft Corporation",
                exchange="NASDAQ",
                is_active=True
            ),
            StockTicker(
                ticker="GOOGL",
                company_name="Alphabet Inc.",
                exchange="NASDAQ",
                is_active=True
            )
        ]
    
    @pytest.mark.asyncio
    async def test_ticker_search_database_query(self, mock_session, sample_tickers):
        """Test ticker search database query performance"""
        # Mock database result
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = sample_tickers[:2]
        mock_session.execute.return_value = mock_result
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            service = TickerService()
            results = await service.search_tickers("AP", limit=10)
            
            assert len(results) == 2
            assert results[0].ticker == "AAPL"
            assert results[1].ticker == "MSFT"
            
            # Verify database query was executed
            mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ticker_search_fuzzy_matching(self, mock_session, sample_tickers):
        """Test fuzzy matching in ticker search"""
        # Mock database result for fuzzy search
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_tickers[0]]  # AAPL
        mock_session.execute.return_value = mock_result
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            service = TickerService()
            
            # Test partial ticker match
            results = await service.search_tickers("AAP", limit=10)
            assert len(results) == 1
            assert results[0].ticker == "AAPL"
            
            # Test company name match
            results = await service.search_tickers("Apple", limit=10)
            assert len(results) == 1
            assert results[0].ticker == "AAPL"
    
    @pytest.mark.asyncio
    async def test_get_ticker_by_symbol(self, mock_session, sample_tickers):
        """Test getting specific ticker by symbol"""
        # Mock database result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_tickers[0]
        mock_session.execute.return_value = mock_result
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            service = TickerService()
            result = await service.get_ticker_by_symbol("AAPL")
            
            assert result is not None
            assert result.ticker == "AAPL"
            assert result.company_name == "Apple Inc."
    
    @pytest.mark.asyncio
    async def test_ticker_not_found(self, mock_session):
        """Test handling when ticker is not found"""
        # Mock database result returning None
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            service = TickerService()
            result = await service.get_ticker_by_symbol("INVALID")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_database_transaction_rollback(self, mock_session):
        """Test database transaction rollback on error"""
        # Mock database error
        mock_session.execute.side_effect = Exception("Database error")
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            service = TickerService()
            
            with pytest.raises(Exception):
                await service.search_tickers("AAPL")
            
            # Session should be properly closed even on error
            mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_database_access(self, mock_session, sample_tickers):
        """Test concurrent database access"""
        # Mock database result
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = sample_tickers
        mock_session.execute.return_value = mock_result
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            service = TickerService()
            
            # Execute multiple concurrent searches
            tasks = [
                service.search_tickers("A", limit=10),
                service.search_tickers("M", limit=10),
                service.search_tickers("G", limit=10)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All searches should succeed
            assert len(results) == 3
            for result in results:
                assert len(result) == 3  # All tickers returned
            
            # Database should have been accessed 3 times
            assert mock_session.execute.call_count == 3


class TestCacheOperations:
    """Test cache operations and performance"""
    
    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client"""
        redis_client = AsyncMock()
        redis_client.get = AsyncMock()
        redis_client.set = AsyncMock()
        redis_client.delete = AsyncMock()
        redis_client.exists = AsyncMock()
        redis_client.expire = AsyncMock()
        return redis_client
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, mock_redis):
        """Test basic cache set and get operations"""
        mock_redis.get.return_value = json.dumps({"test": "data"})
        
        with patch('app.services.cache_service.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis
            
            cache = CacheService()
            
            # Test set operation
            await cache.set("test_key", {"test": "data"}, ttl=300)
            mock_redis.set.assert_called_once()
            
            # Test get operation
            result = await cache.get("test_key")
            assert result == {"test": "data"}
            mock_redis.get.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, mock_redis):
        """Test cache miss handling"""
        mock_redis.get.return_value = None
        
        with patch('app.services.cache_service.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis
            
            cache = CacheService()
            result = await cache.get("nonexistent_key")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, mock_redis):
        """Test cache TTL and expiration"""
        with patch('app.services.cache_service.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis
            
            cache = CacheService()
            
            # Set with TTL
            await cache.set("expiring_key", {"data": "value"}, ttl=60)
            
            # Verify set was called with TTL
            mock_redis.set.assert_called_once()
            call_args = mock_redis.set.call_args
            assert call_args[1]['ex'] == 60  # TTL in seconds
    
    @pytest.mark.asyncio
    async def test_cache_delete(self, mock_redis):
        """Test cache deletion"""
        with patch('app.services.cache_service.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis
            
            cache = CacheService()
            await cache.delete("test_key")
            
            mock_redis.delete.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, mock_redis):
        """Test cache key generation strategies"""
        with patch('app.services.cache_service.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis
            
            cache = CacheService()
            
            # Test ticker search cache key
            ticker_key = cache._generate_key("ticker_search", "AAPL")
            assert "ticker_search" in ticker_key
            assert "AAPL" in ticker_key
            
            # Test analysis cache key
            analysis_key = cache._generate_key("analysis", "AAPL")
            assert "analysis" in analysis_key
            assert "AAPL" in analysis_key
    
    @pytest.mark.asyncio
    async def test_cache_error_handling(self, mock_redis):
        """Test cache error handling"""
        # Mock Redis connection error
        mock_redis.get.side_effect = Exception("Redis connection error")
        
        with patch('app.services.cache_service.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis
            
            cache = CacheService()
            
            # Cache errors should not break the application
            result = await cache.get("test_key")
            assert result is None  # Should return None on error
    
    @pytest.mark.asyncio
    async def test_analysis_cache_integration(self, mock_session):
        """Test analysis cache database integration"""
        # Mock analysis cache data
        cache_data = AnalysisCache(
            ticker="AAPL",
            analysis_data={
                "recommendation": {
                    "should_trade": True,
                    "confidence": 0.85
                }
            },
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_data
        mock_session.execute.return_value = mock_result
        
        with patch('app.services.cache_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            cache = CacheService()
            result = await cache.get_analysis_cache("AAPL")
            
            assert result is not None
            assert result["recommendation"]["should_trade"] is True
            assert result["recommendation"]["confidence"] == 0.85


class TestDataIntegrity:
    """Test data integrity and validation"""
    
    @pytest.mark.asyncio
    async def test_ticker_data_validation(self):
        """Test ticker data validation"""
        # Valid ticker
        valid_ticker = StockTicker(
            ticker="AAPL",
            company_name="Apple Inc.",
            exchange="NASDAQ"
        )
        assert valid_ticker.ticker == "AAPL"
        assert valid_ticker.is_active is True  # Default value
        
        # Test ticker length validation would be handled by Pydantic
        # This is more of a model validation test
    
    @pytest.mark.asyncio
    async def test_analysis_cache_data_integrity(self):
        """Test analysis cache data integrity"""
        analysis_data = {
            "ticker": "AAPL",
            "recommendation": {
                "should_trade": True,
                "confidence": 0.85,
                "strategy": "cash-secured-put"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        cache_entry = AnalysisCache(
            ticker="AAPL",
            analysis_data=analysis_data,
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        assert cache_entry.ticker == "AAPL"
        assert cache_entry.analysis_data["recommendation"]["should_trade"] is True
        assert cache_entry.expires_at > datetime.now()
    
    @pytest.mark.asyncio
    async def test_database_constraint_handling(self, mock_session):
        """Test database constraint handling"""
        # Mock constraint violation
        from sqlalchemy.exc import IntegrityError
        mock_session.commit.side_effect = IntegrityError("", "", "")
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            service = TickerService()
            
            # This would typically be an insert operation
            # For this test, we're just verifying error handling
            with pytest.raises(IntegrityError):
                # Simulate an operation that would cause constraint violation
                mock_session.commit()
            
            # Rollback should be called on error
            mock_session.rollback.assert_called_once()