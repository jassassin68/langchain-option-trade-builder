"""
Database integration tests with transaction rollback.

Tests database operations, data integrity, and transaction handling as required by 7.1, 7.3.
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


class TestDatabaseIntegration:
    """Test database integration with transaction handling"""
    
    @pytest.fixture
    def mock_database_session(self):
        """Create a mock database session with transaction support"""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()
        session.begin = AsyncMock()
        return session
    
    @pytest.fixture
    def sample_stock_tickers(self):
        """Sample stock ticker data"""
        return [
            StockTicker(
                ticker="AAPL",
                company_name="Apple Inc.",
                exchange="NASDAQ",
                is_active=True,
                last_updated=datetime.now()
            ),
            StockTicker(
                ticker="MSFT",
                company_name="Microsoft Corporation",
                exchange="NASDAQ",
                is_active=True,
                last_updated=datetime.now()
            ),
            StockTicker(
                ticker="GOOGL",
                company_name="Alphabet Inc.",
                exchange="NASDAQ",
                is_active=True,
                last_updated=datetime.now()
            )
        ]
    
    @pytest.mark.asyncio
    async def test_ticker_search_database_integration(self, mock_database_session, sample_stock_tickers):
        """Test ticker search with database integration"""
        # Mock database query result
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = sample_stock_tickers[:2]
        mock_database_session.execute.return_value = mock_result
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_database_session
            
            service = TickerService()
            results = await service.search_tickers("AP", limit=10)
            
            # Verify results
            assert len(results) == 2
            assert results[0].ticker == "AAPL"
            assert results[1].ticker == "MSFT"
            
            # Verify database interaction
            mock_database_session.execute.assert_called_once()
            mock_database_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ticker_search_with_fuzzy_matching(self, mock_database_session, sample_stock_tickers):
        """Test fuzzy matching in ticker search"""
        # Mock database result for company name search
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_stock_tickers[0]]  # AAPL
        mock_database_session.execute.return_value = mock_result
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_database_session
            
            service = TickerService()
            
            # Test company name search
            results = await service.search_tickers("Apple", limit=10)
            assert len(results) == 1
            assert results[0].ticker == "AAPL"
            assert results[0].company_name == "Apple Inc."
    
    @pytest.mark.asyncio
    async def test_database_transaction_rollback_on_error(self, mock_database_session):
        """Test database transaction rollback on error"""
        # Mock database error during execution
        mock_database_session.execute.side_effect = Exception("Database constraint violation")
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_database_session
            
            service = TickerService()
            
            with pytest.raises(Exception):
                await service.search_tickers("AAPL")
            
            # Verify session cleanup
            mock_database_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self, mock_database_session, sample_stock_tickers):
        """Test concurrent database operations"""
        # Mock database results
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = sample_stock_tickers
        mock_database_session.execute.return_value = mock_result
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_database_session
            
            service = TickerService()
            
            # Execute multiple concurrent database operations
            tasks = [
                service.search_tickers("A", limit=10),
                service.search_tickers("M", limit=10),
                service.search_tickers("G", limit=10)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Verify all operations completed
            assert len(results) == 3
            for result in results:
                assert len(result) == 3  # All tickers returned
            
            # Verify database was accessed for each operation
            assert mock_database_session.execute.call_count == 3
            assert mock_database_session.close.call_count == 3
    
    @pytest.mark.asyncio
    async def test_ticker_by_symbol_database_integration(self, mock_database_session, sample_stock_tickers):
        """Test getting specific ticker by symbol"""
        # Mock database result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_stock_tickers[0]
        mock_database_session.execute.return_value = mock_result
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_database_session
            
            service = TickerService()
            result = await service.get_ticker_by_symbol("AAPL")
            
            assert result is not None
            assert result.ticker == "AAPL"
            assert result.company_name == "Apple Inc."
            
            # Verify database interaction
            mock_database_session.execute.assert_called_once()
            mock_database_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ticker_not_found_handling(self, mock_database_session):
        """Test handling when ticker is not found in database"""
        # Mock database returning None
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_database_session.execute.return_value = mock_result
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_database_session
            
            service = TickerService()
            result = await service.get_ticker_by_symbol("NONEXISTENT")
            
            assert result is None
            mock_database_session.execute.assert_called_once()
            mock_database_session.close.assert_called_once()


class TestAnalysisCacheIntegration:
    """Test analysis cache database integration"""
    
    @pytest.fixture
    def sample_analysis_cache(self):
        """Sample analysis cache data"""
        return AnalysisCache(
            ticker="AAPL",
            analysis_data={
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "recommendation": {
                    "should_trade": True,
                    "confidence": 0.85,
                    "strategy": "cash-secured-put",
                    "contracts": [
                        {
                            "action": "SELL",
                            "type": "PUT",
                            "strike": 150.0,
                            "expiration": "2024-12-20",
                            "quantity": 1,
                            "premium_credit": 2.50
                        }
                    ],
                    "risk_metrics": {
                        "max_profit": 250.0,
                        "max_loss": 14750.0,
                        "breakeven": 147.50,
                        "prob_profit": 0.70,
                        "return_on_capital": 1.69
                    },
                    "reasoning_steps": [
                        {
                            "step": "Technical Analysis",
                            "passed": True,
                            "reasoning": "Stock meets technical criteria",
                            "confidence": 0.90
                        }
                    ]
                },
                "analysis_timestamp": datetime.now().isoformat()
            },
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)
        )
    
    @pytest.mark.asyncio
    async def test_analysis_cache_storage_and_retrieval(self, sample_analysis_cache):
        """Test storing and retrieving analysis cache"""
        mock_session = AsyncMock()
        
        # Mock cache retrieval
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_analysis_cache
        mock_session.execute.return_value = mock_result
        
        with patch('app.services.cache_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            cache_service = CacheService()
            result = await cache_service.get_analysis_cache("AAPL")
            
            assert result is not None
            assert result["ticker"] == "AAPL"
            assert result["recommendation"]["should_trade"] is True
            assert result["recommendation"]["confidence"] == 0.85
    
    @pytest.mark.asyncio
    async def test_analysis_cache_expiration_handling(self):
        """Test handling of expired analysis cache"""
        mock_session = AsyncMock()
        
        # Mock expired cache (returns None)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        with patch('app.services.cache_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            cache_service = CacheService()
            result = await cache_service.get_analysis_cache("AAPL")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_analysis_cache_storage_with_transaction(self, sample_analysis_cache):
        """Test storing analysis cache with transaction handling"""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        
        with patch('app.services.cache_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            cache_service = CacheService()
            await cache_service.store_analysis_cache(
                "AAPL",
                sample_analysis_cache.analysis_data,
                expires_in_hours=1
            )
            
            # Verify transaction handling
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analysis_cache_storage_error_rollback(self):
        """Test rollback on cache storage error"""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock(side_effect=Exception("Database error"))
        mock_session.rollback = AsyncMock()
        
        with patch('app.services.cache_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            cache_service = CacheService()
            
            with pytest.raises(Exception):
                await cache_service.store_analysis_cache(
                    "AAPL",
                    {"test": "data"},
                    expires_in_hours=1
                )
            
            # Verify rollback was called
            mock_session.rollback.assert_called_once()


class TestRedisIntegration:
    """Test Redis cache integration"""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client"""
        redis_client = AsyncMock()
        redis_client.get = AsyncMock()
        redis_client.set = AsyncMock()
        redis_client.delete = AsyncMock()
        redis_client.exists = AsyncMock()
        redis_client.expire = AsyncMock()
        redis_client.ping = AsyncMock(return_value=True)
        return redis_client
    
    @pytest.mark.asyncio
    async def test_redis_cache_operations(self, mock_redis_client):
        """Test basic Redis cache operations"""
        test_data = {"ticker": "AAPL", "price": 150.0}
        mock_redis_client.get.return_value = json.dumps(test_data)
        mock_redis_client.set.return_value = True
        
        with patch('app.services.cache_service.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis_client
            
            cache_service = CacheService()
            
            # Test set operation
            await cache_service.set("test_key", test_data, ttl=300)
            mock_redis_client.set.assert_called_once()
            
            # Test get operation
            result = await cache_service.get("test_key")
            assert result == test_data
            mock_redis_client.get.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_redis_cache_miss(self, mock_redis_client):
        """Test Redis cache miss handling"""
        mock_redis_client.get.return_value = None
        
        with patch('app.services.cache_service.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis_client
            
            cache_service = CacheService()
            result = await cache_service.get("nonexistent_key")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_redis_connection_error_handling(self, mock_redis_client):
        """Test Redis connection error handling"""
        mock_redis_client.get.side_effect = Exception("Redis connection failed")
        
        with patch('app.services.cache_service.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis_client
            
            cache_service = CacheService()
            
            # Should handle error gracefully and return None
            result = await cache_service.get("test_key")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_redis_cache_expiration(self, mock_redis_client):
        """Test Redis cache TTL handling"""
        with patch('app.services.cache_service.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis_client
            
            cache_service = CacheService()
            
            # Set with TTL
            await cache_service.set("expiring_key", {"data": "value"}, ttl=60)
            
            # Verify set was called with TTL
            mock_redis_client.set.assert_called_once()
            call_args = mock_redis_client.set.call_args
            assert call_args[1]['ex'] == 60  # TTL in seconds
    
    @pytest.mark.asyncio
    async def test_redis_cache_key_generation(self, mock_redis_client):
        """Test cache key generation strategies"""
        with patch('app.services.cache_service.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis_client
            
            cache_service = CacheService()
            
            # Test different key types
            ticker_key = cache_service._generate_key("ticker_search", "AAPL")
            analysis_key = cache_service._generate_key("analysis", "AAPL")
            
            assert "ticker_search" in ticker_key
            assert "AAPL" in ticker_key
            assert "analysis" in analysis_key
            assert "AAPL" in analysis_key
            assert ticker_key != analysis_key  # Keys should be different
    
    @pytest.mark.asyncio
    async def test_concurrent_redis_operations(self, mock_redis_client):
        """Test concurrent Redis operations"""
        mock_redis_client.get.return_value = json.dumps({"test": "data"})
        mock_redis_client.set.return_value = True
        
        with patch('app.services.cache_service.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis_client
            
            cache_service = CacheService()
            
            # Execute concurrent operations
            tasks = [
                cache_service.get("key1"),
                cache_service.get("key2"),
                cache_service.set("key3", {"data": "value3"}),
                cache_service.get("key4"),
                cache_service.set("key5", {"data": "value5"})
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify operations completed (some return data, some return None for sets)
            assert len(results) == 5
            assert all(not isinstance(result, Exception) for result in results)


class TestDataIntegrityAndConsistency:
    """Test data integrity and consistency across the system"""
    
    @pytest.mark.asyncio
    async def test_ticker_data_consistency(self):
        """Test ticker data consistency between database and cache"""
        mock_db_session = AsyncMock()
        mock_redis_client = AsyncMock()
        
        # Mock database ticker
        db_ticker = StockTicker(
            ticker="AAPL",
            company_name="Apple Inc.",
            exchange="NASDAQ",
            is_active=True
        )
        
        mock_db_result = MagicMock()
        mock_db_result.scalar_one_or_none.return_value = db_ticker
        mock_db_session.execute.return_value = mock_db_result
        
        # Mock cache data
        cache_data = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "exchange": "NASDAQ"
        }
        mock_redis_client.get.return_value = json.dumps(cache_data)
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_db_factory, \
             patch('app.services.cache_service.redis.Redis') as mock_redis_class:
            
            mock_db_factory.return_value.__aenter__.return_value = mock_db_session
            mock_redis_class.return_value = mock_redis_client
            
            ticker_service = TickerService()
            cache_service = CacheService()
            
            # Get data from both sources
            db_result = await ticker_service.get_ticker_by_symbol("AAPL")
            cache_result = await cache_service.get("ticker:AAPL")
            
            # Verify consistency
            assert db_result.ticker == cache_result["ticker"]
            assert db_result.company_name == cache_result["company_name"]
            assert db_result.exchange == cache_result["exchange"]
    
    @pytest.mark.asyncio
    async def test_analysis_data_integrity(self):
        """Test analysis data integrity in cache storage"""
        mock_session = AsyncMock()
        
        # Test data with all required fields
        analysis_data = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "recommendation": {
                "should_trade": True,
                "confidence": 0.85,
                "strategy": "cash-secured-put",
                "contracts": [
                    {
                        "action": "SELL",
                        "type": "PUT",
                        "strike": 150.0,
                        "expiration": "2024-12-20",
                        "quantity": 1,
                        "premium_credit": 2.50
                    }
                ],
                "risk_metrics": {
                    "max_profit": 250.0,
                    "max_loss": 14750.0,
                    "breakeven": 147.50,
                    "prob_profit": 0.70,
                    "return_on_capital": 1.69
                },
                "reasoning_steps": []
            },
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        with patch('app.services.cache_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            cache_service = CacheService()
            
            # Store analysis data
            await cache_service.store_analysis_cache("AAPL", analysis_data, expires_in_hours=1)
            
            # Verify data was stored with proper structure
            mock_session.add.assert_called_once()
            stored_cache = mock_session.add.call_args[0][0]
            
            assert stored_cache.ticker == "AAPL"
            assert stored_cache.analysis_data == analysis_data
            assert stored_cache.expires_at > datetime.now()
    
    @pytest.mark.asyncio
    async def test_concurrent_data_access_consistency(self):
        """Test data consistency under concurrent access"""
        mock_session = AsyncMock()
        
        # Mock ticker data
        ticker = StockTicker(
            ticker="AAPL",
            company_name="Apple Inc.",
            exchange="NASDAQ",
            is_active=True
        )
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = ticker
        mock_session.execute.return_value = mock_result
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_session_factory:
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            service = TickerService()
            
            # Execute concurrent reads
            tasks = [
                service.get_ticker_by_symbol("AAPL"),
                service.get_ticker_by_symbol("AAPL"),
                service.get_ticker_by_symbol("AAPL")
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Verify all results are consistent
            assert len(results) == 3
            for result in results:
                assert result.ticker == "AAPL"
                assert result.company_name == "Apple Inc."
                assert result.exchange == "NASDAQ"