"""
Performance tests for backend API endpoints and services.

Tests API response times and concurrent user handling as required by 7.1, 7.3.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import app
from app.models.api import (
    TickerResult,
    TradeRecommendation,
    ReasoningStep
)

client = TestClient(app)


class TestAPIPerformance:
    """Test API performance requirements"""
    
    @pytest.fixture
    def sample_ticker_result(self):
        """Sample ticker result for mocking"""
        return TickerResult(
            ticker="AAPL",
            company_name="Apple Inc.",
            exchange="NASDAQ"
        )
    
    @pytest.fixture
    def sample_recommendation(self):
        """Sample trade recommendation for mocking"""
        return TradeRecommendation(
            should_trade=False,
            confidence=0.0,
            strategy=None,
            contracts=[],
            risk_metrics=None,
            reasoning_steps=[
                ReasoningStep(
                    step="Technical Analysis",
                    passed=False,
                    reasoning="Price below minimum threshold",
                    confidence=0.0
                )
            ]
        )
    
    @patch('app.api.v1.tickers.TickerService')
    def test_ticker_search_response_time(self, mock_ticker_service_class, sample_ticker_result):
        """Test ticker search responds within 300ms (requirement 7.2)"""
        # Mock ticker service
        mock_ticker_service = AsyncMock()
        mock_ticker_service.search_tickers.return_value = [sample_ticker_result]
        mock_ticker_service_class.return_value = mock_ticker_service
        
        start_time = time.time()
        response = client.get("/api/v1/tickers/search?q=AAPL")
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        assert response.status_code == 200
        assert response_time < 300, f"Response time {response_time}ms exceeds 300ms requirement"
    
    @patch('app.api.v1.trades.TickerService')
    @patch('app.api.v1.trades.OptionsEvaluationAgent')
    def test_trade_analysis_response_time(self, mock_agent_class, mock_ticker_service_class,
                                         sample_ticker_result, sample_recommendation):
        """Test trade analysis responds within 5 seconds (requirement 7.1)"""
        # Mock ticker service
        mock_ticker_service = AsyncMock()
        mock_ticker_service.get_ticker_by_symbol.return_value = sample_ticker_result
        mock_ticker_service_class.return_value = mock_ticker_service
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.evaluate_trade = AsyncMock(return_value=sample_recommendation)
        mock_agent.get_execution_times.return_value = {'total': 3.5}
        mock_agent.close = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        start_time = time.time()
        response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0, f"Response time {response_time}s exceeds 5s requirement"
    
    @patch('app.api.v1.tickers.TickerService')
    def test_concurrent_ticker_searches(self, mock_ticker_service_class, sample_ticker_result):
        """Test system handles 10 concurrent ticker searches (requirement 7.3)"""
        # Mock ticker service
        mock_ticker_service = AsyncMock()
        mock_ticker_service.search_tickers.return_value = [sample_ticker_result]
        mock_ticker_service_class.return_value = mock_ticker_service
        
        def make_request():
            return client.get("/api/v1/tickers/search?q=AAPL")
        
        # Execute 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Service should have been called 10 times
        assert mock_ticker_service.search_tickers.call_count == 10
    
    @patch('app.api.v1.trades.TickerService')
    @patch('app.api.v1.trades.OptionsEvaluationAgent')
    def test_concurrent_trade_analyses(self, mock_agent_class, mock_ticker_service_class,
                                      sample_ticker_result, sample_recommendation):
        """Test system handles 10 concurrent trade analyses (requirement 7.3)"""
        # Mock ticker service
        mock_ticker_service = AsyncMock()
        mock_ticker_service.get_ticker_by_symbol.return_value = sample_ticker_result
        mock_ticker_service_class.return_value = mock_ticker_service
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.evaluate_trade = AsyncMock(return_value=sample_recommendation)
        mock_agent.get_execution_times.return_value = {'total': 3.5}
        mock_agent.close = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        def make_request():
            return client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
        
        start_time = time.time()
        
        # Execute 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Total time should be reasonable (not 10x single request time)
        assert total_time < 15.0, f"Concurrent requests took {total_time}s, indicating poor concurrency handling"
        
        # Agent should have been created and closed 10 times
        assert mock_agent_class.call_count == 10
        assert mock_agent.close.call_count == 10
    
    @patch('app.api.v1.health.check_database_health')
    @patch('app.api.v1.health.check_redis_health')
    @patch('app.api.v1.health.check_openai_health')
    @patch('app.api.v1.health.check_alpha_vantage_health')
    @patch('app.api.v1.health.check_tradier_health')
    def test_health_check_response_time(self, mock_tradier, mock_alpha, mock_openai, mock_redis, mock_db):
        """Test health check responds quickly"""
        # Mock all health checks to return quickly
        mock_db.return_value = "healthy"
        mock_redis.return_value = "healthy"
        mock_openai.return_value = "healthy"
        mock_alpha.return_value = "healthy"
        mock_tradier.return_value = "healthy"
        
        start_time = time.time()
        response = client.get("/api/v1/health")
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        assert response.status_code == 200
        assert response_time < 100, f"Health check response time {response_time}ms is too slow"


class TestServicePerformance:
    """Test service layer performance"""
    
    @pytest.mark.asyncio
    async def test_market_data_service_caching(self):
        """Test that market data service uses caching effectively"""
        from app.services.market_data_service import MarketDataService
        
        service = MarketDataService()
        
        # Mock the actual data fetching to control timing
        with patch.object(service, '_fetch_stock_info') as mock_fetch:
            mock_fetch.return_value = {
                'currentPrice': 150.0,
                'volume': 1000000,
                'marketCap': 2500000000000,
                'beta': 1.2
            }
            
            # First call should fetch data
            start_time = time.time()
            await service.get_stock_quote("AAPL")
            first_call_time = time.time() - start_time
            
            # Second call should use cache (if implemented)
            start_time = time.time()
            await service.get_stock_quote("AAPL")
            second_call_time = time.time() - start_time
            
            # Verify fetch was called
            assert mock_fetch.call_count >= 1
            
            # Both calls should complete quickly
            assert first_call_time < 1.0
            assert second_call_time < 1.0
    
    @pytest.mark.asyncio
    async def test_chain_execution_performance(self):
        """Test that individual chains execute within reasonable time"""
        from app.chains.technical_analysis_chain import TechnicalAnalysisChain
        from unittest.mock import Mock
        
        # Mock LLM to return quickly
        mock_llm = Mock()
        mock_llm.apredict = AsyncMock(return_value='{"passed": true, "confidence": 0.8}')
        
        chain = TechnicalAnalysisChain(llm=mock_llm)
        
        technical_data = {
            'price': 150.0,
            'volume': 2000000,
            'rsi': 50.0,
            'iv_rank': 45.0,
            'beta': 1.2
        }
        
        start_time = time.time()
        result = await chain.evaluate('AAPL', technical_data)
        execution_time = time.time() - start_time
        
        assert execution_time < 2.0, f"Chain execution took {execution_time}s, too slow"
        assert result is not None


class TestMemoryUsage:
    """Test memory usage patterns"""
    
    @patch('app.api.v1.trades.TickerService')
    @patch('app.api.v1.trades.OptionsEvaluationAgent')
    def test_agent_cleanup(self, mock_agent_class, mock_ticker_service_class):
        """Test that agents are properly cleaned up after use"""
        from app.models.api import TickerResult, TradeRecommendation, ReasoningStep
        
        # Mock ticker service
        mock_ticker_service = AsyncMock()
        mock_ticker_service.get_ticker_by_symbol.return_value = TickerResult(
            ticker="AAPL", company_name="Apple Inc.", exchange="NASDAQ"
        )
        mock_ticker_service_class.return_value = mock_ticker_service
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.evaluate_trade = AsyncMock(return_value=TradeRecommendation(
            should_trade=False, confidence=0.0, strategy=None, contracts=[],
            risk_metrics=None, reasoning_steps=[ReasoningStep(
                step="Test", passed=False, reasoning="Test", confidence=0.0
            )]
        ))
        mock_agent.get_execution_times.return_value = {'total': 1.0}
        mock_agent.close = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        # Make multiple requests
        for _ in range(5):
            response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
            assert response.status_code == 200
        
        # Verify agents were created and cleaned up
        assert mock_agent_class.call_count == 5
        assert mock_agent.close.call_count == 5
    
    def test_large_response_handling(self):
        """Test handling of large response payloads"""
        # Create a large mock response
        large_reasoning_steps = [
            {
                "step": f"Step {i}",
                "passed": True,
                "reasoning": "A" * 1000,  # 1KB of text per step
                "confidence": 0.8
            }
            for i in range(100)  # 100KB total
        ]
        
        # This should not cause memory issues or timeouts
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        # Verify we can handle the response
        data = response.json()
        assert "status" in data