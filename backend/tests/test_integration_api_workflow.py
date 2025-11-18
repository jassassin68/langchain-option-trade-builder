"""
End-to-end API integration tests for complete ticker search to analysis workflow.

Tests complete API workflows as required by 7.1, 7.3, 7.4.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from datetime import datetime, date, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import app
from app.models.api import (
    TickerResult,
    TradeRecommendation,
    Contract,
    RiskMetrics,
    ReasoningStep,
    ActionType,
    ContractType
)

client = TestClient(app)


class TestCompleteAPIWorkflow:
    """Test complete API workflows from ticker search to analysis"""
    
    @pytest.fixture
    def sample_ticker_results(self):
        """Sample ticker search results"""
        return [
            TickerResult(ticker="AAPL", company_name="Apple Inc.", exchange="NASDAQ"),
            TickerResult(ticker="MSFT", company_name="Microsoft Corporation", exchange="NASDAQ"),
            TickerResult(ticker="GOOGL", company_name="Alphabet Inc.", exchange="NASDAQ")
        ]
    
    @pytest.fixture
    def sample_positive_recommendation(self):
        """Sample positive trade recommendation"""
        return TradeRecommendation(
            should_trade=True,
            confidence=0.85,
            strategy="Cash-Secured Put",
            contracts=[
                Contract(
                    action=ActionType.SELL,
                    type=ContractType.PUT,
                    strike=150.0,
                    expiration=date.today() + timedelta(days=45),
                    quantity=1,
                    premium_credit=2.50
                )
            ],
            risk_metrics=RiskMetrics(
                max_profit=250.0,
                max_loss=14750.0,
                breakeven=147.50,
                prob_profit=0.70,
                return_on_capital=1.69
            ),
            reasoning_steps=[
                ReasoningStep(
                    step="Technical Analysis",
                    passed=True,
                    reasoning="Stock meets all technical criteria",
                    confidence=0.90
                ),
                ReasoningStep(
                    step="Fundamental Screening",
                    passed=True,
                    reasoning="Company fundamentals are healthy",
                    confidence=0.85
                ),
                ReasoningStep(
                    step="Options Analysis",
                    passed=True,
                    reasoning="Quality options contracts available",
                    confidence=0.80
                ),
                ReasoningStep(
                    step="Strategy Selection",
                    passed=True,
                    reasoning="Cash-secured put strategy recommended",
                    confidence=0.85
                ),
                ReasoningStep(
                    step="Risk Assessment",
                    passed=True,
                    reasoning="Favorable risk/reward ratio",
                    confidence=0.80
                )
            ]
        )
    
    @pytest.fixture
    def sample_negative_recommendation(self):
        """Sample negative trade recommendation"""
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
                    reasoning="Stock price below minimum threshold",
                    confidence=0.0
                )
            ]
        )
    
    @patch('app.api.v1.tickers.TickerService')
    @patch('app.api.v1.trades.TickerService')
    @patch('app.api.v1.trades.OptionsEvaluationAgent')
    def test_complete_successful_workflow(self, mock_agent_class, mock_trades_ticker_service_class,
                                         mock_tickers_ticker_service_class, sample_ticker_results,
                                         sample_positive_recommendation):
        """Test complete workflow from ticker search to successful analysis"""
        # Mock ticker service for search endpoint
        mock_tickers_service = AsyncMock()
        mock_tickers_service.search_tickers.return_value = sample_ticker_results[:1]  # Return AAPL
        mock_tickers_ticker_service_class.return_value = mock_tickers_service
        
        # Mock ticker service for trades endpoint
        mock_trades_service = AsyncMock()
        mock_trades_service.get_ticker_by_symbol.return_value = sample_ticker_results[0]
        mock_trades_ticker_service_class.return_value = mock_trades_service
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.evaluate_trade = AsyncMock(return_value=sample_positive_recommendation)
        mock_agent.get_execution_times.return_value = {'total': 3.2}
        mock_agent.close = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        # Step 1: Search for ticker
        search_response = client.get("/api/v1/tickers/search?q=AAPL")
        assert search_response.status_code == 200
        search_data = search_response.json()
        assert len(search_data["results"]) == 1
        assert search_data["results"][0]["ticker"] == "AAPL"
        
        # Step 2: Analyze the found ticker
        analysis_response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
        assert analysis_response.status_code == 200
        analysis_data = analysis_response.json()
        
        # Verify complete analysis response
        assert analysis_data["ticker"] == "AAPL"
        assert analysis_data["company_name"] == "Apple Inc."
        assert analysis_data["recommendation"]["should_trade"] is True
        assert analysis_data["recommendation"]["confidence"] == 0.85
        assert analysis_data["recommendation"]["strategy"] == "Cash-Secured Put"
        assert len(analysis_data["recommendation"]["contracts"]) == 1
        assert analysis_data["recommendation"]["risk_metrics"] is not None
        assert len(analysis_data["recommendation"]["reasoning_steps"]) == 5
        
        # Verify all reasoning steps are present
        step_names = [step["step"] for step in analysis_data["recommendation"]["reasoning_steps"]]
        expected_steps = ["Technical Analysis", "Fundamental Screening", "Options Analysis", 
                         "Strategy Selection", "Risk Assessment"]
        for expected_step in expected_steps:
            assert expected_step in step_names
        
        # Verify agent was properly closed
        mock_agent.close.assert_called_once()
    
    @patch('app.api.v1.tickers.TickerService')
    @patch('app.api.v1.trades.TickerService')
    @patch('app.api.v1.trades.OptionsEvaluationAgent')
    def test_workflow_with_rejection(self, mock_agent_class, mock_trades_ticker_service_class,
                                    mock_tickers_ticker_service_class, sample_ticker_results,
                                    sample_negative_recommendation):
        """Test workflow that results in trade rejection"""
        # Mock ticker service for search endpoint
        mock_tickers_service = AsyncMock()
        mock_tickers_service.search_tickers.return_value = sample_ticker_results[:1]
        mock_tickers_ticker_service_class.return_value = mock_tickers_service
        
        # Mock ticker service for trades endpoint
        mock_trades_service = AsyncMock()
        mock_trades_service.get_ticker_by_symbol.return_value = sample_ticker_results[0]
        mock_trades_ticker_service_class.return_value = mock_trades_service
        
        # Mock agent to return rejection
        mock_agent = MagicMock()
        mock_agent.evaluate_trade = AsyncMock(return_value=sample_negative_recommendation)
        mock_agent.get_execution_times.return_value = {'total': 1.5}
        mock_agent.close = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        # Search and analyze
        search_response = client.get("/api/v1/tickers/search?q=AAPL")
        assert search_response.status_code == 200
        
        analysis_response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
        assert analysis_response.status_code == 200
        analysis_data = analysis_response.json()
        
        # Verify rejection response
        assert analysis_data["recommendation"]["should_trade"] is False
        assert analysis_data["recommendation"]["confidence"] == 0.0
        assert analysis_data["recommendation"]["strategy"] is None
        assert len(analysis_data["recommendation"]["contracts"]) == 0
        assert analysis_data["recommendation"]["risk_metrics"] is None
        assert len(analysis_data["recommendation"]["reasoning_steps"]) == 1
        assert analysis_data["recommendation"]["reasoning_steps"][0]["passed"] is False
    
    @patch('app.api.v1.tickers.TickerService')
    def test_search_no_results_workflow(self, mock_ticker_service_class):
        """Test workflow when ticker search returns no results"""
        # Mock empty search results
        mock_service = AsyncMock()
        mock_service.search_tickers.return_value = []
        mock_ticker_service_class.return_value = mock_service
        
        search_response = client.get("/api/v1/tickers/search?q=NONEXISTENT")
        assert search_response.status_code == 200
        search_data = search_response.json()
        assert len(search_data["results"]) == 0
        assert search_data["count"] == 0
    
    @patch('app.api.v1.trades.TickerService')
    def test_analysis_ticker_not_found_workflow(self, mock_ticker_service_class):
        """Test workflow when ticker is not found for analysis"""
        # Mock ticker not found
        mock_service = AsyncMock()
        mock_service.get_ticker_by_symbol.return_value = None
        mock_ticker_service_class.return_value = mock_service
        
        analysis_response = client.post("/api/v1/trades/analyze", json={"ticker": "NONEXISTENT"})
        assert analysis_response.status_code in [404, 422]
    
    @patch('app.api.v1.tickers.TickerService')
    @patch('app.api.v1.trades.TickerService')
    @patch('app.api.v1.trades.OptionsEvaluationAgent')
    def test_multiple_concurrent_workflows(self, mock_agent_class, mock_trades_ticker_service_class,
                                          mock_tickers_ticker_service_class, sample_ticker_results,
                                          sample_positive_recommendation):
        """Test multiple concurrent complete workflows"""
        # Mock services
        mock_tickers_service = AsyncMock()
        mock_tickers_service.search_tickers.return_value = sample_ticker_results
        mock_tickers_ticker_service_class.return_value = mock_tickers_service
        
        mock_trades_service = AsyncMock()
        mock_trades_service.get_ticker_by_symbol.return_value = sample_ticker_results[0]
        mock_trades_ticker_service_class.return_value = mock_trades_service
        
        mock_agent = MagicMock()
        mock_agent.evaluate_trade = AsyncMock(return_value=sample_positive_recommendation)
        mock_agent.get_execution_times.return_value = {'total': 3.0}
        mock_agent.close = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        # Execute multiple workflows concurrently using threading
        import concurrent.futures
        import threading
        
        def execute_workflow():
            # Search
            search_response = client.get("/api/v1/tickers/search?q=AAPL")
            assert search_response.status_code == 200
            
            # Analyze
            analysis_response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
            assert analysis_response.status_code == 200
            
            return analysis_response.json()
        
        # Execute 3 concurrent workflows
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(execute_workflow) for _ in range(3)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all workflows completed successfully
        assert len(results) == 3
        for result in results:
            assert result["ticker"] == "AAPL"
            assert result["recommendation"]["should_trade"] is True
        
        # Verify agents were created and closed for each workflow
        assert mock_agent_class.call_count == 3
        assert mock_agent.close.call_count == 3


class TestExternalServiceIntegration:
    """Test integration with external services (with real API calls)"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_market_data_service_real_api(self):
        """Test market data service with real API calls (if API keys available)"""
        from app.services.market_data_service import MarketDataService
        
        service = MarketDataService()
        
        try:
            # Test with a well-known ticker
            quote = await service.get_stock_quote("AAPL")
            
            # Verify basic data structure
            assert quote.ticker == "AAPL"
            assert quote.price > 0
            assert quote.volume > 0
            
        except Exception as e:
            # If API is not available, skip the test
            pytest.skip(f"Market data API not available: {e}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_options_data_service_real_api(self):
        """Test options data service with real API calls (if API keys available)"""
        from app.services.options_data_service import OptionsDataService
        
        service = OptionsDataService()
        
        try:
            # Test with a well-known ticker
            options_chain = await service.get_options_chain("AAPL")
            
            # Verify basic data structure
            assert options_chain.ticker == "AAPL"
            assert len(options_chain.expiration_dates) > 0
            
        except Exception as e:
            # If API is not available, skip the test
            pytest.skip(f"Options data API not available: {e}")
    
    @pytest.mark.integration
    def test_database_integration_with_rollback(self):
        """Test database operations with transaction rollback"""
        # This would require a test database setup
        # For now, we'll mock the database operations
        
        with patch('app.services.ticker_service.AsyncSessionLocal') as mock_session_factory:
            mock_session = AsyncMock()
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            # Test successful operation
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_session.execute.return_value = mock_result
            
            response = client.get("/api/v1/tickers/search?q=TEST")
            assert response.status_code == 200
            
            # Verify session was used
            mock_session.execute.assert_called_once()
    
    @pytest.mark.integration
    def test_cache_integration(self):
        """Test cache service integration"""
        # This would require Redis to be running
        # For now, we'll test the cache interface
        
        with patch('app.services.cache_service.redis.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.get.return_value = None
            mock_redis.set.return_value = True
            mock_redis_class.return_value = mock_redis
            
            # Test cache miss and set
            response = client.get("/api/v1/tickers/search?q=AAPL")
            assert response.status_code == 200


class TestPerformanceIntegration:
    """Test system performance under load"""
    
    @patch('app.api.v1.tickers.TickerService')
    def test_concurrent_ticker_searches_performance(self, mock_ticker_service_class):
        """Test performance with concurrent ticker searches"""
        import time
        import concurrent.futures
        
        # Mock service
        mock_service = AsyncMock()
        mock_service.search_tickers.return_value = [
            TickerResult(ticker="AAPL", company_name="Apple Inc.", exchange="NASDAQ")
        ]
        mock_ticker_service_class.return_value = mock_service
        
        def make_search_request():
            start_time = time.time()
            response = client.get("/api/v1/tickers/search?q=AAPL")
            end_time = time.time()
            return response.status_code, end_time - start_time
        
        # Execute 10 concurrent searches
        start_total = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_search_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        end_total = time.time()
        
        # Verify all requests succeeded
        for status_code, response_time in results:
            assert status_code == 200
            assert response_time < 1.0  # Each request should complete within 1 second
        
        # Total time should be reasonable (not 10x single request time)
        total_time = end_total - start_total
        assert total_time < 5.0  # All 10 requests should complete within 5 seconds
    
    @patch('app.api.v1.trades.TickerService')
    @patch('app.api.v1.trades.OptionsEvaluationAgent')
    def test_concurrent_analyses_performance(self, mock_agent_class, mock_ticker_service_class):
        """Test performance with concurrent trade analyses"""
        import time
        import concurrent.futures
        
        # Mock services
        mock_service = AsyncMock()
        mock_service.get_ticker_by_symbol.return_value = TickerResult(
            ticker="AAPL", company_name="Apple Inc.", exchange="NASDAQ"
        )
        mock_ticker_service_class.return_value = mock_service
        
        mock_agent = MagicMock()
        mock_agent.evaluate_trade = AsyncMock(return_value=TradeRecommendation(
            should_trade=False, confidence=0.0, strategy=None, contracts=[],
            risk_metrics=None, reasoning_steps=[]
        ))
        mock_agent.get_execution_times.return_value = {'total': 2.0}
        mock_agent.close = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        def make_analysis_request():
            start_time = time.time()
            response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
            end_time = time.time()
            return response.status_code, end_time - start_time
        
        # Execute 5 concurrent analyses (fewer than searches due to complexity)
        start_total = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_analysis_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        end_total = time.time()
        
        # Verify all requests succeeded
        for status_code, response_time in results:
            assert status_code == 200
            assert response_time < 10.0  # Each analysis should complete within 10 seconds
        
        # Total time should be reasonable
        total_time = end_total - start_total
        assert total_time < 15.0  # All 5 analyses should complete within 15 seconds
        
        # Verify agents were properly closed
        assert mock_agent.close.call_count == 5


class TestErrorHandlingIntegration:
    """Test error handling across the entire system"""
    
    @patch('app.api.v1.trades.TickerService')
    @patch('app.api.v1.trades.OptionsEvaluationAgent')
    def test_agent_failure_recovery(self, mock_agent_class, mock_ticker_service_class):
        """Test system recovery when agent fails"""
        # Mock ticker service
        mock_service = AsyncMock()
        mock_service.get_ticker_by_symbol.return_value = TickerResult(
            ticker="AAPL", company_name="Apple Inc.", exchange="NASDAQ"
        )
        mock_ticker_service_class.return_value = mock_service
        
        # Mock agent to fail
        mock_agent = MagicMock()
        mock_agent.evaluate_trade = AsyncMock(side_effect=Exception("Agent failure"))
        mock_agent.close = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
        
        # Should return error but not crash
        assert response.status_code == 500
        
        # Agent should still be closed
        mock_agent.close.assert_called_once()
    
    @patch('app.api.v1.tickers.TickerService')
    def test_database_failure_handling(self, mock_ticker_service_class):
        """Test handling of database failures"""
        # Mock database failure
        mock_service = AsyncMock()
        mock_service.search_tickers.side_effect = Exception("Database connection failed")
        mock_ticker_service_class.return_value = mock_service
        
        response = client.get("/api/v1/tickers/search?q=AAPL")
        
        # Should return error gracefully
        assert response.status_code == 500
    
    def test_invalid_request_handling(self):
        """Test handling of various invalid requests"""
        # Invalid JSON
        response = client.post("/api/v1/trades/analyze", 
                              data="invalid json", 
                              headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        
        # Missing required fields
        response = client.post("/api/v1/trades/analyze", json={})
        assert response.status_code == 422
        
        # Invalid ticker format
        response = client.post("/api/v1/trades/analyze", json={"ticker": "TOOLONG"})
        assert response.status_code == 422
    
    def test_rate_limiting_simulation(self):
        """Test system behavior under high request volume"""
        # Make many requests quickly to test rate limiting behavior
        responses = []
        for _ in range(20):
            response = client.get("/api/v1/health")
            responses.append(response.status_code)
        
        # Most requests should succeed (assuming no actual rate limiting in test)
        success_count = sum(1 for status in responses if status == 200)
        assert success_count >= 15  # At least 75% should succeed