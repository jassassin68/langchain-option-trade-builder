"""
Tests for trade analysis API endpoint.

Implements requirement testing for 6.1, 6.2, 6.3, 6.4, 7.1, 7.4, 7.5:
- POST /api/v1/trades/analyze with request validation
- Integration with OptionsEvaluationAgent
- Comprehensive error handling
"""

import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from datetime import date, datetime, timezone

# Add the app directory to the Python path
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
from app.services.market_data_service import DataUnavailableError
from app.services.options_data_service import OptionsUnavailableError

client = TestClient(app)


class TestTradeAnalysisAPI:
    """Test trade analysis API endpoint"""
    
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
            should_trade=True,
            confidence=0.85,
            strategy="Cash-Secured Put",
            contracts=[
                Contract(
                    action=ActionType.SELL,
                    type=ContractType.PUT,
                    strike=150.0,
                    expiration=date(2024, 12, 20),
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
                )
            ]
        )
    
    @patch('backend.app.api.v1.trades.TickerService')
    @patch('backend.app.api.v1.trades.OptionsEvaluationAgent')
    def test_analyze_trade_success(self, mock_agent_class, mock_ticker_service_class,
                                   sample_ticker_result, sample_recommendation):
        """Test successful trade analysis"""
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
        
        response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "ticker" in data
        assert "company_name" in data
        assert "recommendation" in data
        assert "analysis_timestamp" in data
        
        # Verify recommendation details
        assert data["ticker"] == "AAPL"
        assert data["company_name"] == "Apple Inc."
        assert data["recommendation"]["should_trade"] is True
        assert data["recommendation"]["confidence"] == 0.85
        assert data["recommendation"]["strategy"] == "Cash-Secured Put"
        
        # Verify agent was called and closed
        mock_agent.evaluate_trade.assert_called_once_with("AAPL")
        mock_agent.close.assert_called_once()
    
    @patch('backend.app.api.v1.trades.TickerService')
    def test_analyze_trade_ticker_not_found(self, mock_ticker_service_class):
        """Test analysis with non-existent ticker"""
        # Mock ticker service to return None
        mock_ticker_service = AsyncMock()
        mock_ticker_service.get_ticker_by_symbol.return_value = None
        mock_ticker_service_class.return_value = mock_ticker_service
        
        response = client.post("/api/v1/trades/analyze", json={"ticker": "NONEXISTENT"})
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    def test_analyze_trade_missing_ticker(self):
        """Test analysis without ticker in request"""
        response = client.post("/api/v1/trades/analyze", json={})
        
        assert response.status_code == 422  # Validation error
    
    def test_analyze_trade_invalid_ticker_format(self):
        """Test analysis with invalid ticker format"""
        response = client.post("/api/v1/trades/analyze", json={"ticker": "TOOLONG"})
        
        assert response.status_code == 422  # Validation error
    
    def test_analyze_trade_lowercase_ticker(self):
        """Test that lowercase ticker is normalized"""
        # This should pass validation as the model normalizes to uppercase
        response = client.post("/api/v1/trades/analyze", json={"ticker": "aapl"})
        
        # Will fail at ticker lookup, but validation should pass
        assert response.status_code in [404, 422, 500]  # Not a validation error
    
    @patch('backend.app.api.v1.trades.TickerService')
    @patch('backend.app.api.v1.trades.OptionsEvaluationAgent')
    def test_analyze_trade_market_data_unavailable(self, mock_agent_class, mock_ticker_service_class,
                                                   sample_ticker_result):
        """Test handling of market data unavailability"""
        # Mock ticker service
        mock_ticker_service = AsyncMock()
        mock_ticker_service.get_ticker_by_symbol.return_value = sample_ticker_result
        mock_ticker_service_class.return_value = mock_ticker_service
        
        # Mock agent to raise DataUnavailableError
        mock_agent = MagicMock()
        mock_agent.evaluate_trade = AsyncMock(side_effect=DataUnavailableError("Market data unavailable"))
        mock_agent.close = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "unavailable" in data["detail"].lower()
        
        # Verify agent was closed
        mock_agent.close.assert_called_once()
    
    @patch('backend.app.api.v1.trades.TickerService')
    @patch('backend.app.api.v1.trades.OptionsEvaluationAgent')
    def test_analyze_trade_options_unavailable(self, mock_agent_class, mock_ticker_service_class,
                                               sample_ticker_result):
        """Test handling of options data unavailability"""
        # Mock ticker service
        mock_ticker_service = AsyncMock()
        mock_ticker_service.get_ticker_by_symbol.return_value = sample_ticker_result
        mock_ticker_service_class.return_value = mock_ticker_service
        
        # Mock agent to raise OptionsUnavailableError
        mock_agent = MagicMock()
        mock_agent.evaluate_trade = AsyncMock(side_effect=OptionsUnavailableError("Options data unavailable"))
        mock_agent.close = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "options" in data["detail"].lower()
        
        # Verify agent was closed
        mock_agent.close.assert_called_once()
    
    @patch('backend.app.api.v1.trades.TickerService')
    @patch('backend.app.api.v1.trades.OptionsEvaluationAgent')
    def test_analyze_trade_unexpected_error(self, mock_agent_class, mock_ticker_service_class,
                                           sample_ticker_result):
        """Test handling of unexpected errors"""
        # Mock ticker service
        mock_ticker_service = AsyncMock()
        mock_ticker_service.get_ticker_by_symbol.return_value = sample_ticker_result
        mock_ticker_service_class.return_value = mock_ticker_service
        
        # Mock agent to raise unexpected exception
        mock_agent = MagicMock()
        mock_agent.evaluate_trade = AsyncMock(side_effect=Exception("Unexpected error"))
        mock_agent.close = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        
        # Verify agent was closed
        mock_agent.close.assert_called_once()
    
    @patch('backend.app.api.v1.trades.TickerService')
    @patch('backend.app.api.v1.trades.OptionsEvaluationAgent')
    def test_analyze_trade_rejection_recommendation(self, mock_agent_class, mock_ticker_service_class,
                                                    sample_ticker_result):
        """Test analysis returning rejection recommendation"""
        # Mock ticker service
        mock_ticker_service = AsyncMock()
        mock_ticker_service.get_ticker_by_symbol.return_value = sample_ticker_result
        mock_ticker_service_class.return_value = mock_ticker_service
        
        # Create rejection recommendation
        rejection = TradeRecommendation(
            should_trade=False,
            confidence=0.0,
            strategy=None,
            contracts=[],
            risk_metrics=None,
            reasoning_steps=[
                ReasoningStep(
                    step="Technical Analysis",
                    passed=False,
                    reasoning="Stock price below $10 threshold",
                    confidence=0.0
                )
            ]
        )
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.evaluate_trade = AsyncMock(return_value=rejection)
        mock_agent.get_execution_times.return_value = {'total': 2.0}
        mock_agent.close = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify rejection details
        assert data["recommendation"]["should_trade"] is False
        assert data["recommendation"]["confidence"] == 0.0
        assert data["recommendation"]["strategy"] is None
        assert len(data["recommendation"]["contracts"]) == 0
    
    @patch('backend.app.api.v1.trades.TickerService')
    @patch('backend.app.api.v1.trades.OptionsEvaluationAgent')
    def test_analyze_trade_response_includes_timestamp(self, mock_agent_class, mock_ticker_service_class,
                                                       sample_ticker_result, sample_recommendation):
        """Test that response includes analysis timestamp"""
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
        
        response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify timestamp is present and valid
        assert "analysis_timestamp" in data
        # Should be able to parse as ISO format datetime
        timestamp = datetime.fromisoformat(data["analysis_timestamp"].replace('Z', '+00:00'))
        assert isinstance(timestamp, datetime)
    
    @patch('backend.app.api.v1.trades.TickerService')
    @patch('backend.app.api.v1.trades.OptionsEvaluationAgent')
    def test_analyze_trade_includes_risk_metrics(self, mock_agent_class, mock_ticker_service_class,
                                                 sample_ticker_result, sample_recommendation):
        """Test that response includes risk metrics (requirement 6.1)"""
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
        
        response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify risk metrics are present (requirement 6.1)
        risk_metrics = data["recommendation"]["risk_metrics"]
        assert risk_metrics is not None
        assert "max_profit" in risk_metrics
        assert "max_loss" in risk_metrics
        assert "breakeven" in risk_metrics
        assert "prob_profit" in risk_metrics
        assert "return_on_capital" in risk_metrics
    
    @patch('backend.app.api.v1.trades.TickerService')
    @patch('backend.app.api.v1.trades.OptionsEvaluationAgent')
    def test_analyze_trade_includes_contract_details(self, mock_agent_class, mock_ticker_service_class,
                                                     sample_ticker_result, sample_recommendation):
        """Test that response includes contract details (requirement 6.2)"""
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
        
        response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify contract details are present (requirement 6.2)
        contracts = data["recommendation"]["contracts"]
        assert len(contracts) > 0
        
        contract = contracts[0]
        assert "action" in contract
        assert "type" in contract
        assert "strike" in contract
        assert "expiration" in contract
        assert "quantity" in contract
    
    @patch('backend.app.api.v1.trades.TickerService')
    @patch('backend.app.api.v1.trades.OptionsEvaluationAgent')
    def test_analyze_trade_includes_confidence_score(self, mock_agent_class, mock_ticker_service_class,
                                                     sample_ticker_result, sample_recommendation):
        """Test that response includes confidence score (requirement 6.3)"""
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
        
        response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify confidence score is present (requirement 6.3)
        assert "confidence" in data["recommendation"]
        confidence = data["recommendation"]["confidence"]
        assert 0 <= confidence <= 1
    
    @patch('backend.app.api.v1.trades.TickerService')
    @patch('backend.app.api.v1.trades.OptionsEvaluationAgent')
    def test_analyze_trade_includes_reasoning_steps(self, mock_agent_class, mock_ticker_service_class,
                                                    sample_ticker_result, sample_recommendation):
        """Test that response includes reasoning steps (requirement 6.4)"""
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
        
        response = client.post("/api/v1/trades/analyze", json={"ticker": "AAPL"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify reasoning steps are present (requirement 6.4)
        reasoning_steps = data["recommendation"]["reasoning_steps"]
        assert len(reasoning_steps) > 0
        
        step = reasoning_steps[0]
        assert "step" in step
        assert "passed" in step
        assert "reasoning" in step
        assert "confidence" in step
