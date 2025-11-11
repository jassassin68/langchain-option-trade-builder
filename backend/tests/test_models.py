import pytest
import sys
import os
from datetime import date, datetime, timedelta
from pydantic import ValidationError

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.models.api import (
    TickerResult,
    TickerSearchResponse,
    TradeAnalysisRequest,
    Contract,
    RiskMetrics,
    ReasoningStep,
    TradeRecommendation,
    TradeAnalysisResponse,
    HealthResponse,
    ErrorResponse,
    ActionType,
    ContractType
)

class TestTickerResult:
    """Test TickerResult model validation"""
    
    def test_valid_ticker_result(self):
        """Test valid ticker result creation"""
        ticker = TickerResult(
            ticker="AAPL",
            company_name="Apple Inc.",
            exchange="NASDAQ"
        )
        assert ticker.ticker == "AAPL"
        assert ticker.company_name == "Apple Inc."
        assert ticker.exchange == "NASDAQ"
    
    def test_invalid_ticker_format(self):
        """Test invalid ticker format validation"""
        with pytest.raises(ValidationError):
            TickerResult(ticker="TOOLONG", company_name="Test Company")
        
        with pytest.raises(ValidationError):
            TickerResult(ticker="123", company_name="Test Company")
        
        with pytest.raises(ValidationError):
            TickerResult(ticker="", company_name="Test Company")
    
    def test_default_exchange(self):
        """Test default exchange value"""
        ticker = TickerResult(ticker="AAPL", company_name="Apple Inc.")
        assert ticker.exchange == "NYSE"

class TestTickerSearchResponse:
    """Test TickerSearchResponse model validation"""
    
    def test_valid_search_response(self):
        """Test valid search response creation"""
        results = [
            TickerResult(ticker="AAPL", company_name="Apple Inc."),
            TickerResult(ticker="MSFT", company_name="Microsoft Corp.")
        ]
        response = TickerSearchResponse(results=results, count=2)
        assert len(response.results) == 2
        assert response.count == 2
    
    def test_empty_results(self):
        """Test empty search results"""
        response = TickerSearchResponse(results=[], count=0)
        assert len(response.results) == 0
        assert response.count == 0

class TestTradeAnalysisRequest:
    """Test TradeAnalysisRequest model validation"""
    
    def test_valid_request(self):
        """Test valid analysis request"""
        request = TradeAnalysisRequest(ticker="AAPL")
        assert request.ticker == "AAPL"
    
    def test_ticker_normalization(self):
        """Test ticker normalization (uppercase, strip)"""
        request = TradeAnalysisRequest(ticker=" aapl ")
        assert request.ticker == "AAPL"
    
    def test_invalid_ticker_format(self):
        """Test invalid ticker format"""
        with pytest.raises(ValidationError):
            TradeAnalysisRequest(ticker="TOOLONG")
        
        with pytest.raises(ValidationError):
            TradeAnalysisRequest(ticker="123")

class TestContract:
    """Test Contract model validation"""
    
    def test_valid_contract(self):
        """Test valid contract creation"""
        future_date = date.today() + timedelta(days=30)
        contract = Contract(
            action=ActionType.BUY,
            type=ContractType.CALL,
            strike=150.0,
            expiration=future_date,
            quantity=1,
            premium_credit=5.50
        )
        assert contract.action == ActionType.BUY
        assert contract.type == ContractType.CALL
        assert contract.strike == 150.0
        assert contract.quantity == 1
        assert contract.premium_credit == 5.50
    
    def test_invalid_strike_price(self):
        """Test invalid strike price validation"""
        future_date = date.today() + timedelta(days=30)
        with pytest.raises(ValidationError):
            Contract(
                action=ActionType.BUY,
                type=ContractType.CALL,
                strike=-10.0,  # Invalid negative strike
                expiration=future_date,
                quantity=1
            )
    
    def test_invalid_quantity(self):
        """Test invalid quantity validation"""
        future_date = date.today() + timedelta(days=30)
        with pytest.raises(ValidationError):
            Contract(
                action=ActionType.BUY,
                type=ContractType.CALL,
                strike=150.0,
                expiration=future_date,
                quantity=0  # Invalid zero quantity
            )
    
    def test_past_expiration_date(self):
        """Test past expiration date validation"""
        past_date = date.today() - timedelta(days=1)
        with pytest.raises(ValidationError):
            Contract(
                action=ActionType.BUY,
                type=ContractType.CALL,
                strike=150.0,
                expiration=past_date,  # Invalid past date
                quantity=1
            )

class TestRiskMetrics:
    """Test RiskMetrics model validation"""
    
    def test_valid_risk_metrics(self):
        """Test valid risk metrics creation"""
        metrics = RiskMetrics(
            max_profit=1000.0,
            max_loss=-500.0,
            breakeven=150.0,
            prob_profit=0.65,
            return_on_capital=20.0
        )
        assert metrics.max_profit == 1000.0
        assert metrics.max_loss == -500.0
        assert metrics.breakeven == 150.0
        assert metrics.prob_profit == 0.65
        assert metrics.return_on_capital == 20.0
    
    def test_multiple_breakeven_points(self):
        """Test multiple breakeven points"""
        metrics = RiskMetrics(
            max_profit=500.0,
            max_loss=-200.0,
            breakeven=[145.0, 155.0],  # Iron condor scenario
            prob_profit=0.70,
            return_on_capital=15.0
        )
        assert isinstance(metrics.breakeven, list)
        assert len(metrics.breakeven) == 2
    
    def test_invalid_probability(self):
        """Test invalid probability validation"""
        with pytest.raises(ValidationError):
            RiskMetrics(
                max_profit=1000.0,
                max_loss=-500.0,
                breakeven=150.0,
                prob_profit=1.5,  # Invalid > 1
                return_on_capital=20.0
            )
        
        with pytest.raises(ValidationError):
            RiskMetrics(
                max_profit=1000.0,
                max_loss=-500.0,
                breakeven=150.0,
                prob_profit=-0.1,  # Invalid < 0
                return_on_capital=20.0
            )

class TestReasoningStep:
    """Test ReasoningStep model validation"""
    
    def test_valid_reasoning_step(self):
        """Test valid reasoning step creation"""
        step = ReasoningStep(
            step="Technical Analysis",
            passed=True,
            reasoning="Stock meets all technical criteria with RSI at 45",
            confidence=0.85
        )
        assert step.step == "Technical Analysis"
        assert step.passed is True
        assert step.confidence == 0.85
    
    def test_invalid_confidence(self):
        """Test invalid confidence validation"""
        with pytest.raises(ValidationError):
            ReasoningStep(
                step="Test Step",
                passed=True,
                reasoning="Test reasoning",
                confidence=1.5  # Invalid > 1
            )

class TestTradeRecommendation:
    """Test TradeRecommendation model validation"""
    
    def test_valid_recommendation(self):
        """Test valid trade recommendation"""
        future_date = date.today() + timedelta(days=30)
        contract = Contract(
            action=ActionType.BUY,
            type=ContractType.PUT,
            strike=145.0,
            expiration=future_date,
            quantity=1
        )
        
        risk_metrics = RiskMetrics(
            max_profit=500.0,
            max_loss=-1000.0,
            breakeven=140.0,
            prob_profit=0.70,
            return_on_capital=50.0
        )
        
        reasoning = ReasoningStep(
            step="Final Assessment",
            passed=True,
            reasoning="All criteria met",
            confidence=0.80
        )
        
        recommendation = TradeRecommendation(
            should_trade=True,
            confidence=0.80,
            strategy="Cash-Secured Put",
            contracts=[contract],
            risk_metrics=risk_metrics,
            reasoning_steps=[reasoning]
        )
        
        assert recommendation.should_trade is True
        assert recommendation.confidence == 0.80
        assert recommendation.strategy == "Cash-Secured Put"
        assert len(recommendation.contracts) == 1
        assert recommendation.risk_metrics is not None
        assert len(recommendation.reasoning_steps) == 1
    
    def test_no_trade_recommendation(self):
        """Test recommendation against trading"""
        recommendation = TradeRecommendation(
            should_trade=False,
            confidence=0.90,
            strategy=None,
            contracts=[],
            risk_metrics=None,
            reasoning_steps=[]
        )
        
        assert recommendation.should_trade is False
        assert recommendation.strategy is None
        assert len(recommendation.contracts) == 0
        assert recommendation.risk_metrics is None

class TestTradeAnalysisResponse:
    """Test TradeAnalysisResponse model validation"""
    
    def test_valid_analysis_response(self):
        """Test valid analysis response"""
        recommendation = TradeRecommendation(
            should_trade=False,
            confidence=0.60,
        )
        
        response = TradeAnalysisResponse(
            ticker="AAPL",
            company_name="Apple Inc.",
            recommendation=recommendation
        )
        
        assert response.ticker == "AAPL"
        assert response.company_name == "Apple Inc."
        assert isinstance(response.analysis_timestamp, datetime)

class TestHealthResponse:
    """Test HealthResponse model validation"""
    
    def test_valid_health_response(self):
        """Test valid health response"""
        response = HealthResponse(
            services={
                "database": "healthy",
                "redis": "healthy",
                "openai": "healthy"
            }
        )
        
        assert response.status == "healthy"
        assert isinstance(response.timestamp, datetime)
        assert "database" in response.services

class TestErrorResponse:
    """Test ErrorResponse model validation"""
    
    def test_valid_error_response(self):
        """Test valid error response"""
        error = ErrorResponse(
            error="ValidationError",
            message="Invalid ticker format",
            details={"field": "ticker", "value": "INVALID"},
            retry_after=60
        )
        
        assert error.error == "ValidationError"
        assert error.message == "Invalid ticker format"
        assert error.details["field"] == "ticker"
        assert error.retry_after == 60
    
    def test_minimal_error_response(self):
        """Test minimal error response"""
        error = ErrorResponse(
            error="NotFound",
            message="Ticker not found"
        )
        
        assert error.error == "NotFound"
        assert error.message == "Ticker not found"
        assert error.details is None
        assert error.retry_after is None