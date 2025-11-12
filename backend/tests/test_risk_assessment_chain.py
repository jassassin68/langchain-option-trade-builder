"""
Unit tests for Risk Assessment Chain.

Tests risk calculations with various strategy scenarios:
- Cash-secured put calculations
- Iron condor calculations
- Credit spread calculations
- Covered call calculations
- Risk/reward ratio evaluation
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from backend.app.chains.risk_assessment_chain import (
    RiskAssessmentChain,
    RiskAssessmentResult,
    RiskMetrics,
    ContractDetail
)


class TestRiskAssessmentChain:
    """Test suite for RiskAssessmentChain"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing"""
        llm = Mock()
        llm.apredict = AsyncMock()
        return llm
    
    @pytest.fixture
    def chain(self, mock_llm):
        """Create a RiskAssessmentChain instance with mock LLM"""
        return RiskAssessmentChain(llm=mock_llm)
    
    def test_calculate_cash_secured_put_metrics(self, chain):
        """Test risk calculations for cash-secured put strategy"""
        contracts = [{
            'action': 'SELL',
            'type': 'PUT',
            'strike': 145.0,
            'expiration': '2024-12-15',
            'quantity': 1,
            'premium_credit': 3.50
        }]
        current_price = 150.0
        
        metrics = chain.calculate_risk_metrics("Cash-Secured Put", contracts, current_price)
        
        # Max profit = premium * 100
        assert metrics['max_profit'] == 350.0
        
        # Max loss = (strike - premium) * 100
        assert metrics['max_loss'] == 14150.0
        
        # Breakeven = strike - premium
        assert metrics['breakeven'] == 141.5
        
        # Should have positive return on capital
        assert metrics['return_on_capital'] > 0
        
        # Probability of profit should be reasonable
        assert 0 <= metrics['prob_profit'] <= 1
    
    def test_calculate_iron_condor_metrics(self, chain):
        """Test risk calculations for iron condor strategy"""
        contracts = [
            {'action': 'SELL', 'type': 'PUT', 'strike': 140.0, 'quantity': 1, 'premium_credit': 2.0},
            {'action': 'BUY', 'type': 'PUT', 'strike': 135.0, 'quantity': 1, 'premium_credit': -1.0},
            {'action': 'SELL', 'type': 'CALL', 'strike': 160.0, 'quantity': 1, 'premium_credit': 2.0},
            {'action': 'BUY', 'type': 'CALL', 'strike': 165.0, 'quantity': 1, 'premium_credit': -1.0}
        ]
        current_price = 150.0
        
        metrics = chain.calculate_risk_metrics("Iron Condor", contracts, current_price)
        
        # Max profit = net premium
        assert metrics['max_profit'] == 200.0  # (2 - 1 + 2 - 1) * 100
        
        # Should have two breakeven points
        assert isinstance(metrics['breakeven'], list)
        assert len(metrics['breakeven']) == 2
        
        # Probability should be reasonable for iron condor
        assert metrics['prob_profit'] >= 0.50
    
    def test_calculate_credit_spread_metrics(self, chain):
        """Test risk calculations for credit spread strategy"""
        contracts = [
            {'action': 'SELL', 'type': 'PUT', 'strike': 145.0, 'quantity': 1, 'premium_credit': 3.0},
            {'action': 'BUY', 'type': 'PUT', 'strike': 140.0, 'quantity': 1, 'premium_credit': -1.5}
        ]
        current_price = 150.0
        
        metrics = chain.calculate_risk_metrics("Credit Put Spread", contracts, current_price)
        
        # Max profit = net premium
        assert metrics['max_profit'] == 150.0  # (3.0 - 1.5) * 100
        
        # Max loss = spread width - premium
        assert metrics['max_loss'] == 350.0  # (5 - 1.5) * 100
        
        # Return on capital should be calculated
        assert metrics['return_on_capital'] > 0
    
    def test_calculate_covered_call_metrics(self, chain):
        """Test risk calculations for covered call strategy"""
        contracts = [{
            'action': 'SELL',
            'type': 'CALL',
            'strike': 155.0,
            'expiration': '2024-12-15',
            'quantity': 1,
            'premium_credit': 2.50
        }]
        current_price = 150.0
        
        metrics = chain.calculate_risk_metrics("Covered Call", contracts, current_price)
        
        # Max profit includes premium + potential stock appreciation
        assert metrics['max_profit'] > 0
        
        # Breakeven = current price - premium
        assert metrics['breakeven'] == 147.5
        
        # Covered calls typically have high probability of profit
        assert metrics['prob_profit'] >= 0.60
    
    def test_should_trade_favorable_metrics(self, chain):
        """Test should_trade decision with favorable metrics"""
        contracts = [{
            'action': 'SELL',
            'type': 'PUT',
            'strike': 145.0,
            'quantity': 1,
            'premium_credit': 4.0  # Higher premium for better ROC
        }]
        current_price = 150.0
        
        metrics = chain.calculate_risk_metrics("Cash-Secured Put", contracts, current_price)
        
        # With good premium, should recommend trading
        assert metrics['return_on_capital'] >= chain.MIN_RETURN_ON_CAPITAL
    
    def test_should_trade_unfavorable_metrics(self, chain):
        """Test should_trade decision with unfavorable metrics"""
        contracts = [{
            'action': 'SELL',
            'type': 'PUT',
            'strike': 145.0,
            'quantity': 1,
            'premium_credit': 0.50  # Very low premium
        }]
        current_price = 150.0
        
        metrics = chain.calculate_risk_metrics("Cash-Secured Put", contracts, current_price)
        
        # With low premium, ROC will be low
        assert metrics['return_on_capital'] < chain.MIN_RETURN_ON_CAPITAL
        assert metrics['should_trade'] is False
    
    def test_create_contracts_summary(self, chain):
        """Test contracts summary creation"""
        contracts = [
            {
                'action': 'SELL',
                'type': 'PUT',
                'strike': 145.0,
                'expiration': '2024-12-15',
                'quantity': 1,
                'premium_credit': 3.50
            }
        ]
        current_price = 150.0
        
        summary = chain._create_contracts_summary(contracts, current_price)
        
        assert "SELL" in summary
        assert "PUT" in summary
        assert "145.00" in summary
        assert "OTM" in summary  # Put is OTM when strike < current price
    
    def test_create_contracts_summary_itm_contract(self, chain):
        """Test contracts summary with ITM contract"""
        contracts = [
            {
                'action': 'SELL',
                'type': 'PUT',
                'strike': 155.0,  # ITM
                'expiration': '2024-12-15',
                'quantity': 1,
                'premium_credit': 6.00
            }
        ]
        current_price = 150.0
        
        summary = chain._create_contracts_summary(contracts, current_price)
        
        assert "ITM" in summary  # Put is ITM when strike > current price
    
    def test_create_contracts_summary_no_contracts(self, chain):
        """Test contracts summary with no contracts"""
        summary = chain._create_contracts_summary([], 150.0)
        
        assert "No specific contracts" in summary
    
    @pytest.mark.asyncio
    async def test_evaluate_with_valid_context(self, chain, mock_llm):
        """Test async evaluate method with valid context"""
        strategy_context = {
            'strategy_name': 'Cash-Secured Put',
            'contracts': [{
                'action': 'SELL',
                'type': 'PUT',
                'strike': 145.0,
                'expiration': '2024-12-15',
                'quantity': 1,
                'premium_credit': 3.50
            }],
            'current_price': 150.0,
            'technical_outlook': 'Bullish',
            'fundamental_health': 'Healthy',
            'iv_rank': 50.0
        }
        
        # Mock LLM response
        mock_response = """{
            "should_trade": true,
            "confidence": 0.80,
            "reasoning": "Favorable risk/reward with 2.4% return on capital and 65% probability of profit",
            "risk_metrics": {
                "max_profit": 350.0,
                "max_loss": 14150.0,
                "breakeven": 141.5,
                "prob_profit": 0.65,
                "return_on_capital": 2.4
            },
            "contracts": [{
                "action": "SELL",
                "type": "PUT",
                "strike": 145.0,
                "expiration": "2024-12-15",
                "quantity": 1,
                "premium_credit": 3.50
            }],
            "recommendation": "Execute cash-secured put trade"
        }"""
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = mock_response
            
            result = await chain.evaluate('AAPL', strategy_context)
            
            assert isinstance(result, RiskAssessmentResult)
            assert result.should_trade is True
            assert result.confidence > 0.0
            assert isinstance(result.risk_metrics, RiskMetrics)
    
    @pytest.mark.asyncio
    async def test_evaluate_missing_strategy_name(self, chain):
        """Test evaluate method with missing strategy name"""
        strategy_context = {
            'contracts': [],
            'current_price': 150.0
        }
        
        result = await chain.evaluate('AAPL', strategy_context)
        
        assert result.should_trade is False
        assert result.confidence == 0.0
        assert 'error' in result.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_evaluate_llm_error_handling(self, chain, mock_llm):
        """Test error handling when LLM fails"""
        strategy_context = {
            'strategy_name': 'Cash-Secured Put',
            'contracts': [],
            'current_price': 150.0
        }
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.side_effect = Exception("LLM API error")
            
            result = await chain.evaluate('AAPL', strategy_context)
            
            assert result.should_trade is False
            assert result.confidence == 0.0
            assert 'error' in result.reasoning.lower()
    
    def test_return_on_capital_calculation(self, chain):
        """Test ROC calculation accuracy"""
        contracts = [{
            'action': 'SELL',
            'type': 'PUT',
            'strike': 100.0,
            'quantity': 1,
            'premium_credit': 2.0
        }]
        current_price = 105.0
        
        metrics = chain.calculate_risk_metrics("Cash-Secured Put", contracts, current_price)
        
        # ROC = (max_profit / capital_required) * 100
        # max_profit = 200, capital_required = 10000
        # ROC = (200 / 10000) * 100 = 2.0%
        assert abs(metrics['return_on_capital'] - 2.0) < 0.1
    
    def test_probability_of_profit_calculation(self, chain):
        """Test probability of profit calculation"""
        # Contract far OTM should have higher probability
        contracts_otm = [{
            'action': 'SELL',
            'type': 'PUT',
            'strike': 130.0,  # Far OTM
            'quantity': 1,
            'premium_credit': 2.0
        }]
        
        # Contract near ATM should have lower probability
        contracts_atm = [{
            'action': 'SELL',
            'type': 'PUT',
            'strike': 148.0,  # Near ATM
            'quantity': 1,
            'premium_credit': 4.0
        }]
        
        current_price = 150.0
        
        metrics_otm = chain.calculate_risk_metrics("Cash-Secured Put", contracts_otm, current_price)
        metrics_atm = chain.calculate_risk_metrics("Cash-Secured Put", contracts_atm, current_price)
        
        # OTM should have higher probability of profit
        assert metrics_otm['prob_profit'] > metrics_atm['prob_profit']
    
    def test_multiple_contracts_handling(self, chain):
        """Test handling of strategies with multiple contracts"""
        contracts = [
            {'action': 'SELL', 'type': 'PUT', 'strike': 145.0, 'quantity': 2, 'premium_credit': 3.0},
            {'action': 'BUY', 'type': 'PUT', 'strike': 140.0, 'quantity': 2, 'premium_credit': -1.5}
        ]
        current_price = 150.0
        
        metrics = chain.calculate_risk_metrics("Credit Put Spread", contracts, current_price)
        
        # Should handle multiple contracts correctly
        assert metrics['max_profit'] > 0
        assert metrics['max_loss'] > 0
    
    def test_generic_metrics_fallback(self, chain):
        """Test generic metrics calculation for unknown strategy"""
        contracts = [{
            'action': 'SELL',
            'type': 'PUT',
            'strike': 145.0,
            'quantity': 1,
            'premium_credit': 3.0
        }]
        current_price = 150.0
        
        metrics = chain.calculate_risk_metrics("Unknown Strategy", contracts, current_price)
        
        # Should return conservative estimates
        assert metrics['should_trade'] is False
        assert metrics['max_profit'] > 0
        assert metrics['max_loss'] > 0
