"""
Comprehensive LangChain chain tests with mocked LLM responses.

Tests all LangChain chains with various mocked LLM scenarios as required by 7.1, 7.4.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, date, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.chains.technical_analysis_chain import TechnicalAnalysisChain, TechnicalAnalysisResult
from app.chains.fundamental_screening_chain import FundamentalScreeningChain, FundamentalScreeningResult
from app.chains.options_analysis_chain import OptionsAnalysisChain, OptionsAnalysisResult
from app.chains.strategy_selection_chain import StrategySelectionChain, StrategyRecommendation
from app.chains.risk_assessment_chain import RiskAssessmentChain, RiskAssessmentResult


class TestTechnicalAnalysisChainMocking:
    """Test TechnicalAnalysisChain with various LLM responses"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing"""
        llm = Mock()
        llm.apredict = AsyncMock()
        return llm
    
    @pytest.fixture
    def chain(self, mock_llm):
        """Create a TechnicalAnalysisChain instance with mock LLM"""
        return TechnicalAnalysisChain(llm=mock_llm)
    
    @pytest.mark.asyncio
    async def test_successful_llm_response(self, chain, mock_llm):
        """Test chain with successful LLM response"""
        # Mock successful LLM response
        mock_response = {
            "passed": True,
            "confidence": 0.85,
            "reasoning": "Stock meets all technical criteria",
            "criteria_results": {
                "price": True,
                "volume": True,
                "rsi": True,
                "iv_rank": True,
                "beta": True
            },
            "recommendation": "Suitable for options trading"
        }
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = json.dumps(mock_response)
            
            technical_data = {
                'price': 150.0,
                'volume': 2000000,
                'rsi': 50.0,
                'iv_rank': 45.0,
                'beta': 1.2
            }
            
            result = await chain.evaluate('AAPL', technical_data)
            
            assert isinstance(result, TechnicalAnalysisResult)
            assert result.passed is True
            assert result.confidence == 0.85
            assert "technical criteria" in result.reasoning
    
    @pytest.mark.asyncio
    async def test_llm_json_parsing_error(self, chain, mock_llm):
        """Test chain handling of invalid JSON from LLM"""
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = "Invalid JSON response"
            
            technical_data = {
                'price': 150.0,
                'volume': 2000000,
                'rsi': 50.0,
                'iv_rank': 45.0,
                'beta': 1.2
            }
            
            result = await chain.evaluate('AAPL', technical_data)
            
            # Should fallback to programmatic evaluation
            assert isinstance(result, TechnicalAnalysisResult)
            assert result.passed is True  # Based on programmatic criteria
    
    @pytest.mark.asyncio
    async def test_llm_timeout_handling(self, chain, mock_llm):
        """Test chain handling of LLM timeout"""
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.side_effect = asyncio.TimeoutError("LLM timeout")
            
            technical_data = {
                'price': 150.0,
                'volume': 2000000,
                'rsi': 50.0,
                'iv_rank': 45.0,
                'beta': 1.2
            }
            
            result = await chain.evaluate('AAPL', technical_data)
            
            # Should fallback to programmatic evaluation
            assert isinstance(result, TechnicalAnalysisResult)
            assert result.passed is True
    
    @pytest.mark.asyncio
    async def test_llm_rate_limit_handling(self, chain, mock_llm):
        """Test chain handling of LLM rate limiting"""
        from openai import RateLimitError
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.side_effect = RateLimitError("Rate limit exceeded", response=None, body=None)
            
            technical_data = {
                'price': 150.0,
                'volume': 2000000,
                'rsi': 50.0,
                'iv_rank': 45.0,
                'beta': 1.2
            }
            
            result = await chain.evaluate('AAPL', technical_data)
            
            # Should fallback to programmatic evaluation
            assert isinstance(result, TechnicalAnalysisResult)
            assert result.passed is True


class TestFundamentalScreeningChainMocking:
    """Test FundamentalScreeningChain with various LLM responses"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing"""
        llm = Mock()
        llm.apredict = AsyncMock()
        return llm
    
    @pytest.fixture
    def chain(self, mock_llm):
        """Create a FundamentalScreeningChain instance with mock LLM"""
        return FundamentalScreeningChain(llm=mock_llm)
    
    @pytest.mark.asyncio
    async def test_successful_fundamental_analysis(self, chain, mock_llm):
        """Test chain with successful fundamental analysis"""
        mock_response = {
            "passed": True,
            "confidence": 0.90,
            "reasoning": "Strong fundamentals with healthy ratios",
            "criteria_results": {
                "market_cap": True,
                "pe_ratio": True,
                "debt_to_equity": True,
                "earnings_date": True
            },
            "recommendation": "Fundamentally sound for options trading"
        }
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = json.dumps(mock_response)
            
            fundamental_data = {
                'market_cap': 2500000000000,
                'pe_ratio': 25.0,
                'debt_to_equity': 1.5,
                'earnings_date': datetime.now() + timedelta(days=30)
            }
            
            result = await chain.evaluate('AAPL', fundamental_data)
            
            assert isinstance(result, FundamentalScreeningResult)
            assert result.passed is True
            assert result.confidence == 0.90
    
    @pytest.mark.asyncio
    async def test_fundamental_rejection(self, chain, mock_llm):
        """Test chain rejecting based on fundamentals"""
        mock_response = {
            "passed": False,
            "confidence": 0.0,
            "reasoning": "Market cap too small for options trading",
            "criteria_results": {
                "market_cap": False,
                "pe_ratio": True,
                "debt_to_equity": True,
                "earnings_date": True
            },
            "recommendation": "Not suitable due to small market cap"
        }
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = json.dumps(mock_response)
            
            fundamental_data = {
                'market_cap': 500000000,  # Below $1B threshold
                'pe_ratio': 25.0,
                'debt_to_equity': 1.5,
                'earnings_date': datetime.now() + timedelta(days=30)
            }
            
            result = await chain.evaluate('AAPL', fundamental_data)
            
            assert isinstance(result, FundamentalScreeningResult)
            assert result.passed is False
            assert "market cap" in result.reasoning.lower()


class TestOptionsAnalysisChainMocking:
    """Test OptionsAnalysisChain with various LLM responses"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing"""
        llm = Mock()
        llm.apredict = AsyncMock()
        return llm
    
    @pytest.fixture
    def chain(self, mock_llm):
        """Create an OptionsAnalysisChain instance with mock LLM"""
        return OptionsAnalysisChain(llm=mock_llm)
    
    @pytest.fixture
    def sample_options_data(self):
        """Sample options chain data"""
        return {
            'calls': [
                {
                    'strike': 155.0,
                    'expiration': '2024-12-20',
                    'bid': 3.00,
                    'ask': 3.10,
                    'volume': 400,
                    'open_interest': 800,
                    'implied_volatility': 0.28
                }
            ],
            'puts': [
                {
                    'strike': 145.0,
                    'expiration': '2024-12-20',
                    'bid': 2.50,
                    'ask': 2.60,
                    'volume': 500,
                    'open_interest': 1000,
                    'implied_volatility': 0.30
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_quality_options_analysis(self, chain, mock_llm, sample_options_data):
        """Test chain with quality options contracts"""
        mock_response = {
            "passed": True,
            "confidence": 0.80,
            "reasoning": "Quality contracts with good liquidity",
            "quality_contracts_count": 2,
            "best_contracts": [
                {
                    "type": "PUT",
                    "strike": 145.0,
                    "expiration": "2024-12-20",
                    "bid_ask_spread": 0.10,
                    "open_interest": 1000
                }
            ],
            "recommendation": "Suitable options available"
        }
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = json.dumps(mock_response)
            
            result = await chain.evaluate('AAPL', sample_options_data)
            
            assert isinstance(result, OptionsAnalysisResult)
            assert result.passed is True
            assert result.quality_contracts_count == 2
    
    @pytest.mark.asyncio
    async def test_poor_liquidity_options(self, chain, mock_llm):
        """Test chain with poor liquidity options"""
        poor_options_data = {
            'calls': [
                {
                    'strike': 155.0,
                    'expiration': '2024-12-20',
                    'bid': 3.00,
                    'ask': 3.50,  # Wide spread
                    'volume': 10,  # Low volume
                    'open_interest': 50,  # Low OI
                    'implied_volatility': 0.28
                }
            ],
            'puts': []
        }
        
        mock_response = {
            "passed": False,
            "confidence": 0.0,
            "reasoning": "Poor liquidity with wide spreads",
            "quality_contracts_count": 0,
            "best_contracts": [],
            "recommendation": "No suitable options available"
        }
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = json.dumps(mock_response)
            
            result = await chain.evaluate('AAPL', poor_options_data)
            
            assert isinstance(result, OptionsAnalysisResult)
            assert result.passed is False
            assert result.quality_contracts_count == 0


class TestStrategySelectionChainMocking:
    """Test StrategySelectionChain with various LLM responses"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing"""
        llm = Mock()
        llm.apredict = AsyncMock()
        return llm
    
    @pytest.fixture
    def chain(self, mock_llm):
        """Create a StrategySelectionChain instance with mock LLM"""
        return StrategySelectionChain(llm=mock_llm)
    
    @pytest.mark.asyncio
    async def test_cash_secured_put_strategy(self, chain, mock_llm):
        """Test chain recommending cash-secured put strategy"""
        mock_response = {
            "strategy_name": "Cash-Secured Put",
            "passed": True,
            "confidence": 0.85,
            "reasoning": "Bullish conditions with high IV favor selling puts",
            "contract_recommendations": [
                {
                    "action": "SELL",
                    "type": "PUT",
                    "strike": 145.0,
                    "expiration": "2024-12-20",
                    "quantity": 1
                }
            ],
            "recommendation": "Execute cash-secured put strategy"
        }
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = json.dumps(mock_response)
            
            analysis_context = {
                'technical_bullish': True,
                'iv_rank': 45.0,
                'support_level': 140.0
            }
            
            result = await chain.evaluate('AAPL', analysis_context)
            
            assert isinstance(result, StrategyRecommendation)
            assert result.strategy_name == "Cash-Secured Put"
            assert result.passed is True
            assert len(result.contract_recommendations) == 1
    
    @pytest.mark.asyncio
    async def test_iron_condor_strategy(self, chain, mock_llm):
        """Test chain recommending iron condor strategy"""
        mock_response = {
            "strategy_name": "Iron Condor",
            "passed": True,
            "confidence": 0.75,
            "reasoning": "Neutral conditions with moderate IV favor range-bound strategy",
            "contract_recommendations": [
                {
                    "action": "SELL",
                    "type": "PUT",
                    "strike": 145.0,
                    "expiration": "2024-12-20",
                    "quantity": 1
                },
                {
                    "action": "BUY",
                    "type": "PUT",
                    "strike": 140.0,
                    "expiration": "2024-12-20",
                    "quantity": 1
                }
            ],
            "recommendation": "Execute iron condor strategy"
        }
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = json.dumps(mock_response)
            
            analysis_context = {
                'technical_neutral': True,
                'iv_rank': 35.0,
                'expected_range': [140.0, 160.0]
            }
            
            result = await chain.evaluate('AAPL', analysis_context)
            
            assert isinstance(result, StrategyRecommendation)
            assert result.strategy_name == "Iron Condor"
            assert len(result.contract_recommendations) == 2
    
    @pytest.mark.asyncio
    async def test_no_suitable_strategy(self, chain, mock_llm):
        """Test chain finding no suitable strategy"""
        mock_response = {
            "strategy_name": None,
            "passed": False,
            "confidence": 0.0,
            "reasoning": "Market conditions do not favor any options strategy",
            "contract_recommendations": [],
            "recommendation": "Avoid options trading at this time"
        }
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = json.dumps(mock_response)
            
            analysis_context = {
                'technical_uncertain': True,
                'iv_rank': 15.0,  # Too low
                'earnings_soon': True
            }
            
            result = await chain.evaluate('AAPL', analysis_context)
            
            assert isinstance(result, StrategyRecommendation)
            assert result.strategy_name is None
            assert result.passed is False


class TestRiskAssessmentChainMocking:
    """Test RiskAssessmentChain with various LLM responses"""
    
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
    
    @pytest.mark.asyncio
    async def test_favorable_risk_assessment(self, chain, mock_llm):
        """Test chain with favorable risk/reward assessment"""
        mock_response = {
            "should_trade": True,
            "confidence": 0.80,
            "reasoning": "Favorable risk/reward ratio with limited downside",
            "risk_metrics": {
                "max_profit": 350.0,
                "max_loss": 14150.0,
                "breakeven": 141.5,
                "prob_profit": 0.65,
                "return_on_capital": 2.4
            },
            "contracts": [
                {
                    "action": "SELL",
                    "type": "PUT",
                    "strike": 145.0,
                    "expiration": "2024-12-15",
                    "quantity": 1,
                    "premium_credit": 3.50
                }
            ],
            "recommendation": "Execute trade with defined risk"
        }
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = json.dumps(mock_response)
            
            strategy_context = {
                'strategy': 'Cash-Secured Put',
                'contracts': [{'strike': 145.0, 'premium': 3.50}],
                'current_price': 150.0
            }
            
            result = await chain.evaluate('AAPL', strategy_context)
            
            assert isinstance(result, RiskAssessmentResult)
            assert result.should_trade is True
            assert result.risk_metrics.max_profit == 350.0
            assert result.risk_metrics.prob_profit == 0.65
    
    @pytest.mark.asyncio
    async def test_unfavorable_risk_assessment(self, chain, mock_llm):
        """Test chain with unfavorable risk/reward assessment"""
        mock_response = {
            "should_trade": False,
            "confidence": 0.90,
            "reasoning": "Risk/reward ratio is unfavorable with high downside risk",
            "risk_metrics": {
                "max_profit": 100.0,
                "max_loss": 10000.0,
                "breakeven": 149.0,
                "prob_profit": 0.30,
                "return_on_capital": 1.0
            },
            "contracts": [],
            "recommendation": "Avoid trade due to poor risk/reward"
        }
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = json.dumps(mock_response)
            
            strategy_context = {
                'strategy': 'Cash-Secured Put',
                'contracts': [{'strike': 149.0, 'premium': 1.00}],
                'current_price': 150.0
            }
            
            result = await chain.evaluate('AAPL', strategy_context)
            
            assert isinstance(result, RiskAssessmentResult)
            assert result.should_trade is False
            assert result.risk_metrics.prob_profit == 0.30
            assert "unfavorable" in result.reasoning.lower()


class TestChainErrorHandling:
    """Test error handling across all chains"""
    
    @pytest.mark.asyncio
    async def test_llm_service_unavailable(self):
        """Test all chains handle LLM service unavailability"""
        from openai import APIError
        
        mock_llm = Mock()
        mock_llm.apredict = AsyncMock(side_effect=APIError("Service unavailable"))
        
        chains = [
            TechnicalAnalysisChain(llm=mock_llm),
            FundamentalScreeningChain(llm=mock_llm),
            OptionsAnalysisChain(llm=mock_llm),
            StrategySelectionChain(llm=mock_llm),
            RiskAssessmentChain(llm=mock_llm)
        ]
        
        test_data = {
            'price': 150.0,
            'volume': 2000000,
            'rsi': 50.0
        }
        
        for chain in chains:
            with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
                mock_arun.side_effect = APIError("Service unavailable")
                
                result = await chain.evaluate('AAPL', test_data)
                
                # All chains should handle errors gracefully
                assert result is not None
                # Most should fallback to programmatic evaluation or return error state
    
    @pytest.mark.asyncio
    async def test_malformed_llm_responses(self):
        """Test chains handle malformed LLM responses"""
        mock_llm = Mock()
        
        malformed_responses = [
            "Not JSON at all",
            '{"incomplete": json',
            '{"missing_required_fields": true}',
            '[]',  # Array instead of object
            'null'
        ]
        
        chain = TechnicalAnalysisChain(llm=mock_llm)
        test_data = {'price': 150.0, 'volume': 2000000}
        
        for malformed_response in malformed_responses:
            with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
                mock_arun.return_value = malformed_response
                
                result = await chain.evaluate('AAPL', test_data)
                
                # Should handle malformed responses gracefully
                assert isinstance(result, TechnicalAnalysisResult)
                # Should fallback to programmatic evaluation
                assert result.passed is True  # Based on good test data