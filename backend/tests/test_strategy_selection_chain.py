"""
Unit tests for Strategy Selection Chain.

Tests strategy selection with different market condition combinations:
- Cash-secured put conditions (bullish + healthy + high IV)
- Iron condor conditions (neutral + moderate IV)
- Credit put spread conditions (support + high IV)
- Covered call conditions
- No suitable strategy conditions
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import date, timedelta
from backend.app.chains.strategy_selection_chain import (
    StrategySelectionChain,
    StrategyRecommendation
)


class TestStrategySelectionChain:
    """Test suite for StrategySelectionChain"""
    
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
    
    @pytest.fixture
    def quality_contracts(self):
        """Create sample quality contracts"""
        return [
            {
                'type': 'PUT',
                'strike': 145.0,
                'expiration': date.today() + timedelta(days=45),
                'bid': 2.50,
                'ask': 2.60
            },
            {
                'type': 'CALL',
                'strike': 155.0,
                'expiration': date.today() + timedelta(days=45),
                'bid': 3.00,
                'ask': 3.10
            }
        ]
    
    def test_determine_technical_outlook_bullish(self, chain):
        """Test bullish technical outlook determination"""
        technical_data = {
            'price': 150.0,
            'ma_50': 145.0,  # Price above
            'ma_200': 140.0,  # Price above
            'rsi': 60.0  # Above 50
        }
        
        outlook = chain._determine_technical_outlook(technical_data)
        assert outlook == "Bullish"
    
    def test_determine_technical_outlook_bearish(self, chain):
        """Test bearish technical outlook determination"""
        technical_data = {
            'price': 140.0,
            'ma_50': 145.0,  # Price below
            'ma_200': 150.0,  # Price below
            'rsi': 40.0  # Below 50
        }
        
        outlook = chain._determine_technical_outlook(technical_data)
        assert outlook == "Bearish"
    
    def test_determine_technical_outlook_neutral(self, chain):
        """Test neutral technical outlook determination"""
        technical_data = {
            'price': 150.0,
            'ma_50': 145.0,  # Price above (bullish)
            'ma_200': 155.0,  # Price below (bearish)
            'rsi': 50.0  # Neutral
        }
        
        outlook = chain._determine_technical_outlook(technical_data)
        assert outlook == "Neutral"
    
    def test_determine_fundamental_health_healthy(self, chain):
        """Test healthy fundamental determination"""
        fundamental_data = {
            'market_cap': 50_000_000_000,  # > $1B
            'pe_ratio': 25.0,  # < 50
            'debt_to_equity': 1.2  # < 2.0
        }
        
        health = chain._determine_fundamental_health(fundamental_data)
        assert health == "Healthy"
    
    def test_determine_fundamental_health_weak(self, chain):
        """Test weak fundamental determination"""
        fundamental_data = {
            'market_cap': 500_000_000,  # < $1B
            'pe_ratio': 75.0,  # > 50
            'debt_to_equity': 3.5  # > 2.0
        }
        
        health = chain._determine_fundamental_health(fundamental_data)
        assert health == "Weak"
    
    def test_determine_fundamental_health_moderate(self, chain):
        """Test moderate fundamental determination"""
        fundamental_data = {
            'market_cap': 50_000_000_000,  # Healthy
            'pe_ratio': 75.0,  # Weak
            'debt_to_equity': 1.2  # Healthy
        }
        
        health = chain._determine_fundamental_health(fundamental_data)
        assert health == "Moderate"
    
    def test_select_strategy_cash_secured_put(self, chain, quality_contracts):
        """Test cash-secured put strategy selection (bullish + healthy + high IV)"""
        analysis_context = {
            'technical_data': {
                'price': 150.0,
                'ma_50': 145.0,
                'ma_200': 140.0,
                'rsi': 60.0,
                'iv_rank': 50.0  # High IV
            },
            'fundamental_data': {
                'market_cap': 50_000_000_000,
                'pe_ratio': 25.0,
                'debt_to_equity': 1.2
            },
            'quality_contracts': quality_contracts
        }
        
        result = chain.select_strategy_programmatically(analysis_context)
        
        assert result['strategy_name'] == chain.CASH_SECURED_PUT
        assert result['passed'] is True
        assert result['confidence'] > 0.8
        assert 'bullish' in result['reasoning'].lower()
    
    def test_select_strategy_iron_condor(self, chain, quality_contracts):
        """Test iron condor strategy selection (neutral + moderate IV)"""
        analysis_context = {
            'technical_data': {
                'price': 150.0,
                'ma_50': 145.0,
                'ma_200': 155.0,
                'rsi': 50.0,
                'iv_rank': 30.0  # Moderate IV
            },
            'fundamental_data': {
                'market_cap': 50_000_000_000,
                'pe_ratio': 25.0,
                'debt_to_equity': 1.2
            },
            'quality_contracts': quality_contracts
        }
        
        result = chain.select_strategy_programmatically(analysis_context)
        
        assert result['strategy_name'] == chain.IRON_CONDOR
        assert result['passed'] is True
        assert result['confidence'] > 0.7
        assert 'neutral' in result['reasoning'].lower()
    
    def test_select_strategy_credit_put_spread(self, chain, quality_contracts):
        """Test credit put spread strategy selection (support + high IV)"""
        analysis_context = {
            'technical_data': {
                'price': 150.0,
                'ma_50': 145.0,  # Price above MA (support)
                'ma_200': 140.0,
                'rsi': 45.0,
                'iv_rank': 50.0  # High IV
            },
            'fundamental_data': {
                'market_cap': 500_000_000,  # Weak fundamentals
                'pe_ratio': 75.0,
                'debt_to_equity': 3.5
            },
            'quality_contracts': quality_contracts
        }
        
        result = chain.select_strategy_programmatically(analysis_context)
        
        assert result['strategy_name'] == chain.CREDIT_PUT_SPREAD
        assert result['passed'] is True
        assert result['confidence'] > 0.7
        assert 'support' in result['reasoning'].lower() or 'spread' in result['reasoning'].lower()
    
    def test_select_strategy_covered_call(self, chain, quality_contracts):
        """Test covered call strategy selection"""
        analysis_context = {
            'technical_data': {
                'price': 150.0,
                'ma_50': 145.0,
                'ma_200': 155.0,
                'rsi': 50.0,
                'iv_rank': 25.0  # Moderate IV
            },
            'fundamental_data': {
                'market_cap': 50_000_000_000,
                'pe_ratio': 25.0,
                'debt_to_equity': 1.2
            },
            'quality_contracts': quality_contracts
        }
        
        result = chain.select_strategy_programmatically(analysis_context)
        
        # Should select covered call or iron condor
        assert result['strategy_name'] in [chain.COVERED_CALL, chain.IRON_CONDOR]
        assert result['passed'] is True
    
    def test_select_strategy_no_suitable_strategy(self, chain, quality_contracts):
        """Test no suitable strategy recommendation"""
        analysis_context = {
            'technical_data': {
                'price': 140.0,
                'ma_50': 145.0,  # Bearish
                'ma_200': 150.0,
                'rsi': 40.0,
                'iv_rank': 15.0  # Low IV
            },
            'fundamental_data': {
                'market_cap': 500_000_000,  # Weak
                'pe_ratio': 75.0,
                'debt_to_equity': 3.5
            },
            'quality_contracts': quality_contracts
        }
        
        result = chain.select_strategy_programmatically(analysis_context)
        
        assert result['strategy_name'] is None
        assert result['passed'] is False
        assert result['confidence'] == 0.0
    
    def test_create_contracts_summary_with_contracts(self, chain, quality_contracts):
        """Test contracts summary creation"""
        summary = chain._create_contracts_summary(quality_contracts)
        
        assert "Total: 2 contracts" in summary
        assert "1 calls, 1 puts" in summary
        assert "Sample PUT" in summary
        assert "Sample CALL" in summary
    
    def test_create_contracts_summary_no_contracts(self, chain):
        """Test contracts summary with no contracts"""
        summary = chain._create_contracts_summary([])
        
        assert "No quality contracts available" in summary
    
    @pytest.mark.asyncio
    async def test_evaluate_with_valid_context(self, chain, mock_llm, quality_contracts):
        """Test async evaluate method with valid context"""
        analysis_context = {
            'technical_data': {
                'price': 150.0,
                'ma_50': 145.0,
                'ma_200': 140.0,
                'rsi': 60.0,
                'iv_rank': 50.0
            },
            'fundamental_data': {
                'market_cap': 50_000_000_000,
                'pe_ratio': 25.0,
                'debt_to_equity': 1.2
            },
            'quality_contracts': quality_contracts
        }
        
        # Mock LLM response
        mock_response = """{
            "strategy_name": "Cash-Secured Put",
            "passed": true,
            "confidence": 0.85,
            "reasoning": "Bullish technical outlook with healthy fundamentals and high IV rank supports cash-secured put strategy",
            "contract_recommendations": [],
            "recommendation": "Execute cash-secured put strategy"
        }"""
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = mock_response
            
            result = await chain.evaluate('AAPL', analysis_context)
            
            assert isinstance(result, StrategyRecommendation)
            assert result.passed is True
            assert result.strategy_name == "Cash-Secured Put"
    
    @pytest.mark.asyncio
    async def test_evaluate_llm_error_handling(self, chain, mock_llm, quality_contracts):
        """Test error handling when LLM fails"""
        analysis_context = {
            'technical_data': {'price': 150.0},
            'fundamental_data': {},
            'quality_contracts': quality_contracts
        }
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.side_effect = Exception("LLM API error")
            
            result = await chain.evaluate('AAPL', analysis_context)
            
            assert result.passed is False
            assert result.confidence == 0.0
            assert 'error' in result.reasoning.lower()
    
    def test_strategy_priority_cash_secured_put_over_others(self, chain, quality_contracts):
        """Test that cash-secured put is prioritized when conditions are met"""
        # Conditions that could match multiple strategies
        analysis_context = {
            'technical_data': {
                'price': 150.0,
                'ma_50': 145.0,
                'ma_200': 140.0,
                'rsi': 60.0,
                'iv_rank': 50.0  # High IV
            },
            'fundamental_data': {
                'market_cap': 50_000_000_000,
                'pe_ratio': 25.0,
                'debt_to_equity': 1.2
            },
            'quality_contracts': quality_contracts
        }
        
        result = chain.select_strategy_programmatically(analysis_context)
        
        # Should select cash-secured put as it's checked first
        assert result['strategy_name'] == chain.CASH_SECURED_PUT
    
    def test_missing_technical_data(self, chain, quality_contracts):
        """Test handling of missing technical data"""
        analysis_context = {
            'technical_data': {},
            'fundamental_data': {
                'market_cap': 50_000_000_000,
                'pe_ratio': 25.0,
                'debt_to_equity': 1.2
            },
            'quality_contracts': quality_contracts
        }
        
        result = chain.select_strategy_programmatically(analysis_context)
        
        # Should not crash, but likely won't find a strategy
        assert result['strategy_name'] is None or isinstance(result['strategy_name'], str)
    
    def test_edge_case_iv_rank_boundaries(self, chain, quality_contracts):
        """Test IV rank at exact boundaries"""
        # Test at IV rank = 40 (boundary between moderate and high)
        analysis_context = {
            'technical_data': {
                'price': 150.0,
                'ma_50': 145.0,
                'ma_200': 140.0,
                'rsi': 60.0,
                'iv_rank': 40.0  # Exactly at boundary
            },
            'fundamental_data': {
                'market_cap': 50_000_000_000,
                'pe_ratio': 25.0,
                'debt_to_equity': 1.2
            },
            'quality_contracts': quality_contracts
        }
        
        result = chain.select_strategy_programmatically(analysis_context)
        
        # Should select iron condor (20-40 range) not cash-secured put (>40)
        assert result['strategy_name'] == chain.IRON_CONDOR
