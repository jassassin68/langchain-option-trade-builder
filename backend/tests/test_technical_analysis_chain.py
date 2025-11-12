"""
Unit tests for Technical Analysis Chain.

Tests chain execution with various market conditions including:
- Stocks that pass all criteria
- Penny stocks (price < $10)
- Low liquidity stocks (volume < 500k)
- Oversold/overbought conditions (RSI outside 30-70)
- Low IV rank (< 20)
- Extreme beta values
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from backend.app.chains.technical_analysis_chain import (
    TechnicalAnalysisChain,
    TechnicalAnalysisResult
)


class TestTechnicalAnalysisChain:
    """Test suite for TechnicalAnalysisChain"""
    
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
    
    def test_evaluate_criteria_programmatically_all_pass(self, chain):
        """Test programmatic evaluation with all criteria passing"""
        technical_data = {
            'price': 150.0,
            'volume': 2000000,
            'rsi': 50.0,
            'iv_rank': 45.0,
            'beta': 1.2,
            'ma_50': 145.0,
            'ma_200': 140.0
        }
        
        result = chain.evaluate_criteria_programmatically(technical_data)
        
        assert result['passed'] is True
        assert result['confidence'] > 0.8
        assert result['criteria_results']['price'] is True
        assert result['criteria_results']['volume'] is True
        assert result['criteria_results']['rsi'] is True
        assert result['criteria_results']['iv_rank'] is True
        assert result['criteria_results']['beta'] is True
    
    def test_evaluate_criteria_penny_stock(self, chain):
        """Test rejection of penny stocks (price < $10)"""
        technical_data = {
            'price': 8.50,
            'volume': 2000000,
            'rsi': 50.0,
            'iv_rank': 45.0,
            'beta': 1.2
        }
        
        result = chain.evaluate_criteria_programmatically(technical_data)
        
        assert result['passed'] is False
        assert result['criteria_results']['price'] is False
        assert any('penny stock' in reason.lower() for reason in result['reasons'])
    
    def test_evaluate_criteria_low_volume(self, chain):
        """Test rejection of low liquidity stocks (volume < 500k)"""
        technical_data = {
            'price': 150.0,
            'volume': 300000,
            'rsi': 50.0,
            'iv_rank': 45.0,
            'beta': 1.2
        }
        
        result = chain.evaluate_criteria_programmatically(technical_data)
        
        assert result['passed'] is False
        assert result['criteria_results']['volume'] is False
        assert any('liquidity' in reason.lower() for reason in result['reasons'])
    
    def test_evaluate_criteria_oversold_rsi(self, chain):
        """Test flagging of oversold condition (RSI < 30)"""
        technical_data = {
            'price': 150.0,
            'volume': 2000000,
            'rsi': 25.0,
            'iv_rank': 45.0,
            'beta': 1.2
        }
        
        result = chain.evaluate_criteria_programmatically(technical_data)
        
        assert result['criteria_results']['rsi'] is False
        assert any('oversold' in reason.lower() for reason in result['reasons'])
    
    def test_evaluate_criteria_overbought_rsi(self, chain):
        """Test flagging of overbought condition (RSI > 70)"""
        technical_data = {
            'price': 150.0,
            'volume': 2000000,
            'rsi': 75.0,
            'iv_rank': 45.0,
            'beta': 1.2
        }
        
        result = chain.evaluate_criteria_programmatically(technical_data)
        
        assert result['criteria_results']['rsi'] is False
        assert any('overbought' in reason.lower() for reason in result['reasons'])
    
    def test_evaluate_criteria_low_iv_rank(self, chain):
        """Test rejection of low IV rank (< 20)"""
        technical_data = {
            'price': 150.0,
            'volume': 2000000,
            'rsi': 50.0,
            'iv_rank': 15.0,
            'beta': 1.2
        }
        
        result = chain.evaluate_criteria_programmatically(technical_data)
        
        assert result['passed'] is False
        assert result['criteria_results']['iv_rank'] is False
        assert any('premium' in reason.lower() for reason in result['reasons'])
    
    def test_evaluate_criteria_low_beta(self, chain):
        """Test flagging of low beta (< 0.5)"""
        technical_data = {
            'price': 150.0,
            'volume': 2000000,
            'rsi': 50.0,
            'iv_rank': 45.0,
            'beta': 0.3
        }
        
        result = chain.evaluate_criteria_programmatically(technical_data)
        
        assert result['criteria_results']['beta'] is False
        assert any('volatility' in reason.lower() for reason in result['reasons'])
    
    def test_evaluate_criteria_high_beta(self, chain):
        """Test flagging of high beta (> 2.0)"""
        technical_data = {
            'price': 150.0,
            'volume': 2000000,
            'rsi': 50.0,
            'iv_rank': 45.0,
            'beta': 2.5
        }
        
        result = chain.evaluate_criteria_programmatically(technical_data)
        
        assert result['criteria_results']['beta'] is False
        assert any('volatility' in reason.lower() for reason in result['reasons'])
    
    def test_evaluate_criteria_missing_optional_data(self, chain):
        """Test handling of missing optional data (RSI, IV rank, beta)"""
        technical_data = {
            'price': 150.0,
            'volume': 2000000
        }
        
        result = chain.evaluate_criteria_programmatically(technical_data)
        
        # Should fail because IV rank is critical and missing
        assert result['passed'] is False
        assert result['criteria_results']['rsi'] is None
        assert result['criteria_results']['iv_rank'] is None
        assert result['criteria_results']['beta'] is None
    
    def test_evaluate_criteria_edge_case_boundaries(self, chain):
        """Test boundary values for criteria"""
        # Test exact boundary values
        technical_data = {
            'price': 10.0,  # Exactly at minimum
            'volume': 500000,  # Exactly at minimum
            'rsi': 30.0,  # Exactly at lower bound
            'iv_rank': 20.0,  # Exactly at minimum
            'beta': 0.5  # Exactly at lower bound
        }
        
        result = chain.evaluate_criteria_programmatically(technical_data)
        
        assert result['passed'] is True
        assert result['criteria_results']['price'] is True
        assert result['criteria_results']['volume'] is True
        assert result['criteria_results']['rsi'] is True
        assert result['criteria_results']['iv_rank'] is True
        assert result['criteria_results']['beta'] is True
    
    @pytest.mark.asyncio
    async def test_evaluate_with_valid_data(self, chain, mock_llm):
        """Test async evaluate method with valid data"""
        technical_data = {
            'price': 150.0,
            'volume': 2000000,
            'rsi': 50.0,
            'iv_rank': 45.0,
            'beta': 1.2,
            'ma_50': 145.0,
            'ma_200': 140.0
        }
        
        # Mock LLM response
        mock_response = """{
            "passed": true,
            "confidence": 0.85,
            "reasoning": "Stock meets all technical criteria for options trading",
            "criteria_results": {
                "price": true,
                "volume": true,
                "rsi": true,
                "iv_rank": true,
                "beta": true
            },
            "recommendation": "Stock is suitable for options trading"
        }"""
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = mock_response
            
            result = await chain.evaluate('AAPL', technical_data)
            
            assert isinstance(result, TechnicalAnalysisResult)
            assert result.passed is True
            assert result.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_evaluate_missing_required_data(self, chain):
        """Test evaluate method with missing required data"""
        technical_data = {
            'rsi': 50.0,
            'iv_rank': 45.0
        }
        
        result = await chain.evaluate('AAPL', technical_data)
        
        assert result.passed is False
        assert result.confidence == 0.0
        assert 'error' in result.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_evaluate_llm_error_handling(self, chain, mock_llm):
        """Test error handling when LLM fails"""
        technical_data = {
            'price': 150.0,
            'volume': 2000000,
            'rsi': 50.0,
            'iv_rank': 45.0,
            'beta': 1.2
        }
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.side_effect = Exception("LLM API error")
            
            result = await chain.evaluate('AAPL', technical_data)
            
            assert result.passed is False
            assert result.confidence == 0.0
            assert 'error' in result.reasoning.lower()
    
    def test_confidence_calculation(self, chain):
        """Test confidence score calculation based on criteria met"""
        # All criteria pass
        technical_data_all_pass = {
            'price': 150.0,
            'volume': 2000000,
            'rsi': 50.0,
            'iv_rank': 45.0,
            'beta': 1.2
        }
        result = chain.evaluate_criteria_programmatically(technical_data_all_pass)
        assert result['confidence'] == 1.0
        
        # Some criteria fail
        technical_data_partial = {
            'price': 150.0,
            'volume': 2000000,
            'rsi': 75.0,  # Fails
            'iv_rank': 45.0,
            'beta': 2.5  # Fails
        }
        result = chain.evaluate_criteria_programmatically(technical_data_partial)
        assert 0.0 < result['confidence'] < 1.0
