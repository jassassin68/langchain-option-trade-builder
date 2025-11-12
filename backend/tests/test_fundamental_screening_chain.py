"""
Unit tests for Fundamental Screening Chain.

Tests chain execution with various fundamental conditions including:
- Companies that pass all criteria
- Small-cap companies (market cap < $1B)
- Overvalued companies (P/E > 50)
- High leverage companies (debt-to-equity > 2.0)
- Companies with earnings within 14 days
- Negative news sentiment
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from backend.app.chains.fundamental_screening_chain import (
    FundamentalScreeningChain,
    FundamentalScreeningResult
)


class TestFundamentalScreeningChain:
    """Test suite for FundamentalScreeningChain"""
    
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
    
    def test_evaluate_criteria_programmatically_all_pass(self, chain):
        """Test programmatic evaluation with all criteria passing"""
        fundamental_data = {
            'market_cap': 50_000_000_000,  # $50B
            'pe_ratio': 25.0,
            'debt_to_equity': 1.2,
            'earnings_date': datetime.now() + timedelta(days=30),
            'news_sentiment': 'positive'
        }
        
        result = chain.evaluate_criteria_programmatically(fundamental_data)
        
        assert result['passed'] is True
        assert result['confidence'] > 0.8
        assert result['criteria_results']['market_cap'] is True
        assert result['criteria_results']['pe_ratio'] is True
        assert result['criteria_results']['debt_to_equity'] is True
        assert result['criteria_results']['earnings_date'] is True
        assert result['criteria_results']['news_sentiment'] is True
    
    def test_evaluate_criteria_small_cap(self, chain):
        """Test rejection of small-cap companies (market cap < $1B)"""
        fundamental_data = {
            'market_cap': 500_000_000,  # $500M
            'pe_ratio': 25.0,
            'debt_to_equity': 1.2,
            'earnings_date': datetime.now() + timedelta(days=30),
            'news_sentiment': 'neutral'
        }
        
        result = chain.evaluate_criteria_programmatically(fundamental_data)
        
        assert result['passed'] is False
        assert result['criteria_results']['market_cap'] is False
        assert any('insufficient size' in reason.lower() for reason in result['reasons'])
    
    def test_evaluate_criteria_high_pe_ratio(self, chain):
        """Test flagging of overvalued companies (P/E > 50)"""
        fundamental_data = {
            'market_cap': 50_000_000_000,
            'pe_ratio': 75.0,
            'debt_to_equity': 1.2,
            'earnings_date': datetime.now() + timedelta(days=30),
            'news_sentiment': 'neutral'
        }
        
        result = chain.evaluate_criteria_programmatically(fundamental_data)
        
        assert result['criteria_results']['pe_ratio'] is False
        assert any('overvalued' in reason.lower() for reason in result['reasons'])
    
    def test_evaluate_criteria_high_debt(self, chain):
        """Test flagging of high leverage companies (debt-to-equity > 2.0)"""
        fundamental_data = {
            'market_cap': 50_000_000_000,
            'pe_ratio': 25.0,
            'debt_to_equity': 3.5,
            'earnings_date': datetime.now() + timedelta(days=30),
            'news_sentiment': 'neutral'
        }
        
        result = chain.evaluate_criteria_programmatically(fundamental_data)
        
        assert result['criteria_results']['debt_to_equity'] is False
        assert any('leverage' in reason.lower() for reason in result['reasons'])
    
    def test_evaluate_criteria_earnings_too_soon(self, chain):
        """Test rejection of stocks with earnings within 14 days"""
        fundamental_data = {
            'market_cap': 50_000_000_000,
            'pe_ratio': 25.0,
            'debt_to_equity': 1.2,
            'earnings_date': datetime.now() + timedelta(days=7),
            'news_sentiment': 'neutral'
        }
        
        result = chain.evaluate_criteria_programmatically(fundamental_data)
        
        assert result['passed'] is False
        assert result['criteria_results']['earnings_date'] is False
        assert any('earnings risk' in reason.lower() for reason in result['reasons'])
    
    def test_evaluate_criteria_negative_sentiment(self, chain):
        """Test flagging of negative news sentiment"""
        fundamental_data = {
            'market_cap': 50_000_000_000,
            'pe_ratio': 25.0,
            'debt_to_equity': 1.2,
            'earnings_date': datetime.now() + timedelta(days=30),
            'news_sentiment': 'negative'
        }
        
        result = chain.evaluate_criteria_programmatically(fundamental_data)
        
        assert result['criteria_results']['news_sentiment'] is False
        assert any('risk' in reason.lower() for reason in result['reasons'])
    
    def test_evaluate_criteria_missing_optional_data(self, chain):
        """Test handling of missing optional data"""
        fundamental_data = {
            'market_cap': 50_000_000_000,
            'earnings_date': datetime.now() + timedelta(days=30)
        }
        
        result = chain.evaluate_criteria_programmatically(fundamental_data)
        
        # Should pass critical criteria (market cap and earnings date)
        assert result['passed'] is True
        assert result['criteria_results']['pe_ratio'] is None
        assert result['criteria_results']['debt_to_equity'] is None
        assert result['criteria_results']['news_sentiment'] is None
    
    def test_evaluate_criteria_edge_case_boundaries(self, chain):
        """Test boundary values for criteria"""
        # Test exact boundary values
        fundamental_data = {
            'market_cap': 1_000_000_000,  # Exactly at minimum
            'pe_ratio': 50.0,  # Exactly at maximum
            'debt_to_equity': 2.0,  # Exactly at maximum
            'earnings_date': datetime.now() + timedelta(days=14),  # Exactly at minimum
            'news_sentiment': 'neutral'
        }
        
        result = chain.evaluate_criteria_programmatically(fundamental_data)
        
        assert result['passed'] is True
        assert result['criteria_results']['market_cap'] is True
        assert result['criteria_results']['pe_ratio'] is True
        assert result['criteria_results']['debt_to_equity'] is True
        assert result['criteria_results']['earnings_date'] is True
    
    def test_evaluate_criteria_earnings_date_string_format(self, chain):
        """Test handling of earnings date as ISO string"""
        future_date = (datetime.now() + timedelta(days=30)).isoformat()
        
        fundamental_data = {
            'market_cap': 50_000_000_000,
            'pe_ratio': 25.0,
            'debt_to_equity': 1.2,
            'earnings_date': future_date,
            'news_sentiment': 'neutral'
        }
        
        result = chain.evaluate_criteria_programmatically(fundamental_data)
        
        assert result['criteria_results']['earnings_date'] is True
    
    def test_evaluate_criteria_no_earnings_date(self, chain):
        """Test handling when earnings date is not available"""
        fundamental_data = {
            'market_cap': 50_000_000_000,
            'pe_ratio': 25.0,
            'debt_to_equity': 1.2,
            'news_sentiment': 'neutral'
        }
        
        result = chain.evaluate_criteria_programmatically(fundamental_data)
        
        # Should still pass if earnings date is not available (defaults to True)
        assert result['passed'] is True
        assert result['criteria_results']['earnings_date'] is None
    
    @pytest.mark.asyncio
    async def test_evaluate_with_valid_data(self, chain, mock_llm):
        """Test async evaluate method with valid data"""
        fundamental_data = {
            'market_cap': 50_000_000_000,
            'pe_ratio': 25.0,
            'debt_to_equity': 1.2,
            'earnings_date': datetime.now() + timedelta(days=30),
            'news_sentiment': 'positive'
        }
        
        # Mock LLM response
        mock_response = """{
            "passed": true,
            "confidence": 0.90,
            "reasoning": "Company shows strong fundamental health with solid market cap and reasonable valuation",
            "criteria_results": {
                "market_cap": true,
                "pe_ratio": true,
                "debt_to_equity": true,
                "earnings_date": true,
                "news_sentiment": true
            },
            "recommendation": "Company is fundamentally sound for options trading"
        }"""
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = mock_response
            
            result = await chain.evaluate('AAPL', fundamental_data)
            
            assert isinstance(result, FundamentalScreeningResult)
            assert result.passed is True
            assert result.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_evaluate_llm_error_handling(self, chain, mock_llm):
        """Test error handling when LLM fails"""
        fundamental_data = {
            'market_cap': 50_000_000_000,
            'pe_ratio': 25.0,
            'debt_to_equity': 1.2,
            'earnings_date': datetime.now() + timedelta(days=30),
            'news_sentiment': 'neutral'
        }
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.side_effect = Exception("LLM API error")
            
            result = await chain.evaluate('AAPL', fundamental_data)
            
            assert result.passed is False
            assert result.confidence == 0.0
            assert 'error' in result.reasoning.lower()
    
    def test_confidence_calculation(self, chain):
        """Test confidence score calculation based on criteria met"""
        # All criteria pass
        fundamental_data_all_pass = {
            'market_cap': 50_000_000_000,
            'pe_ratio': 25.0,
            'debt_to_equity': 1.2,
            'earnings_date': datetime.now() + timedelta(days=30),
            'news_sentiment': 'positive'
        }
        result = chain.evaluate_criteria_programmatically(fundamental_data_all_pass)
        assert result['confidence'] == 1.0
        
        # Some criteria fail
        fundamental_data_partial = {
            'market_cap': 50_000_000_000,
            'pe_ratio': 75.0,  # Fails
            'debt_to_equity': 3.5,  # Fails
            'earnings_date': datetime.now() + timedelta(days=30),
            'news_sentiment': 'negative'  # Fails
        }
        result = chain.evaluate_criteria_programmatically(fundamental_data_partial)
        assert 0.0 < result['confidence'] < 1.0
    
    def test_multiple_failures(self, chain):
        """Test handling of multiple criterion failures"""
        fundamental_data = {
            'market_cap': 500_000_000,  # Fails
            'pe_ratio': 75.0,  # Fails
            'debt_to_equity': 3.5,  # Fails
            'earnings_date': datetime.now() + timedelta(days=7),  # Fails
            'news_sentiment': 'negative'  # Fails
        }
        
        result = chain.evaluate_criteria_programmatically(fundamental_data)
        
        assert result['passed'] is False
        assert result['confidence'] == 0.0
        assert len(result['reasons']) >= 5
