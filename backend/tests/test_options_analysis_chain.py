"""
Unit tests for Options Analysis Chain.

Tests chain execution with various contract scenarios including:
- High quality contracts
- Low open interest contracts
- Wide bid-ask spreads
- Contracts outside expiration window
- Mixed quality contracts
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import date, timedelta
from backend.app.chains.options_analysis_chain import (
    OptionsAnalysisChain,
    OptionsAnalysisResult
)


class TestOptionsAnalysisChain:
    """Test suite for OptionsAnalysisChain"""
    
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
    def quality_contract(self):
        """Create a high-quality contract for testing"""
        return {
            'type': 'PUT',
            'strike': 150.0,
            'expiration': date.today() + timedelta(days=45),
            'days_to_expiration': 45,
            'bid': 2.50,
            'ask': 2.60,
            'open_interest': 500,
            'volume': 100
        }
    
    @pytest.fixture
    def low_oi_contract(self):
        """Create a contract with low open interest"""
        return {
            'type': 'CALL',
            'strike': 155.0,
            'expiration': date.today() + timedelta(days=45),
            'days_to_expiration': 45,
            'bid': 3.00,
            'ask': 3.10,
            'open_interest': 50,  # Below minimum
            'volume': 10
        }
    
    @pytest.fixture
    def wide_spread_contract(self):
        """Create a contract with wide bid-ask spread"""
        return {
            'type': 'PUT',
            'strike': 145.0,
            'expiration': date.today() + timedelta(days=45),
            'days_to_expiration': 45,
            'bid': 2.00,
            'ask': 2.50,  # 22% spread
            'open_interest': 200,
            'volume': 50
        }
    
    @pytest.fixture
    def short_expiration_contract(self):
        """Create a contract with expiration too soon"""
        return {
            'type': 'CALL',
            'strike': 150.0,
            'expiration': date.today() + timedelta(days=20),
            'days_to_expiration': 20,  # Below minimum
            'bid': 2.80,
            'ask': 2.90,
            'open_interest': 300,
            'volume': 75
        }
    
    @pytest.fixture
    def long_expiration_contract(self):
        """Create a contract with expiration too far"""
        return {
            'type': 'PUT',
            'strike': 150.0,
            'expiration': date.today() + timedelta(days=90),
            'days_to_expiration': 90,  # Above maximum
            'bid': 4.00,
            'ask': 4.10,
            'open_interest': 400,
            'volume': 80
        }
    
    def test_meets_quality_criteria_valid_contract(self, chain, quality_contract):
        """Test that a quality contract passes all criteria"""
        assert chain._meets_quality_criteria(quality_contract) is True
    
    def test_meets_quality_criteria_low_open_interest(self, chain, low_oi_contract):
        """Test rejection of contracts with low open interest"""
        assert chain._meets_quality_criteria(low_oi_contract) is False
    
    def test_meets_quality_criteria_wide_spread(self, chain, wide_spread_contract):
        """Test rejection of contracts with wide bid-ask spread"""
        assert chain._meets_quality_criteria(wide_spread_contract) is False
    
    def test_meets_quality_criteria_short_expiration(self, chain, short_expiration_contract):
        """Test rejection of contracts with expiration too soon"""
        assert chain._meets_quality_criteria(short_expiration_contract) is False
    
    def test_meets_quality_criteria_long_expiration(self, chain, long_expiration_contract):
        """Test rejection of contracts with expiration too far"""
        assert chain._meets_quality_criteria(long_expiration_contract) is False
    
    def test_meets_quality_criteria_missing_bid_ask(self, chain):
        """Test rejection of contracts with missing bid/ask data"""
        contract = {
            'type': 'CALL',
            'strike': 150.0,
            'days_to_expiration': 45,
            'open_interest': 300,
            'bid': 0,  # Invalid
            'ask': 0   # Invalid
        }
        assert chain._meets_quality_criteria(contract) is False
    
    def test_filter_quality_contracts(self, chain, quality_contract, low_oi_contract, wide_spread_contract):
        """Test filtering of mixed quality contracts"""
        contracts = [quality_contract, low_oi_contract, wide_spread_contract]
        
        quality_contracts = chain._filter_quality_contracts(contracts)
        
        assert len(quality_contracts) == 1
        assert quality_contracts[0] == quality_contract
    
    def test_evaluate_contracts_programmatically_all_quality(self, chain, quality_contract):
        """Test programmatic evaluation with all quality contracts"""
        contracts = [quality_contract] * 5  # 5 quality contracts
        
        result = chain.evaluate_contracts_programmatically(contracts)
        
        assert result['passed'] is True
        assert result['confidence'] > 0.0
        assert result['quality_contracts_count'] == 5
        assert len(result['best_contracts']) == 5
    
    def test_evaluate_contracts_programmatically_no_quality(self, chain, low_oi_contract, wide_spread_contract):
        """Test programmatic evaluation with no quality contracts"""
        contracts = [low_oi_contract, wide_spread_contract]
        
        result = chain.evaluate_contracts_programmatically(contracts)
        
        assert result['passed'] is False
        assert result['confidence'] == 0.0
        assert result['quality_contracts_count'] == 0
    
    def test_evaluate_contracts_programmatically_mixed(self, chain, quality_contract, low_oi_contract):
        """Test programmatic evaluation with mixed quality"""
        contracts = [quality_contract, quality_contract, low_oi_contract, low_oi_contract]
        
        result = chain.evaluate_contracts_programmatically(contracts)
        
        assert result['passed'] is True
        assert result['quality_contracts_count'] == 2
        assert 0.0 < result['confidence'] < 1.0
    
    def test_create_quality_summary_no_contracts(self, chain):
        """Test summary creation with no quality contracts"""
        summary = chain._create_quality_summary([])
        
        assert "No contracts meet the quality criteria" in summary
    
    def test_create_quality_summary_with_contracts(self, chain, quality_contract):
        """Test summary creation with quality contracts"""
        contracts = [quality_contract] * 3
        
        summary = chain._create_quality_summary(contracts)
        
        assert "Total Quality Contracts: 3" in summary
        assert "Puts: 3" in summary
        assert "Top Quality Contracts:" in summary
    
    def test_create_quality_summary_mixed_types(self, chain, quality_contract):
        """Test summary creation with mixed call/put contracts"""
        call_contract = quality_contract.copy()
        call_contract['type'] = 'CALL'
        
        contracts = [quality_contract, call_contract]
        
        summary = chain._create_quality_summary(contracts)
        
        assert "Calls: 1" in summary
        assert "Puts: 1" in summary
    
    @pytest.mark.asyncio
    async def test_evaluate_with_quality_contracts(self, chain, mock_llm, quality_contract):
        """Test async evaluate method with quality contracts"""
        options_data = {
            'contracts': [quality_contract] * 5,
            'expiration_dates': [date.today() + timedelta(days=45)]
        }
        
        # Mock LLM response
        mock_response = """{
            "passed": true,
            "confidence": 0.85,
            "reasoning": "Multiple quality contracts available with good liquidity and tight spreads",
            "quality_contracts_count": 5,
            "best_contracts": [],
            "recommendation": "Sufficient quality contracts available for trading"
        }"""
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = mock_response
            
            result = await chain.evaluate('AAPL', options_data)
            
            assert isinstance(result, OptionsAnalysisResult)
            assert result.passed is True
            assert result.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_evaluate_with_no_quality_contracts(self, chain, mock_llm, low_oi_contract):
        """Test async evaluate method with no quality contracts"""
        options_data = {
            'contracts': [low_oi_contract] * 3,
            'expiration_dates': [date.today() + timedelta(days=45)]
        }
        
        # Mock LLM response
        mock_response = """{
            "passed": false,
            "confidence": 0.0,
            "reasoning": "No contracts meet the quality criteria for liquidity and spreads",
            "quality_contracts_count": 0,
            "best_contracts": [],
            "recommendation": "Insufficient quality contracts for trading"
        }"""
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = mock_response
            
            result = await chain.evaluate('AAPL', options_data)
            
            assert isinstance(result, OptionsAnalysisResult)
            assert result.passed is False
    
    @pytest.mark.asyncio
    async def test_evaluate_missing_contracts_data(self, chain):
        """Test evaluate method with missing contracts data"""
        options_data = {
            'expiration_dates': [date.today() + timedelta(days=45)]
        }
        
        result = await chain.evaluate('AAPL', options_data)
        
        assert result.passed is False
        assert result.confidence == 0.0
        assert 'error' in result.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_evaluate_llm_error_handling(self, chain, mock_llm, quality_contract):
        """Test error handling when LLM fails"""
        options_data = {
            'contracts': [quality_contract],
            'expiration_dates': [date.today() + timedelta(days=45)]
        }
        
        with patch.object(chain.chain, 'arun', new_callable=AsyncMock) as mock_arun:
            mock_arun.side_effect = Exception("LLM API error")
            
            result = await chain.evaluate('AAPL', options_data)
            
            assert result.passed is False
            assert result.confidence == 0.0
            assert 'error' in result.reasoning.lower()
    
    def test_confidence_calculation_high_quality(self, chain, quality_contract):
        """Test confidence calculation with high quality contracts"""
        # Many contracts with high open interest
        high_oi_contracts = []
        for i in range(10):
            contract = quality_contract.copy()
            contract['open_interest'] = 1000 + i * 100
            high_oi_contracts.append(contract)
        
        result = chain.evaluate_contracts_programmatically(high_oi_contracts)
        
        assert result['confidence'] > 0.7
    
    def test_confidence_calculation_low_quality(self, chain, quality_contract):
        """Test confidence calculation with marginal quality contracts"""
        # Few contracts with minimum acceptable metrics
        marginal_contract = quality_contract.copy()
        marginal_contract['open_interest'] = 100  # Minimum
        marginal_contract['bid'] = 2.00
        marginal_contract['ask'] = 2.10  # 4.9% spread (just under limit)
        
        result = chain.evaluate_contracts_programmatically([marginal_contract])
        
        assert 0.0 < result['confidence'] < 0.5
    
    def test_boundary_values(self, chain):
        """Test contracts at exact boundary values"""
        # Contract at exact boundaries
        boundary_contract = {
            'type': 'PUT',
            'strike': 150.0,
            'expiration': date.today() + timedelta(days=30),
            'days_to_expiration': 30,  # Exactly at minimum
            'bid': 2.00,
            'ask': 2.10,  # Exactly 5% spread
            'open_interest': 100,  # Exactly at minimum
            'volume': 50
        }
        
        assert chain._meets_quality_criteria(boundary_contract) is True
