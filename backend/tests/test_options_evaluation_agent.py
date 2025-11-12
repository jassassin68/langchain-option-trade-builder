"""
Integration tests for Options Evaluation Agent.

Tests complete agent workflow including:
- Sequential chain execution
- Error handling and fallback logic
- Performance monitoring
- Early exit conditions
- Complete successful evaluation
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, date, timedelta
from backend.app.chains.options_evaluation_agent import OptionsEvaluationAgent
from backend.app.models.api import TradeRecommendation
from backend.app.services.market_data_service import TechnicalData, FundamentalData, DataUnavailableError
from backend.app.services.options_data_service import OptionsChain, OptionsContract, OptionsUnavailableError


class TestOptionsEvaluationAgent:
    """Test suite for OptionsEvaluationAgent"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing"""
        llm = Mock()
        llm.apredict = AsyncMock()
        return llm
    
    @pytest.fixture
    def agent(self, mock_llm):
        """Create an OptionsEvaluationAgent instance with mock LLM"""
        return OptionsEvaluationAgent(llm=mock_llm)
    
    @pytest.fixture
    def mock_technical_data(self):
        """Create mock technical data"""
        return TechnicalData(
            ticker='AAPL',
            price=150.0,
            ma_50=145.0,
            ma_200=140.0,
            rsi=55.0,
            volume=2000000,
            iv_rank=45.0,
            beta=1.2
        )
    
    @pytest.fixture
    def mock_fundamental_data(self):
        """Create mock fundamental data"""
        return FundamentalData(
            ticker='AAPL',
            market_cap=2_500_000_000_000,
            pe_ratio=28.0,
            debt_to_equity=1.5,
            earnings_date=datetime.now() + timedelta(days=30),
            news_sentiment='positive'
        )
    
    @pytest.fixture
    def mock_options_chain(self):
        """Create mock options chain"""
        contracts = [
            OptionsContract(
                ticker='AAPL',
                contract_type='PUT',
                strike=145.0,
                expiration=date.today() + timedelta(days=45),
                bid=2.50,
                ask=2.60,
                last=2.55,
                volume=500,
                open_interest=1000,
                implied_volatility=0.30
            ),
            OptionsContract(
                ticker='AAPL',
                contract_type='CALL',
                strike=155.0,
                expiration=date.today() + timedelta(days=45),
                bid=3.00,
                ask=3.10,
                last=3.05,
                volume=400,
                open_interest=800,
                implied_volatility=0.28
            )
        ]
        
        return OptionsChain(
            ticker='AAPL',
            expiration_dates=[date.today() + timedelta(days=45)],
            calls=[c for c in contracts if c.contract_type == 'CALL'],
            puts=[c for c in contracts if c.contract_type == 'PUT']
        )
    
    @pytest.mark.asyncio
    async def test_complete_successful_evaluation(self, agent, mock_technical_data, 
                                                  mock_fundamental_data, mock_options_chain):
        """Test complete successful evaluation workflow"""
        # Mock all service calls
        with patch.object(agent.market_data_service, 'get_technical_indicators', 
                         new_callable=AsyncMock) as mock_tech, \
             patch.object(agent.market_data_service, 'get_fundamental_data',
                         new_callable=AsyncMock) as mock_fund, \
             patch.object(agent.options_data_service, 'get_options_chain',
                         new_callable=AsyncMock) as mock_opts, \
             patch.object(agent.technical_chain, 'evaluate',
                         new_callable=AsyncMock) as mock_tech_chain, \
             patch.object(agent.fundamental_chain, 'evaluate',
                         new_callable=AsyncMock) as mock_fund_chain, \
             patch.object(agent.options_chain, 'evaluate',
                         new_callable=AsyncMock) as mock_opts_chain, \
             patch.object(agent.strategy_chain, 'evaluate',
                         new_callable=AsyncMock) as mock_strat_chain, \
             patch.object(agent.risk_chain, 'evaluate',
                         new_callable=AsyncMock) as mock_risk_chain:
            
            # Setup mock returns
            mock_tech.return_value = mock_technical_data
            mock_fund.return_value = mock_fundamental_data
            mock_opts.return_value = mock_options_chain
            
            # Mock chain results
            from backend.app.chains.technical_analysis_chain import TechnicalAnalysisResult
            mock_tech_chain.return_value = TechnicalAnalysisResult(
                passed=True,
                confidence=0.85,
                reasoning="All technical criteria met",
                criteria_results={'price': True, 'volume': True},
                recommendation="Pass"
            )
            
            from backend.app.chains.fundamental_screening_chain import FundamentalScreeningResult
            mock_fund_chain.return_value = FundamentalScreeningResult(
                passed=True,
                confidence=0.90,
                reasoning="Strong fundamentals",
                criteria_results={'market_cap': True},
                recommendation="Pass"
            )
            
            from backend.app.chains.options_analysis_chain import OptionsAnalysisResult
            mock_opts_chain.return_value = OptionsAnalysisResult(
                passed=True,
                confidence=0.80,
                reasoning="Quality contracts available",
                quality_contracts_count=2,
                best_contracts=[],
                recommendation="Pass"
            )
            
            from backend.app.chains.strategy_selection_chain import StrategyRecommendation
            mock_strat_chain.return_value = StrategyRecommendation(
                strategy_name="Cash-Secured Put",
                passed=True,
                confidence=0.85,
                reasoning="Bullish conditions with high IV",
                contract_recommendations=[],
                recommendation="Execute strategy"
            )
            
            from backend.app.chains.risk_assessment_chain import (
                RiskAssessmentResult, RiskMetrics as RiskMetricsChain, ContractDetail
            )
            mock_risk_chain.return_value = RiskAssessmentResult(
                should_trade=True,
                confidence=0.80,
                reasoning="Favorable risk/reward",
                risk_metrics=RiskMetricsChain(
                    max_profit=350.0,
                    max_loss=14150.0,
                    breakeven=141.5,
                    prob_profit=0.65,
                    return_on_capital=2.4
                ),
                contracts=[
                    ContractDetail(
                        action="SELL",
                        type="PUT",
                        strike=145.0,
                        expiration="2024-12-15",
                        quantity=1,
                        premium_credit=3.50
                    )
                ],
                recommendation="Execute trade"
            )
            
            # Execute evaluation
            result = await agent.evaluate_trade('AAPL')
            
            # Verify result
            assert isinstance(result, TradeRecommendation)
            assert result.should_trade is True
            assert result.confidence > 0.0
            assert result.strategy == "Cash-Secured Put"
            assert len(result.reasoning_steps) == 5  # All 5 steps completed
            
            # Verify all steps passed
            for step in result.reasoning_steps:
                assert step.passed is True
    
    @pytest.mark.asyncio
    async def test_early_exit_technical_failure(self, agent, mock_technical_data):
        """Test early exit when technical analysis fails"""
        with patch.object(agent.market_data_service, 'get_technical_indicators',
                         new_callable=AsyncMock) as mock_tech, \
             patch.object(agent.technical_chain, 'evaluate',
                         new_callable=AsyncMock) as mock_tech_chain:
            
            mock_tech.return_value = mock_technical_data
            
            from backend.app.chains.technical_analysis_chain import TechnicalAnalysisResult
            mock_tech_chain.return_value = TechnicalAnalysisResult(
                passed=False,
                confidence=0.0,
                reasoning="Price below minimum threshold",
                criteria_results={'price': False},
                recommendation="Reject"
            )
            
            result = await agent.evaluate_trade('AAPL')
            
            assert result.should_trade is False
            assert len(result.reasoning_steps) == 1  # Only technical step executed
            assert result.reasoning_steps[0].step == "Technical Analysis"
            assert result.reasoning_steps[0].passed is False
    
    @pytest.mark.asyncio
    async def test_early_exit_fundamental_failure(self, agent, mock_technical_data, mock_fundamental_data):
        """Test early exit when fundamental screening fails"""
        with patch.object(agent.market_data_service, 'get_technical_indicators',
                         new_callable=AsyncMock) as mock_tech, \
             patch.object(agent.market_data_service, 'get_fundamental_data',
                         new_callable=AsyncMock) as mock_fund, \
             patch.object(agent.technical_chain, 'evaluate',
                         new_callable=AsyncMock) as mock_tech_chain, \
             patch.object(agent.fundamental_chain, 'evaluate',
                         new_callable=AsyncMock) as mock_fund_chain:
            
            mock_tech.return_value = mock_technical_data
            mock_fund.return_value = mock_fundamental_data
            
            from backend.app.chains.technical_analysis_chain import TechnicalAnalysisResult
            mock_tech_chain.return_value = TechnicalAnalysisResult(
                passed=True,
                confidence=0.85,
                reasoning="Technical criteria met",
                criteria_results={'price': True},
                recommendation="Pass"
            )
            
            from backend.app.chains.fundamental_screening_chain import FundamentalScreeningResult
            mock_fund_chain.return_value = FundamentalScreeningResult(
                passed=False,
                confidence=0.0,
                reasoning="Market cap too small",
                criteria_results={'market_cap': False},
                recommendation="Reject"
            )
            
            result = await agent.evaluate_trade('AAPL')
            
            assert result.should_trade is False
            assert len(result.reasoning_steps) == 2  # Technical and fundamental steps
            assert result.reasoning_steps[1].passed is False
    
    @pytest.mark.asyncio
    async def test_error_handling_data_unavailable(self, agent):
        """Test error handling when market data is unavailable"""
        with patch.object(agent.market_data_service, 'get_technical_indicators',
                         new_callable=AsyncMock) as mock_tech:
            
            mock_tech.side_effect = DataUnavailableError("Market data unavailable")
            
            result = await agent.evaluate_trade('AAPL')
            
            assert result.should_trade is False
            assert len(result.reasoning_steps) >= 1
            assert 'error' in result.reasoning_steps[0].reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_error_handling_options_unavailable(self, agent, mock_technical_data, 
                                                      mock_fundamental_data):
        """Test error handling when options data is unavailable"""
        with patch.object(agent.market_data_service, 'get_technical_indicators',
                         new_callable=AsyncMock) as mock_tech, \
             patch.object(agent.market_data_service, 'get_fundamental_data',
                         new_callable=AsyncMock) as mock_fund, \
             patch.object(agent.options_data_service, 'get_options_chain',
                         new_callable=AsyncMock) as mock_opts, \
             patch.object(agent.technical_chain, 'evaluate',
                         new_callable=AsyncMock) as mock_tech_chain, \
             patch.object(agent.fundamental_chain, 'evaluate',
                         new_callable=AsyncMock) as mock_fund_chain:
            
            mock_tech.return_value = mock_technical_data
            mock_fund.return_value = mock_fundamental_data
            mock_opts.side_effect = OptionsUnavailableError("Options data unavailable")
            
            from backend.app.chains.technical_analysis_chain import TechnicalAnalysisResult
            mock_tech_chain.return_value = TechnicalAnalysisResult(
                passed=True, confidence=0.85, reasoning="Pass",
                criteria_results={}, recommendation="Pass"
            )
            
            from backend.app.chains.fundamental_screening_chain import FundamentalScreeningResult
            mock_fund_chain.return_value = FundamentalScreeningResult(
                passed=True, confidence=0.90, reasoning="Pass",
                criteria_results={}, recommendation="Pass"
            )
            
            result = await agent.evaluate_trade('AAPL')
            
            assert result.should_trade is False
            assert len(result.reasoning_steps) == 3  # Technical, fundamental, and options error
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, agent, mock_technical_data):
        """Test that execution times are tracked"""
        with patch.object(agent.market_data_service, 'get_technical_indicators',
                         new_callable=AsyncMock) as mock_tech, \
             patch.object(agent.technical_chain, 'evaluate',
                         new_callable=AsyncMock) as mock_tech_chain:
            
            mock_tech.return_value = mock_technical_data
            
            from backend.app.chains.technical_analysis_chain import TechnicalAnalysisResult
            mock_tech_chain.return_value = TechnicalAnalysisResult(
                passed=False, confidence=0.0, reasoning="Fail",
                criteria_results={}, recommendation="Reject"
            )
            
            await agent.evaluate_trade('AAPL')
            
            execution_times = agent.get_execution_times()
            
            assert 'technical_analysis' in execution_times
            assert 'total' in execution_times
            assert execution_times['technical_analysis'] >= 0
            assert execution_times['total'] >= 0
    
    @pytest.mark.asyncio
    async def test_reasoning_steps_accumulation(self, agent, mock_technical_data, 
                                                mock_fundamental_data, mock_options_chain):
        """Test that reasoning steps accumulate through workflow"""
        with patch.object(agent.market_data_service, 'get_technical_indicators',
                         new_callable=AsyncMock) as mock_tech, \
             patch.object(agent.market_data_service, 'get_fundamental_data',
                         new_callable=AsyncMock) as mock_fund, \
             patch.object(agent.options_data_service, 'get_options_chain',
                         new_callable=AsyncMock) as mock_opts, \
             patch.object(agent.technical_chain, 'evaluate',
                         new_callable=AsyncMock) as mock_tech_chain, \
             patch.object(agent.fundamental_chain, 'evaluate',
                         new_callable=AsyncMock) as mock_fund_chain, \
             patch.object(agent.options_chain, 'evaluate',
                         new_callable=AsyncMock) as mock_opts_chain:
            
            mock_tech.return_value = mock_technical_data
            mock_fund.return_value = mock_fundamental_data
            mock_opts.return_value = mock_options_chain
            
            from backend.app.chains.technical_analysis_chain import TechnicalAnalysisResult
            mock_tech_chain.return_value = TechnicalAnalysisResult(
                passed=True, confidence=0.85, reasoning="Pass",
                criteria_results={}, recommendation="Pass"
            )
            
            from backend.app.chains.fundamental_screening_chain import FundamentalScreeningResult
            mock_fund_chain.return_value = FundamentalScreeningResult(
                passed=True, confidence=0.90, reasoning="Pass",
                criteria_results={}, recommendation="Pass"
            )
            
            from backend.app.chains.options_analysis_chain import OptionsAnalysisResult
            mock_opts_chain.return_value = OptionsAnalysisResult(
                passed=False, confidence=0.0, reasoning="No quality contracts",
                quality_contracts_count=0, best_contracts=[], recommendation="Reject"
            )
            
            result = await agent.evaluate_trade('AAPL')
            
            assert len(result.reasoning_steps) == 3
            assert result.reasoning_steps[0].step == "Technical Analysis"
            assert result.reasoning_steps[1].step == "Fundamental Screening"
            assert result.reasoning_steps[2].step == "Options Analysis"
    
    def test_close_services(self, agent):
        """Test that services are properly closed"""
        agent.close()
        
        # Verify close was called (services should handle cleanup)
        assert True  # If no exception, close worked
