"""
Options Evaluation Agent - Orchestrator for sequential chain execution.

Implements requirements 2.7, 3.6, 4.5, 5.5, 6.4, 7.1:
- Coordinate sequential execution of all analysis chains
- Implement error handling and fallback logic between chains
- Add performance monitoring and logging for chain execution times
- Ensure complete workflow from technical analysis to final recommendation
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI

from backend.app.chains.technical_analysis_chain import TechnicalAnalysisChain
from backend.app.chains.fundamental_screening_chain import FundamentalScreeningChain
from backend.app.chains.options_analysis_chain import OptionsAnalysisChain
from backend.app.chains.strategy_selection_chain import StrategySelectionChain
from backend.app.chains.risk_assessment_chain import RiskAssessmentChain
from backend.app.services.market_data_service import MarketDataService, DataUnavailableError
from backend.app.services.options_data_service import OptionsDataService, OptionsUnavailableError
from backend.app.models.api import TradeRecommendation, ReasoningStep, Contract, RiskMetrics

logger = logging.getLogger(__name__)


class OptionsEvaluationAgent:
    """
    Main orchestrator for options trade evaluation.
    
    Coordinates sequential execution of analysis chains:
    1. Technical Analysis
    2. Fundamental Screening
    3. Options Analysis
    4. Strategy Selection
    5. Risk Assessment
    
    Implements error handling, fallback logic, and performance monitoring.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the options evaluation agent.
        
        Args:
            llm: LangChain LLM instance (defaults to GPT-4)
        """
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        
        # Initialize all chains
        self.technical_chain = TechnicalAnalysisChain(self.llm)
        self.fundamental_chain = FundamentalScreeningChain(self.llm)
        self.options_chain = OptionsAnalysisChain(self.llm)
        self.strategy_chain = StrategySelectionChain(self.llm)
        self.risk_chain = RiskAssessmentChain(self.llm)
        
        # Initialize data services
        self.market_data_service = MarketDataService()
        self.options_data_service = OptionsDataService()
        
        # Performance tracking
        self.execution_times = {}
    
    async def evaluate_trade(self, ticker: str) -> TradeRecommendation:
        """
        Evaluate a potential options trade through complete analysis workflow.
        
        Implements requirement 7.1: Complete analysis within 5 seconds average
        
        Args:
            ticker: Stock ticker symbol to analyze
        
        Returns:
            TradeRecommendation with complete analysis results
        
        Raises:
            Exception: If critical errors occur during evaluation
        """
        start_time = time.time()
        reasoning_steps = []
        
        try:
            logger.info(f"Starting options evaluation for {ticker}")
            
            # Step 1: Technical Analysis
            logger.info(f"Step 1: Technical Analysis for {ticker}")
            step_start = time.time()
            
            try:
                technical_result = await self._execute_technical_analysis(ticker)
                step_time = time.time() - step_start
                self.execution_times['technical_analysis'] = step_time
                
                reasoning_steps.append(ReasoningStep(
                    step="Technical Analysis",
                    passed=technical_result['passed'],
                    reasoning=technical_result['reasoning'],
                    confidence=technical_result['confidence']
                ))
                
                # Early exit if technical analysis fails
                if not technical_result['passed']:
                    logger.info(f"Technical analysis failed for {ticker}, stopping evaluation")
                    return self._create_rejection_recommendation(
                        ticker=ticker,
                        reasoning_steps=reasoning_steps,
                        reason="Failed technical analysis criteria"
                    )
                
            except Exception as e:
                logger.error(f"Technical analysis error for {ticker}: {str(e)}")
                reasoning_steps.append(ReasoningStep(
                    step="Technical Analysis",
                    passed=False,
                    reasoning=f"Error during technical analysis: {str(e)}",
                    confidence=0.0
                ))
                return self._create_rejection_recommendation(
                    ticker=ticker,
                    reasoning_steps=reasoning_steps,
                    reason="Technical analysis error"
                )
            
            # Step 2: Fundamental Screening
            logger.info(f"Step 2: Fundamental Screening for {ticker}")
            step_start = time.time()
            
            try:
                fundamental_result = await self._execute_fundamental_screening(ticker)
                step_time = time.time() - step_start
                self.execution_times['fundamental_screening'] = step_time
                
                reasoning_steps.append(ReasoningStep(
                    step="Fundamental Screening",
                    passed=fundamental_result['passed'],
                    reasoning=fundamental_result['reasoning'],
                    confidence=fundamental_result['confidence']
                ))
                
                # Early exit if fundamental screening fails
                if not fundamental_result['passed']:
                    logger.info(f"Fundamental screening failed for {ticker}, stopping evaluation")
                    return self._create_rejection_recommendation(
                        ticker=ticker,
                        reasoning_steps=reasoning_steps,
                        reason="Failed fundamental screening criteria"
                    )
                
            except Exception as e:
                logger.error(f"Fundamental screening error for {ticker}: {str(e)}")
                reasoning_steps.append(ReasoningStep(
                    step="Fundamental Screening",
                    passed=False,
                    reasoning=f"Error during fundamental screening: {str(e)}",
                    confidence=0.0
                ))
                return self._create_rejection_recommendation(
                    ticker=ticker,
                    reasoning_steps=reasoning_steps,
                    reason="Fundamental screening error"
                )
            
            # Step 3: Options Analysis
            logger.info(f"Step 3: Options Analysis for {ticker}")
            step_start = time.time()
            
            try:
                options_result = await self._execute_options_analysis(ticker)
                step_time = time.time() - step_start
                self.execution_times['options_analysis'] = step_time
                
                reasoning_steps.append(ReasoningStep(
                    step="Options Analysis",
                    passed=options_result['passed'],
                    reasoning=options_result['reasoning'],
                    confidence=options_result['confidence']
                ))
                
                # Early exit if no quality options available
                if not options_result['passed']:
                    logger.info(f"Options analysis failed for {ticker}, stopping evaluation")
                    return self._create_rejection_recommendation(
                        ticker=ticker,
                        reasoning_steps=reasoning_steps,
                        reason="No quality options contracts available"
                    )
                
            except Exception as e:
                logger.error(f"Options analysis error for {ticker}: {str(e)}")
                reasoning_steps.append(ReasoningStep(
                    step="Options Analysis",
                    passed=False,
                    reasoning=f"Error during options analysis: {str(e)}",
                    confidence=0.0
                ))
                return self._create_rejection_recommendation(
                    ticker=ticker,
                    reasoning_steps=reasoning_steps,
                    reason="Options analysis error"
                )
            
            # Step 4: Strategy Selection
            logger.info(f"Step 4: Strategy Selection for {ticker}")
            step_start = time.time()
            
            try:
                strategy_result = await self._execute_strategy_selection(
                    ticker,
                    technical_result,
                    fundamental_result,
                    options_result
                )
                step_time = time.time() - step_start
                self.execution_times['strategy_selection'] = step_time
                
                reasoning_steps.append(ReasoningStep(
                    step="Strategy Selection",
                    passed=strategy_result['passed'],
                    reasoning=strategy_result['reasoning'],
                    confidence=strategy_result['confidence']
                ))
                
                # Early exit if no suitable strategy
                if not strategy_result['passed']:
                    logger.info(f"Strategy selection failed for {ticker}, stopping evaluation")
                    return self._create_rejection_recommendation(
                        ticker=ticker,
                        reasoning_steps=reasoning_steps,
                        reason="No suitable strategy for current market conditions"
                    )
                
            except Exception as e:
                logger.error(f"Strategy selection error for {ticker}: {str(e)}")
                reasoning_steps.append(ReasoningStep(
                    step="Strategy Selection",
                    passed=False,
                    reasoning=f"Error during strategy selection: {str(e)}",
                    confidence=0.0
                ))
                return self._create_rejection_recommendation(
                    ticker=ticker,
                    reasoning_steps=reasoning_steps,
                    reason="Strategy selection error"
                )
            
            # Step 5: Risk Assessment
            logger.info(f"Step 5: Risk Assessment for {ticker}")
            step_start = time.time()
            
            try:
                risk_result = await self._execute_risk_assessment(
                    ticker,
                    strategy_result,
                    technical_result
                )
                step_time = time.time() - step_start
                self.execution_times['risk_assessment'] = step_time
                
                reasoning_steps.append(ReasoningStep(
                    step="Risk Assessment",
                    passed=risk_result['should_trade'],
                    reasoning=risk_result['reasoning'],
                    confidence=risk_result['confidence']
                ))
                
                # Create final recommendation
                total_time = time.time() - start_time
                self.execution_times['total'] = total_time
                
                logger.info(f"Evaluation complete for {ticker} in {total_time:.2f}s: should_trade={risk_result['should_trade']}")
                
                return TradeRecommendation(
                    should_trade=risk_result['should_trade'],
                    confidence=risk_result['confidence'],
                    strategy=strategy_result.get('strategy_name'),
                    contracts=risk_result.get('contracts', []),
                    risk_metrics=risk_result.get('risk_metrics'),
                    reasoning_steps=reasoning_steps
                )
                
            except Exception as e:
                logger.error(f"Risk assessment error for {ticker}: {str(e)}")
                reasoning_steps.append(ReasoningStep(
                    step="Risk Assessment",
                    passed=False,
                    reasoning=f"Error during risk assessment: {str(e)}",
                    confidence=0.0
                ))
                return self._create_rejection_recommendation(
                    ticker=ticker,
                    reasoning_steps=reasoning_steps,
                    reason="Risk assessment error"
                )
        
        except Exception as e:
            logger.error(f"Unexpected error during evaluation for {ticker}: {str(e)}")
            total_time = time.time() - start_time
            self.execution_times['total'] = total_time
            
            return self._create_rejection_recommendation(
                ticker=ticker,
                reasoning_steps=reasoning_steps,
                reason=f"Unexpected error: {str(e)}"
            )
    
    async def _execute_technical_analysis(self, ticker: str) -> Dict[str, Any]:
        """Execute technical analysis step"""
        # Fetch technical data
        technical_data_obj = await self.market_data_service.get_technical_indicators(ticker)
        
        technical_data = {
            'price': technical_data_obj.price,
            'ma_50': technical_data_obj.ma_50,
            'ma_200': technical_data_obj.ma_200,
            'rsi': technical_data_obj.rsi,
            'volume': technical_data_obj.volume,
            'iv_rank': technical_data_obj.iv_rank,
            'beta': technical_data_obj.beta
        }
        
        # Run technical analysis chain
        result = await self.technical_chain.evaluate(ticker, technical_data)
        
        return {
            'passed': result.passed,
            'confidence': result.confidence,
            'reasoning': result.reasoning,
            'technical_data': technical_data
        }
    
    async def _execute_fundamental_screening(self, ticker: str) -> Dict[str, Any]:
        """Execute fundamental screening step"""
        # Fetch fundamental data
        fundamental_data_obj = await self.market_data_service.get_fundamental_data(ticker)
        
        fundamental_data = {
            'market_cap': fundamental_data_obj.market_cap,
            'pe_ratio': fundamental_data_obj.pe_ratio,
            'debt_to_equity': fundamental_data_obj.debt_to_equity,
            'earnings_date': fundamental_data_obj.earnings_date,
            'news_sentiment': fundamental_data_obj.news_sentiment
        }
        
        # Run fundamental screening chain
        result = await self.fundamental_chain.evaluate(ticker, fundamental_data)
        
        return {
            'passed': result.passed,
            'confidence': result.confidence,
            'reasoning': result.reasoning,
            'fundamental_data': fundamental_data
        }
    
    async def _execute_options_analysis(self, ticker: str) -> Dict[str, Any]:
        """Execute options analysis step"""
        # Fetch options chain data
        options_chain = await self.options_data_service.get_options_chain(ticker, max_expirations=2)
        
        # Filter for quality contracts
        all_contracts = options_chain.calls + options_chain.puts
        quality_contracts = self.options_data_service.filter_quality_contracts(all_contracts)
        
        # Convert to dict format for chain
        contracts_dict = [
            {
                'type': c.contract_type,
                'strike': c.strike,
                'expiration': c.expiration,
                'days_to_expiration': c.days_to_expiration,
                'bid': c.bid,
                'ask': c.ask,
                'open_interest': c.open_interest,
                'volume': c.volume
            }
            for c in quality_contracts
        ]
        
        options_data = {
            'contracts': contracts_dict,
            'expiration_dates': options_chain.expiration_dates
        }
        
        # Run options analysis chain
        result = await self.options_chain.evaluate(ticker, options_data)
        
        return {
            'passed': result.passed,
            'confidence': result.confidence,
            'reasoning': result.reasoning,
            'quality_contracts': contracts_dict
        }
    
    async def _execute_strategy_selection(self, ticker: str, technical_result: Dict[str, Any],
                                         fundamental_result: Dict[str, Any],
                                         options_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strategy selection step"""
        analysis_context = {
            'technical_data': technical_result['technical_data'],
            'fundamental_data': fundamental_result['fundamental_data'],
            'quality_contracts': options_result['quality_contracts']
        }
        
        # Run strategy selection chain
        result = await self.strategy_chain.evaluate(ticker, analysis_context)
        
        return {
            'passed': result.passed,
            'confidence': result.confidence,
            'reasoning': result.reasoning,
            'strategy_name': result.strategy_name,
            'contract_recommendations': result.contract_recommendations
        }
    
    async def _execute_risk_assessment(self, ticker: str, strategy_result: Dict[str, Any],
                                      technical_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute risk assessment step"""
        strategy_context = {
            'strategy_name': strategy_result['strategy_name'],
            'contracts': strategy_result.get('contract_recommendations', []),
            'current_price': technical_result['technical_data']['price'],
            'technical_outlook': 'Bullish',  # Simplified
            'fundamental_health': 'Healthy',  # Simplified
            'iv_rank': technical_result['technical_data'].get('iv_rank')
        }
        
        # Run risk assessment chain
        result = await self.risk_chain.evaluate(ticker, strategy_context)
        
        # Convert to dict format
        contracts = [
            Contract(
                action=c.action,
                type=c.type,
                strike=c.strike,
                expiration=c.expiration,
                quantity=c.quantity,
                premium_credit=c.premium_credit
            )
            for c in result.contracts
        ]
        
        risk_metrics = RiskMetrics(
            max_profit=result.risk_metrics.max_profit,
            max_loss=result.risk_metrics.max_loss,
            breakeven=result.risk_metrics.breakeven,
            prob_profit=result.risk_metrics.prob_profit,
            return_on_capital=result.risk_metrics.return_on_capital
        )
        
        return {
            'should_trade': result.should_trade,
            'confidence': result.confidence,
            'reasoning': result.reasoning,
            'contracts': contracts,
            'risk_metrics': risk_metrics
        }
    
    def _create_rejection_recommendation(self, ticker: str, reasoning_steps: list,
                                        reason: str) -> TradeRecommendation:
        """Create a rejection recommendation when analysis fails"""
        return TradeRecommendation(
            should_trade=False,
            confidence=0.0,
            strategy=None,
            contracts=[],
            risk_metrics=None,
            reasoning_steps=reasoning_steps
        )
    
    def get_execution_times(self) -> Dict[str, float]:
        """
        Get execution times for performance monitoring.
        
        Returns:
            Dictionary with execution times for each step
        """
        return self.execution_times.copy()
    
    def close(self):
        """Close all service connections"""
        self.market_data_service.close()
        self.options_data_service.close()
