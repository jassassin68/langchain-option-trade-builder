"""
Strategy Selection Chain for recommending options trading strategies.

Implements requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6:
- Recommend cash-secured put when bullish + healthy fundamentals + high IV
- Recommend iron condor when neutral + moderate IV
- Recommend credit put spread when strong support + high IV
- Recommend covered call when appropriate conditions met
- Return strategy name with specific contract recommendations
- Recommend against trading when no suitable strategy found
"""

import logging
from typing import Dict, Any, Optional, List
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StrategyRecommendation(BaseModel):
    """Output model for strategy selection chain"""
    strategy_name: Optional[str] = Field(None, description="Name of recommended strategy")
    passed: bool = Field(..., description="Whether a suitable strategy was found")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    reasoning: str = Field(..., description="Detailed reasoning for strategy selection")
    contract_recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Specific contracts for the strategy")
    recommendation: str = Field(..., description="Summary recommendation")


class StrategySelectionChain:
    """
    LangChain chain for options strategy selection.
    
    Uses GPT-4 to analyze market conditions and recommend appropriate
    options trading strategies.
    """
    
    # Strategy names
    CASH_SECURED_PUT = "Cash-Secured Put"
    IRON_CONDOR = "Iron Condor"
    CREDIT_PUT_SPREAD = "Credit Put Spread"
    COVERED_CALL = "Covered Call"
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the strategy selection chain.
        
        Args:
            llm: LangChain LLM instance (defaults to GPT-4)
        """
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        self.output_parser = PydanticOutputParser(pydantic_object=StrategyRecommendation)
        self.chain = self._create_chain()
    
    def _create_chain(self) -> LLMChain:
        """Create the LangChain chain with prompt template"""
        
        prompt_template = """You are an expert options strategist selecting the optimal trading strategy based on market conditions.

Analyze the following market data and previous analysis results to recommend an appropriate options strategy.

MARKET CONDITIONS:
- Ticker: {ticker}
- Current Price: ${current_price:.2f}
- Technical Outlook: {technical_outlook}
- Fundamental Health: {fundamental_health}
- IV Rank: {iv_rank}
- RSI: {rsi}
- Price vs 50-day MA: {price_vs_ma50}
- Price vs 200-day MA: {price_vs_ma200}

AVAILABLE QUALITY CONTRACTS:
{quality_contracts_summary}

STRATEGY SELECTION RULES:
1. Cash-Secured Put: Use when technical analysis is bullish AND fundamentals are healthy AND IV rank is high (>40)
   - Sell out-of-the-money puts to collect premium
   - Requires strong support levels and willingness to own stock

2. Iron Condor: Use when technical analysis is neutral AND IV rank is moderate (20-40)
   - Sell both call and put spreads around current price
   - Profits from low volatility and range-bound movement

3. Credit Put Spread: Use when there is strong support level AND IV rank is high (>40)
   - Sell put spread below support
   - Limited risk defined spread strategy

4. Covered Call: Use when already own stock OR willing to buy stock AND IV rank is moderate to high
   - Sell out-of-the-money calls against stock position
   - Generates income on existing holdings

5. No Trade: Recommend when conditions don't clearly favor any strategy OR risk/reward is unfavorable

ANALYSIS REQUIREMENTS:
- Match market conditions to appropriate strategy
- Consider risk tolerance and capital requirements
- Provide specific contract recommendations from available quality contracts
- Explain why the selected strategy fits current conditions
- Assign confidence based on how well conditions match strategy criteria
- If no strategy is suitable, clearly recommend against trading

Provide your analysis in the following format:
{format_instructions}

Be decisive and clear in your strategy recommendation. Focus on risk-adjusted returns."""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["ticker", "current_price", "technical_outlook", "fundamental_health",
                           "iv_rank", "rsi", "price_vs_ma50", "price_vs_ma200", "quality_contracts_summary"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    async def evaluate(self, ticker: str, analysis_context: Dict[str, Any]) -> StrategyRecommendation:
        """
        Select optimal options strategy based on analysis context.
        
        Args:
            ticker: Stock ticker symbol
            analysis_context: Dictionary containing all previous analysis results
                Expected keys: technical_data, fundamental_data, options_data, quality_contracts
        
        Returns:
            StrategyRecommendation with selected strategy and reasoning
        
        Raises:
            ValueError: If required analysis context is missing
        """
        try:
            logger.info(f"Starting strategy selection for {ticker}")
            
            # Extract data from context
            technical_data = analysis_context.get('technical_data', {})
            fundamental_data = analysis_context.get('fundamental_data', {})
            quality_contracts = analysis_context.get('quality_contracts', [])
            
            # Prepare input data
            current_price = technical_data.get('price', 0)
            ma_50 = technical_data.get('ma_50')
            ma_200 = technical_data.get('ma_200')
            
            # Calculate price vs MA relationships
            price_vs_ma50 = "N/A"
            if ma_50:
                diff_pct = ((current_price - ma_50) / ma_50 * 100)
                price_vs_ma50 = f"{diff_pct:+.2f}% ({'above' if diff_pct > 0 else 'below'})"
            
            price_vs_ma200 = "N/A"
            if ma_200:
                diff_pct = ((current_price - ma_200) / ma_200 * 100)
                price_vs_ma200 = f"{diff_pct:+.2f}% ({'above' if diff_pct > 0 else 'below'})"
            
            # Determine technical outlook
            technical_outlook = self._determine_technical_outlook(technical_data)
            
            # Determine fundamental health
            fundamental_health = self._determine_fundamental_health(fundamental_data)
            
            # Create quality contracts summary
            contracts_summary = self._create_contracts_summary(quality_contracts)
            
            input_data = {
                'ticker': ticker,
                'current_price': current_price,
                'technical_outlook': technical_outlook,
                'fundamental_health': fundamental_health,
                'iv_rank': technical_data.get('iv_rank', 'N/A'),
                'rsi': technical_data.get('rsi', 'N/A'),
                'price_vs_ma50': price_vs_ma50,
                'price_vs_ma200': price_vs_ma200,
                'quality_contracts_summary': contracts_summary
            }
            
            # Run the chain
            result = await self.chain.arun(**input_data)
            
            # Parse the output
            parsed_result = self.output_parser.parse(result)
            
            logger.info(f"Strategy selection complete for {ticker}: strategy={parsed_result.strategy_name}, passed={parsed_result.passed}")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error in strategy selection for {ticker}: {str(e)}")
            # Return a failed result with error information
            return StrategyRecommendation(
                strategy_name=None,
                passed=False,
                confidence=0.0,
                reasoning=f"Strategy selection failed due to error: {str(e)}",
                contract_recommendations=[],
                recommendation="Unable to complete strategy selection"
            )
    
    def _determine_technical_outlook(self, technical_data: Dict[str, Any]) -> str:
        """
        Determine technical outlook (bullish, bearish, neutral) from technical data.
        
        Args:
            technical_data: Dictionary with technical indicators
        
        Returns:
            String describing technical outlook
        """
        price = technical_data.get('price', 0)
        ma_50 = technical_data.get('ma_50')
        ma_200 = technical_data.get('ma_200')
        rsi = technical_data.get('rsi')
        
        bullish_signals = 0
        bearish_signals = 0
        
        # Check price vs moving averages
        if ma_50 and price > ma_50:
            bullish_signals += 1
        elif ma_50 and price < ma_50:
            bearish_signals += 1
        
        if ma_200 and price > ma_200:
            bullish_signals += 1
        elif ma_200 and price < ma_200:
            bearish_signals += 1
        
        # Check RSI
        if rsi:
            if rsi > 50:
                bullish_signals += 1
            elif rsi < 50:
                bearish_signals += 1
        
        # Determine outlook
        if bullish_signals > bearish_signals:
            return "Bullish"
        elif bearish_signals > bullish_signals:
            return "Bearish"
        else:
            return "Neutral"
    
    def _determine_fundamental_health(self, fundamental_data: Dict[str, Any]) -> str:
        """
        Determine fundamental health from fundamental data.
        
        Args:
            fundamental_data: Dictionary with fundamental indicators
        
        Returns:
            String describing fundamental health
        """
        healthy_signals = 0
        unhealthy_signals = 0
        
        # Check market cap
        market_cap = fundamental_data.get('market_cap')
        if market_cap and market_cap >= 1_000_000_000:
            healthy_signals += 1
        elif market_cap:
            unhealthy_signals += 1
        
        # Check P/E ratio
        pe_ratio = fundamental_data.get('pe_ratio')
        if pe_ratio and pe_ratio <= 50:
            healthy_signals += 1
        elif pe_ratio:
            unhealthy_signals += 1
        
        # Check debt-to-equity
        debt_to_equity = fundamental_data.get('debt_to_equity')
        if debt_to_equity and debt_to_equity <= 2.0:
            healthy_signals += 1
        elif debt_to_equity:
            unhealthy_signals += 1
        
        # Determine health
        if healthy_signals > unhealthy_signals:
            return "Healthy"
        elif unhealthy_signals > healthy_signals:
            return "Weak"
        else:
            return "Moderate"
    
    def _create_contracts_summary(self, quality_contracts: List[Dict[str, Any]]) -> str:
        """
        Create a summary of available quality contracts.
        
        Args:
            quality_contracts: List of quality contract dictionaries
        
        Returns:
            Formatted summary string
        """
        if not quality_contracts:
            return "No quality contracts available."
        
        calls = [c for c in quality_contracts if c.get('type') == 'CALL']
        puts = [c for c in quality_contracts if c.get('type') == 'PUT']
        
        summary_lines = []
        summary_lines.append(f"Total: {len(quality_contracts)} contracts ({len(calls)} calls, {len(puts)} puts)")
        
        # Add sample contracts
        if puts:
            sample_put = puts[0]
            summary_lines.append(f"Sample PUT: ${sample_put.get('strike', 0):.2f} exp {sample_put.get('expiration', 'N/A')}")
        
        if calls:
            sample_call = calls[0]
            summary_lines.append(f"Sample CALL: ${sample_call.get('strike', 0):.2f} exp {sample_call.get('expiration', 'N/A')}")
        
        return "\n".join(summary_lines)
    
    def select_strategy_programmatically(self, analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Programmatically select strategy without LLM.
        
        This method provides a rule-based strategy selection that can be used
        as a fallback or for validation.
        
        Args:
            analysis_context: Dictionary containing all previous analysis results
        
        Returns:
            Dictionary with strategy selection results
        """
        technical_data = analysis_context.get('technical_data', {})
        fundamental_data = analysis_context.get('fundamental_data', {})
        quality_contracts = analysis_context.get('quality_contracts', [])
        
        technical_outlook = self._determine_technical_outlook(technical_data)
        fundamental_health = self._determine_fundamental_health(fundamental_data)
        iv_rank = technical_data.get('iv_rank', 0)
        
        strategy_name = None
        confidence = 0.0
        reasoning = []
        
        # Rule-based strategy selection
        
        # Cash-Secured Put (requirement 5.1)
        if (technical_outlook == "Bullish" and 
            fundamental_health == "Healthy" and 
            iv_rank and iv_rank > 40):
            strategy_name = self.CASH_SECURED_PUT
            confidence = 0.85
            reasoning.append("Bullish technical outlook with healthy fundamentals and high IV rank")
            reasoning.append("Cash-secured put strategy allows collecting premium while potentially acquiring stock")
        
        # Iron Condor (requirement 5.2)
        elif (technical_outlook == "Neutral" and 
              iv_rank and 20 <= iv_rank <= 40):
            strategy_name = self.IRON_CONDOR
            confidence = 0.75
            reasoning.append("Neutral technical outlook with moderate IV rank")
            reasoning.append("Iron condor profits from range-bound movement and volatility contraction")
        
        # Credit Put Spread (requirement 5.3)
        elif (iv_rank and iv_rank > 40 and 
              technical_data.get('ma_50') and 
              technical_data.get('price', 0) > technical_data.get('ma_50', 0)):
            strategy_name = self.CREDIT_PUT_SPREAD
            confidence = 0.80
            reasoning.append("High IV rank with price above 50-day MA (support level)")
            reasoning.append("Credit put spread offers defined risk with premium collection")
        
        # Covered Call (requirement 5.4)
        elif (iv_rank and iv_rank >= 20 and 
              technical_outlook in ["Neutral", "Bullish"]):
            strategy_name = self.COVERED_CALL
            confidence = 0.70
            reasoning.append("Moderate to high IV with neutral to bullish outlook")
            reasoning.append("Covered call generates income on stock holdings")
        
        # No suitable strategy (requirement 5.6)
        else:
            reasoning.append("Market conditions do not clearly favor any strategy")
            reasoning.append(f"Technical: {technical_outlook}, Fundamental: {fundamental_health}, IV Rank: {iv_rank}")
        
        passed = strategy_name is not None
        
        return {
            'strategy_name': strategy_name,
            'passed': passed,
            'confidence': confidence,
            'reasoning': " | ".join(reasoning),
            'technical_outlook': technical_outlook,
            'fundamental_health': fundamental_health,
            'iv_rank': iv_rank
        }
