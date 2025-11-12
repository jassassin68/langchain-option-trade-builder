"""
Technical Analysis Chain for evaluating stock technical indicators.

Implements requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7:
- Evaluate current stock price, 50-day MA, 200-day MA, RSI, daily volume, IV rank, and beta
- Reject penny stocks (price < $10)
- Reject low liquidity stocks (volume < 500,000)
- Flag oversold/overbought conditions (RSI outside 30-70)
- Reject low premium opportunities (IV rank < 20)
- Flag unreasonable volatility (beta outside 0.5-2.0)
- Return boolean result with detailed reasoning
"""

import logging
from typing import Dict, Any, Optional
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TechnicalAnalysisResult(BaseModel):
    """Output model for technical analysis chain"""
    passed: bool = Field(..., description="Whether the stock passed technical analysis")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    reasoning: str = Field(..., description="Detailed reasoning for the decision")
    criteria_results: Dict[str, bool] = Field(..., description="Individual criteria pass/fail results")
    recommendation: str = Field(..., description="Summary recommendation")


class TechnicalAnalysisChain:
    """
    LangChain chain for technical analysis evaluation.
    
    Uses GPT-4 to evaluate technical indicators and determine if a stock
    meets the criteria for options trading.
    """
    
    # Technical criteria thresholds
    MIN_PRICE = 10.0
    MIN_VOLUME = 500000
    RSI_MIN = 30
    RSI_MAX = 70
    MIN_IV_RANK = 20
    BETA_MIN = 0.5
    BETA_MAX = 2.0
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the technical analysis chain.
        
        Args:
            llm: LangChain LLM instance (defaults to GPT-4)
        """
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        self.output_parser = PydanticOutputParser(pydantic_object=TechnicalAnalysisResult)
        self.chain = self._create_chain()
    
    def _create_chain(self) -> LLMChain:
        """Create the LangChain chain with prompt template"""
        
        prompt_template = """You are an expert technical analyst evaluating stocks for options trading opportunities.

Analyze the following technical indicators and determine if the stock meets the criteria for options trading.

TECHNICAL DATA:
- Ticker: {ticker}
- Current Price: ${price:.2f}
- 50-Day Moving Average: ${ma_50}
- 200-Day Moving Average: ${ma_200}
- RSI (14-period): {rsi}
- Daily Volume: {volume:,} shares
- IV Rank: {iv_rank}
- Beta: {beta}

EVALUATION CRITERIA:
1. Price must be >= $10 (reject penny stocks)
2. Daily volume must be >= 500,000 shares (ensure liquidity)
3. RSI should be between 30-70 (avoid extreme oversold/overbought)
4. IV Rank should be >= 20 (ensure sufficient premium)
5. Beta should be between 0.5-2.0 (reasonable volatility)

ANALYSIS REQUIREMENTS:
- Evaluate each criterion individually
- Consider the overall technical picture
- Provide specific reasoning for pass/fail decisions
- Assign a confidence score based on how well criteria are met
- If any critical criterion fails (price, volume, IV rank), the stock should not pass

Provide your analysis in the following format:
{format_instructions}

Be thorough but concise in your reasoning. Focus on actionable insights."""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["ticker", "price", "ma_50", "ma_200", "rsi", "volume", "iv_rank", "beta"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    async def evaluate(self, ticker: str, technical_data: Dict[str, Any]) -> TechnicalAnalysisResult:
        """
        Evaluate technical indicators for a stock.
        
        Args:
            ticker: Stock ticker symbol
            technical_data: Dictionary containing technical indicator values
                Expected keys: price, ma_50, ma_200, rsi, volume, iv_rank, beta
        
        Returns:
            TechnicalAnalysisResult with evaluation results
        
        Raises:
            ValueError: If required technical data is missing
        """
        try:
            logger.info(f"Starting technical analysis for {ticker}")
            
            # Validate required data
            required_fields = ['price', 'volume']
            for field in required_fields:
                if field not in technical_data or technical_data[field] is None:
                    raise ValueError(f"Missing required technical data: {field}")
            
            # Prepare input data with defaults for optional fields
            input_data = {
                'ticker': ticker,
                'price': technical_data['price'],
                'ma_50': technical_data.get('ma_50', 'N/A'),
                'ma_200': technical_data.get('ma_200', 'N/A'),
                'rsi': technical_data.get('rsi', 'N/A'),
                'volume': technical_data['volume'],
                'iv_rank': technical_data.get('iv_rank', 'N/A'),
                'beta': technical_data.get('beta', 'N/A')
            }
            
            # Run the chain
            result = await self.chain.arun(**input_data)
            
            # Parse the output
            parsed_result = self.output_parser.parse(result)
            
            logger.info(f"Technical analysis complete for {ticker}: passed={parsed_result.passed}, confidence={parsed_result.confidence}")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error in technical analysis for {ticker}: {str(e)}")
            # Return a failed result with error information
            return TechnicalAnalysisResult(
                passed=False,
                confidence=0.0,
                reasoning=f"Technical analysis failed due to error: {str(e)}",
                criteria_results={},
                recommendation="Unable to complete technical analysis"
            )
    
    def evaluate_criteria_programmatically(self, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Programmatically evaluate technical criteria without LLM.
        
        This method provides a rule-based evaluation that can be used
        as a fallback or for validation.
        
        Args:
            technical_data: Dictionary containing technical indicator values
        
        Returns:
            Dictionary with evaluation results
        """
        criteria_results = {}
        reasons = []
        
        # Check price (requirement 2.2)
        price = technical_data.get('price', 0)
        if price < self.MIN_PRICE:
            criteria_results['price'] = False
            reasons.append(f"Price ${price:.2f} is below minimum ${self.MIN_PRICE} (penny stock)")
        else:
            criteria_results['price'] = True
        
        # Check volume (requirement 2.3)
        volume = technical_data.get('volume', 0)
        if volume < self.MIN_VOLUME:
            criteria_results['volume'] = False
            reasons.append(f"Volume {volume:,} is below minimum {self.MIN_VOLUME:,} (insufficient liquidity)")
        else:
            criteria_results['volume'] = True
        
        # Check RSI (requirement 2.4)
        rsi = technical_data.get('rsi')
        if rsi is not None:
            if rsi < self.RSI_MIN:
                criteria_results['rsi'] = False
                reasons.append(f"RSI {rsi:.1f} is below {self.RSI_MIN} (oversold condition)")
            elif rsi > self.RSI_MAX:
                criteria_results['rsi'] = False
                reasons.append(f"RSI {rsi:.1f} is above {self.RSI_MAX} (overbought condition)")
            else:
                criteria_results['rsi'] = True
        else:
            criteria_results['rsi'] = None
            reasons.append("RSI data not available")
        
        # Check IV rank (requirement 2.5)
        iv_rank = technical_data.get('iv_rank')
        if iv_rank is not None:
            if iv_rank < self.MIN_IV_RANK:
                criteria_results['iv_rank'] = False
                reasons.append(f"IV Rank {iv_rank:.1f} is below minimum {self.MIN_IV_RANK} (insufficient premium)")
            else:
                criteria_results['iv_rank'] = True
        else:
            criteria_results['iv_rank'] = None
            reasons.append("IV Rank data not available")
        
        # Check beta (requirement 2.6)
        beta = technical_data.get('beta')
        if beta is not None:
            if beta < self.BETA_MIN or beta > self.BETA_MAX:
                criteria_results['beta'] = False
                reasons.append(f"Beta {beta:.2f} is outside range {self.BETA_MIN}-{self.BETA_MAX} (unreasonable volatility)")
            else:
                criteria_results['beta'] = True
        else:
            criteria_results['beta'] = None
            reasons.append("Beta data not available")
        
        # Determine overall pass/fail
        # Critical criteria: price, volume, iv_rank must pass
        critical_pass = (
            criteria_results.get('price', False) and
            criteria_results.get('volume', False) and
            criteria_results.get('iv_rank', False)
        )
        
        # Calculate confidence based on criteria met
        total_criteria = len([v for v in criteria_results.values() if v is not None])
        passed_criteria = len([v for v in criteria_results.values() if v is True])
        confidence = passed_criteria / total_criteria if total_criteria > 0 else 0.0
        
        return {
            'passed': critical_pass,
            'confidence': confidence,
            'criteria_results': criteria_results,
            'reasons': reasons
        }
