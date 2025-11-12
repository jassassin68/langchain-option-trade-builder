"""
Fundamental Screening Chain for evaluating company financial health.

Implements requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6:
- Evaluate market cap, P/E ratio, debt-to-equity, earnings date, and news sentiment
- Reject companies with market cap < $1B
- Flag potentially overvalued companies (P/E > 50)
- Flag high leverage risk (debt-to-equity > 2.0)
- Reject stocks with earnings within 14 days
- Return boolean result with reasoning
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FundamentalScreeningResult(BaseModel):
    """Output model for fundamental screening chain"""
    passed: bool = Field(..., description="Whether the stock passed fundamental screening")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    reasoning: str = Field(..., description="Detailed reasoning for the decision")
    criteria_results: Dict[str, bool] = Field(..., description="Individual criteria pass/fail results")
    recommendation: str = Field(..., description="Summary recommendation")


class FundamentalScreeningChain:
    """
    LangChain chain for fundamental screening evaluation.
    
    Uses GPT-4 to evaluate fundamental health indicators and determine if a company
    is financially stable enough for options trading.
    """
    
    # Fundamental criteria thresholds
    MIN_MARKET_CAP = 1_000_000_000  # $1 billion
    MAX_PE_RATIO = 50.0
    MAX_DEBT_TO_EQUITY = 2.0
    MIN_DAYS_TO_EARNINGS = 14
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the fundamental screening chain.
        
        Args:
            llm: LangChain LLM instance (defaults to GPT-4)
        """
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        self.output_parser = PydanticOutputParser(pydantic_object=FundamentalScreeningResult)
        self.chain = self._create_chain()
    
    def _create_chain(self) -> LLMChain:
        """Create the LangChain chain with prompt template"""
        
        prompt_template = """You are an expert fundamental analyst evaluating companies for options trading opportunities.

Analyze the following fundamental indicators and determine if the company is financially healthy enough for options trading.

FUNDAMENTAL DATA:
- Ticker: {ticker}
- Market Cap: {market_cap}
- P/E Ratio: {pe_ratio}
- Debt-to-Equity Ratio: {debt_to_equity}
- Days to Next Earnings: {days_to_earnings}
- News Sentiment: {news_sentiment}

EVALUATION CRITERIA:
1. Market cap must be >= $1 billion (avoid small-cap risk)
2. P/E ratio should be <= 50 (flag potentially overvalued companies)
3. Debt-to-equity should be <= 2.0 (avoid high leverage risk)
4. Earnings date must be > 14 days away (avoid earnings volatility)
5. News sentiment should not indicate major negative events

ANALYSIS REQUIREMENTS:
- Evaluate each criterion individually
- Consider the overall fundamental health picture
- Provide specific reasoning for pass/fail decisions
- Assign a confidence score based on how well criteria are met
- If any critical criterion fails (market cap, earnings date), the stock should not pass

Provide your analysis in the following format:
{format_instructions}

Be thorough but concise in your reasoning. Focus on financial stability and risk factors."""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["ticker", "market_cap", "pe_ratio", "debt_to_equity", 
                           "days_to_earnings", "news_sentiment"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    async def evaluate(self, ticker: str, fundamental_data: Dict[str, Any]) -> FundamentalScreeningResult:
        """
        Evaluate fundamental indicators for a company.
        
        Args:
            ticker: Stock ticker symbol
            fundamental_data: Dictionary containing fundamental indicator values
                Expected keys: market_cap, pe_ratio, debt_to_equity, earnings_date, news_sentiment
        
        Returns:
            FundamentalScreeningResult with evaluation results
        
        Raises:
            ValueError: If required fundamental data is missing
        """
        try:
            logger.info(f"Starting fundamental screening for {ticker}")
            
            # Calculate days to earnings if earnings_date is provided
            days_to_earnings = "N/A"
            if fundamental_data.get('earnings_date'):
                earnings_date = fundamental_data['earnings_date']
                if isinstance(earnings_date, datetime):
                    days_to_earnings = (earnings_date.date() - datetime.now().date()).days
                elif isinstance(earnings_date, str):
                    try:
                        earnings_dt = datetime.fromisoformat(earnings_date)
                        days_to_earnings = (earnings_dt.date() - datetime.now().date()).days
                    except:
                        days_to_earnings = "N/A"
            
            # Format market cap for display
            market_cap = fundamental_data.get('market_cap')
            market_cap_display = f"${market_cap:,.0f}" if market_cap else "N/A"
            
            # Prepare input data
            input_data = {
                'ticker': ticker,
                'market_cap': market_cap_display,
                'pe_ratio': fundamental_data.get('pe_ratio', 'N/A'),
                'debt_to_equity': fundamental_data.get('debt_to_equity', 'N/A'),
                'days_to_earnings': days_to_earnings,
                'news_sentiment': fundamental_data.get('news_sentiment', 'N/A')
            }
            
            # Run the chain
            result = await self.chain.arun(**input_data)
            
            # Parse the output
            parsed_result = self.output_parser.parse(result)
            
            logger.info(f"Fundamental screening complete for {ticker}: passed={parsed_result.passed}, confidence={parsed_result.confidence}")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error in fundamental screening for {ticker}: {str(e)}")
            # Return a failed result with error information
            return FundamentalScreeningResult(
                passed=False,
                confidence=0.0,
                reasoning=f"Fundamental screening failed due to error: {str(e)}",
                criteria_results={},
                recommendation="Unable to complete fundamental screening"
            )
    
    def evaluate_criteria_programmatically(self, fundamental_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Programmatically evaluate fundamental criteria without LLM.
        
        This method provides a rule-based evaluation that can be used
        as a fallback or for validation.
        
        Args:
            fundamental_data: Dictionary containing fundamental indicator values
        
        Returns:
            Dictionary with evaluation results
        """
        criteria_results = {}
        reasons = []
        
        # Check market cap (requirement 3.2)
        market_cap = fundamental_data.get('market_cap')
        if market_cap is not None:
            if market_cap < self.MIN_MARKET_CAP:
                criteria_results['market_cap'] = False
                reasons.append(f"Market cap ${market_cap:,.0f} is below minimum ${self.MIN_MARKET_CAP:,.0f} (insufficient size)")
            else:
                criteria_results['market_cap'] = True
        else:
            criteria_results['market_cap'] = None
            reasons.append("Market cap data not available")
        
        # Check P/E ratio (requirement 3.3)
        pe_ratio = fundamental_data.get('pe_ratio')
        if pe_ratio is not None:
            if pe_ratio > self.MAX_PE_RATIO:
                criteria_results['pe_ratio'] = False
                reasons.append(f"P/E ratio {pe_ratio:.2f} exceeds maximum {self.MAX_PE_RATIO} (potentially overvalued)")
            else:
                criteria_results['pe_ratio'] = True
        else:
            criteria_results['pe_ratio'] = None
            reasons.append("P/E ratio data not available")
        
        # Check debt-to-equity (requirement 3.4)
        debt_to_equity = fundamental_data.get('debt_to_equity')
        if debt_to_equity is not None:
            if debt_to_equity > self.MAX_DEBT_TO_EQUITY:
                criteria_results['debt_to_equity'] = False
                reasons.append(f"Debt-to-equity {debt_to_equity:.2f} exceeds maximum {self.MAX_DEBT_TO_EQUITY} (high leverage risk)")
            else:
                criteria_results['debt_to_equity'] = True
        else:
            criteria_results['debt_to_equity'] = None
            reasons.append("Debt-to-equity data not available")
        
        # Check earnings date (requirement 3.5)
        earnings_date = fundamental_data.get('earnings_date')
        if earnings_date is not None:
            if isinstance(earnings_date, datetime):
                days_to_earnings = (earnings_date.date() - datetime.now().date()).days
            elif isinstance(earnings_date, str):
                try:
                    earnings_dt = datetime.fromisoformat(earnings_date)
                    days_to_earnings = (earnings_dt.date() - datetime.now().date()).days
                except:
                    days_to_earnings = None
            else:
                days_to_earnings = None
            
            if days_to_earnings is not None:
                if days_to_earnings < self.MIN_DAYS_TO_EARNINGS:
                    criteria_results['earnings_date'] = False
                    reasons.append(f"Earnings in {days_to_earnings} days, less than minimum {self.MIN_DAYS_TO_EARNINGS} days (earnings risk)")
                else:
                    criteria_results['earnings_date'] = True
            else:
                criteria_results['earnings_date'] = None
                reasons.append("Unable to calculate days to earnings")
        else:
            criteria_results['earnings_date'] = None
            reasons.append("Earnings date not available")
        
        # Check news sentiment (requirement 3.1)
        news_sentiment = fundamental_data.get('news_sentiment')
        if news_sentiment is not None:
            if news_sentiment.lower() in ['negative', 'very negative']:
                criteria_results['news_sentiment'] = False
                reasons.append(f"News sentiment is {news_sentiment} (potential risk)")
            else:
                criteria_results['news_sentiment'] = True
        else:
            criteria_results['news_sentiment'] = None
            reasons.append("News sentiment not available")
        
        # Determine overall pass/fail
        # Critical criteria: market_cap and earnings_date must pass
        critical_pass = (
            criteria_results.get('market_cap', False) and
            criteria_results.get('earnings_date', True)  # Default to True if not available
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
