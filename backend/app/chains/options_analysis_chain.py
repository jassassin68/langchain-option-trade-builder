"""
Options Analysis Chain for evaluating options contract quality.

Implements requirements 4.1, 4.2, 4.3, 4.4, 4.5:
- Analyze options chain data for next 2 expiration cycles
- Reject contracts with open interest < 100
- Reject contracts with bid-ask spread > 5% of mid price
- Reject contracts with expiration < 30 or > 60 days
- Return quality assessment with specific contract data
"""

import logging
from typing import Dict, Any, Optional, List
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OptionsAnalysisResult(BaseModel):
    """Output model for options analysis chain"""
    passed: bool = Field(..., description="Whether quality options contracts are available")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    reasoning: str = Field(..., description="Detailed reasoning for the decision")
    quality_contracts_count: int = Field(..., description="Number of contracts meeting quality criteria")
    best_contracts: List[Dict[str, Any]] = Field(default_factory=list, description="Best quality contracts identified")
    recommendation: str = Field(..., description="Summary recommendation")


class OptionsAnalysisChain:
    """
    LangChain chain for options quality analysis.
    
    Uses GPT-4 to evaluate options chain data and identify quality contracts
    suitable for trading.
    """
    
    # Options quality criteria thresholds
    MIN_OPEN_INTEREST = 100
    MAX_SPREAD_PERCENTAGE = 5.0
    MIN_DAYS_TO_EXPIRATION = 30
    MAX_DAYS_TO_EXPIRATION = 60
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the options analysis chain.
        
        Args:
            llm: LangChain LLM instance (defaults to GPT-4)
        """
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        self.output_parser = PydanticOutputParser(pydantic_object=OptionsAnalysisResult)
        self.chain = self._create_chain()
    
    def _create_chain(self) -> LLMChain:
        """Create the LangChain chain with prompt template"""
        
        prompt_template = """You are an expert options trader evaluating options contract quality for trading opportunities.

Analyze the following options chain data and determine if there are quality contracts available for trading.

OPTIONS CHAIN DATA:
- Ticker: {ticker}
- Total Contracts Available: {total_contracts}
- Expiration Dates: {expiration_dates}
- Quality Contracts Found: {quality_contracts_count}

QUALITY CONTRACTS SUMMARY:
{quality_contracts_summary}

EVALUATION CRITERIA:
1. Open interest must be >= 100 contracts (ensure liquidity)
2. Bid-ask spread must be <= 5% of mid price (tight spreads)
3. Days to expiration must be between 30-60 days (optimal time frame)
4. Multiple quality contracts should be available for strategy flexibility

ANALYSIS REQUIREMENTS:
- Evaluate the overall quality of available options
- Consider liquidity, spreads, and time to expiration
- Identify the best contracts for potential strategies
- Provide specific reasoning for quality assessment
- Assign a confidence score based on contract quality and availability
- If no quality contracts are found, the analysis should not pass

Provide your analysis in the following format:
{format_instructions}

Be thorough but concise in your reasoning. Focus on tradability and risk management."""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["ticker", "total_contracts", "expiration_dates", 
                           "quality_contracts_count", "quality_contracts_summary"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    async def evaluate(self, ticker: str, options_data: Dict[str, Any]) -> OptionsAnalysisResult:
        """
        Evaluate options chain quality for a stock.
        
        Args:
            ticker: Stock ticker symbol
            options_data: Dictionary containing options chain data
                Expected keys: contracts (list), expiration_dates (list)
        
        Returns:
            OptionsAnalysisResult with evaluation results
        
        Raises:
            ValueError: If required options data is missing
        """
        try:
            logger.info(f"Starting options analysis for {ticker}")
            
            # Validate required data
            if 'contracts' not in options_data:
                raise ValueError("Missing required options data: contracts")
            
            contracts = options_data['contracts']
            expiration_dates = options_data.get('expiration_dates', [])
            
            # Filter for quality contracts
            quality_contracts = self._filter_quality_contracts(contracts)
            
            # Prepare summary of quality contracts
            quality_summary = self._create_quality_summary(quality_contracts)
            
            # Prepare input data
            input_data = {
                'ticker': ticker,
                'total_contracts': len(contracts),
                'expiration_dates': ', '.join([str(d) for d in expiration_dates]) if expiration_dates else 'N/A',
                'quality_contracts_count': len(quality_contracts),
                'quality_contracts_summary': quality_summary
            }
            
            # Run the chain
            result = await self.chain.arun(**input_data)
            
            # Parse the output
            parsed_result = self.output_parser.parse(result)
            
            logger.info(f"Options analysis complete for {ticker}: passed={parsed_result.passed}, quality_contracts={len(quality_contracts)}")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error in options analysis for {ticker}: {str(e)}")
            # Return a failed result with error information
            return OptionsAnalysisResult(
                passed=False,
                confidence=0.0,
                reasoning=f"Options analysis failed due to error: {str(e)}",
                quality_contracts_count=0,
                best_contracts=[],
                recommendation="Unable to complete options analysis"
            )
    
    def _filter_quality_contracts(self, contracts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter contracts to only include those meeting quality criteria.
        
        Args:
            contracts: List of contract dictionaries
        
        Returns:
            Filtered list of quality contracts
        """
        quality_contracts = []
        
        for contract in contracts:
            if self._meets_quality_criteria(contract):
                quality_contracts.append(contract)
        
        return quality_contracts
    
    def _meets_quality_criteria(self, contract: Dict[str, Any]) -> bool:
        """
        Check if a contract meets all quality criteria.
        
        Args:
            contract: Contract dictionary
        
        Returns:
            True if contract meets all criteria, False otherwise
        """
        # Check open interest (requirement 4.2)
        open_interest = contract.get('open_interest', 0)
        if open_interest < self.MIN_OPEN_INTEREST:
            return False
        
        # Check bid-ask spread (requirement 4.3)
        bid = contract.get('bid', 0)
        ask = contract.get('ask', 0)
        if bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2
            spread_pct = ((ask - bid) / mid_price * 100) if mid_price > 0 else 100
            if spread_pct > self.MAX_SPREAD_PERCENTAGE:
                return False
        else:
            return False  # No valid bid/ask data
        
        # Check days to expiration (requirement 4.4)
        days_to_expiration = contract.get('days_to_expiration', 0)
        if days_to_expiration < self.MIN_DAYS_TO_EXPIRATION or days_to_expiration > self.MAX_DAYS_TO_EXPIRATION:
            return False
        
        return True
    
    def _create_quality_summary(self, quality_contracts: List[Dict[str, Any]]) -> str:
        """
        Create a summary of quality contracts for the LLM prompt.
        
        Args:
            quality_contracts: List of quality contract dictionaries
        
        Returns:
            Formatted summary string
        """
        if not quality_contracts:
            return "No contracts meet the quality criteria."
        
        # Group by type and expiration
        calls = [c for c in quality_contracts if c.get('type') == 'CALL']
        puts = [c for c in quality_contracts if c.get('type') == 'PUT']
        
        summary_lines = []
        summary_lines.append(f"Total Quality Contracts: {len(quality_contracts)}")
        summary_lines.append(f"- Calls: {len(calls)}")
        summary_lines.append(f"- Puts: {len(puts)}")
        
        # Add details for top contracts
        if quality_contracts:
            summary_lines.append("\nTop Quality Contracts:")
            # Sort by open interest and take top 5
            sorted_contracts = sorted(quality_contracts, key=lambda c: c.get('open_interest', 0), reverse=True)[:5]
            
            for i, contract in enumerate(sorted_contracts, 1):
                contract_type = contract.get('type', 'N/A')
                strike = contract.get('strike', 0)
                expiration = contract.get('expiration', 'N/A')
                open_interest = contract.get('open_interest', 0)
                bid = contract.get('bid', 0)
                ask = contract.get('ask', 0)
                mid = (bid + ask) / 2 if bid and ask else 0
                spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 0
                
                summary_lines.append(
                    f"{i}. {contract_type} ${strike:.2f} exp {expiration}: "
                    f"OI={open_interest}, Mid=${mid:.2f}, Spread={spread_pct:.2f}%"
                )
        
        return "\n".join(summary_lines)
    
    def evaluate_contracts_programmatically(self, contracts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Programmatically evaluate options contracts without LLM.
        
        This method provides a rule-based evaluation that can be used
        as a fallback or for validation.
        
        Args:
            contracts: List of contract dictionaries
        
        Returns:
            Dictionary with evaluation results
        """
        quality_contracts = self._filter_quality_contracts(contracts)
        
        # Analyze quality distribution
        calls = [c for c in quality_contracts if c.get('type') == 'CALL']
        puts = [c for c in quality_contracts if c.get('type') == 'PUT']
        
        # Calculate average metrics for quality contracts
        if quality_contracts:
            avg_open_interest = sum(c.get('open_interest', 0) for c in quality_contracts) / len(quality_contracts)
            avg_spread_pct = sum(
                ((c.get('ask', 0) - c.get('bid', 0)) / ((c.get('ask', 0) + c.get('bid', 0)) / 2) * 100)
                if (c.get('ask', 0) + c.get('bid', 0)) > 0 else 0
                for c in quality_contracts
            ) / len(quality_contracts)
        else:
            avg_open_interest = 0
            avg_spread_pct = 0
        
        # Determine pass/fail
        passed = len(quality_contracts) > 0
        
        # Calculate confidence based on quality and quantity
        if passed:
            # Base confidence on number of quality contracts and their metrics
            quantity_score = min(len(quality_contracts) / 10, 1.0)  # Max at 10 contracts
            quality_score = min(avg_open_interest / 1000, 1.0)  # Max at 1000 OI
            spread_score = max(0, 1.0 - (avg_spread_pct / self.MAX_SPREAD_PERCENTAGE))
            
            confidence = (quantity_score * 0.4 + quality_score * 0.3 + spread_score * 0.3)
        else:
            confidence = 0.0
        
        return {
            'passed': passed,
            'confidence': confidence,
            'quality_contracts_count': len(quality_contracts),
            'calls_count': len(calls),
            'puts_count': len(puts),
            'avg_open_interest': avg_open_interest,
            'avg_spread_percentage': avg_spread_pct,
            'best_contracts': quality_contracts[:5]  # Top 5
        }
