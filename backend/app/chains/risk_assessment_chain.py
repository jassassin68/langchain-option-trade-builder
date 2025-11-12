"""
Risk Assessment Chain for calculating profit/loss and risk metrics.

Implements requirements 6.1, 6.2, 6.3, 6.4, 6.5:
- Calculate max profit, max loss, breakeven prices, probability of profit, and return on capital
- Provide specific contract details (action, type, strike, expiration, quantity, premium)
- Include confidence score between 0-1
- Return clear YES/NO recommendation with detailed reasoning
- Recommend against trading if risk/reward is unfavorable
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import date
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import math

logger = logging.getLogger(__name__)


class RiskMetrics(BaseModel):
    """Risk metrics for the trade"""
    max_profit: float = Field(..., description="Maximum potential profit")
    max_loss: float = Field(..., description="Maximum potential loss")
    breakeven: Union[float, List[float]] = Field(..., description="Breakeven price(s)")
    prob_profit: float = Field(..., ge=0, le=1, description="Probability of profit")
    return_on_capital: float = Field(..., description="Return on capital percentage")


class ContractDetail(BaseModel):
    """Contract detail for recommendation"""
    action: str = Field(..., description="BUY or SELL")
    type: str = Field(..., description="CALL or PUT")
    strike: float = Field(..., description="Strike price")
    expiration: str = Field(..., description="Expiration date")
    quantity: int = Field(..., description="Number of contracts")
    premium_credit: Optional[float] = Field(None, description="Premium received/paid")


class RiskAssessmentResult(BaseModel):
    """Output model for risk assessment chain"""
    should_trade: bool = Field(..., description="Final YES/NO recommendation")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    reasoning: str = Field(..., description="Detailed reasoning for the recommendation")
    risk_metrics: RiskMetrics = Field(..., description="Calculated risk metrics")
    contracts: List[ContractDetail] = Field(..., description="Specific contract recommendations")
    recommendation: str = Field(..., description="Summary recommendation")


class RiskAssessmentChain:
    """
    LangChain chain for risk assessment and final recommendation.
    
    Uses GPT-4 to calculate risk metrics and provide final trading recommendation.
    """
    
    # Risk/reward thresholds
    MIN_RETURN_ON_CAPITAL = 2.0  # Minimum 2% return
    MAX_RISK_REWARD_RATIO = 3.0  # Max loss should not exceed 3x max profit
    MIN_PROB_PROFIT = 0.40  # Minimum 40% probability of profit
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the risk assessment chain.
        
        Args:
            llm: LangChain LLM instance (defaults to GPT-4)
        """
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        self.output_parser = PydanticOutputParser(pydantic_object=RiskAssessmentResult)
        self.chain = self._create_chain()
    
    def _create_chain(self) -> LLMChain:
        """Create the LangChain chain with prompt template"""
        
        prompt_template = """You are an expert risk analyst providing final trading recommendations for options strategies.

Analyze the following strategy details and calculate comprehensive risk metrics to provide a final YES/NO recommendation.

STRATEGY DETAILS:
- Ticker: {ticker}
- Strategy: {strategy_name}
- Current Stock Price: ${current_price:.2f}

PROPOSED CONTRACTS:
{contracts_summary}

MARKET CONTEXT:
- Technical Outlook: {technical_outlook}
- Fundamental Health: {fundamental_health}
- IV Rank: {iv_rank}

RISK ASSESSMENT REQUIREMENTS:
1. Calculate maximum profit potential
2. Calculate maximum loss exposure
3. Determine breakeven price(s)
4. Estimate probability of profit based on:
   - Distance to breakeven
   - Current volatility
   - Time to expiration
5. Calculate return on capital (ROC)

DECISION CRITERIA:
- Return on capital should be >= 2%
- Risk/reward ratio should be favorable (max loss <= 3x max profit)
- Probability of profit should be >= 40%
- Overall risk should be acceptable given market conditions
- If any critical risk metric is unfavorable, recommend against trading

ANALYSIS REQUIREMENTS:
- Provide specific calculations for all risk metrics
- Explain the risk/reward profile clearly
- Give a definitive YES or NO recommendation
- Assign confidence based on risk metrics and market conditions
- Include specific contract details in the recommendation

Provide your analysis in the following format:
{format_instructions}

Be thorough in risk calculations and conservative in recommendations. Focus on capital preservation."""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["ticker", "strategy_name", "current_price", "contracts_summary",
                           "technical_outlook", "fundamental_health", "iv_rank"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    async def evaluate(self, ticker: str, strategy_context: Dict[str, Any]) -> RiskAssessmentResult:
        """
        Perform risk assessment and provide final recommendation.
        
        Args:
            ticker: Stock ticker symbol
            strategy_context: Dictionary containing strategy and market context
                Expected keys: strategy_name, contracts, current_price, technical_outlook,
                              fundamental_health, iv_rank
        
        Returns:
            RiskAssessmentResult with risk metrics and final recommendation
        
        Raises:
            ValueError: If required strategy context is missing
        """
        try:
            logger.info(f"Starting risk assessment for {ticker}")
            
            # Extract data from context
            strategy_name = strategy_context.get('strategy_name')
            contracts = strategy_context.get('contracts', [])
            current_price = strategy_context.get('current_price', 0)
            
            if not strategy_name:
                raise ValueError("Missing required strategy context: strategy_name")
            
            # Create contracts summary
            contracts_summary = self._create_contracts_summary(contracts, current_price)
            
            # Prepare input data
            input_data = {
                'ticker': ticker,
                'strategy_name': strategy_name,
                'current_price': current_price,
                'contracts_summary': contracts_summary,
                'technical_outlook': strategy_context.get('technical_outlook', 'N/A'),
                'fundamental_health': strategy_context.get('fundamental_health', 'N/A'),
                'iv_rank': strategy_context.get('iv_rank', 'N/A')
            }
            
            # Run the chain
            result = await self.chain.arun(**input_data)
            
            # Parse the output
            parsed_result = self.output_parser.parse(result)
            
            logger.info(f"Risk assessment complete for {ticker}: should_trade={parsed_result.should_trade}, confidence={parsed_result.confidence}")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error in risk assessment for {ticker}: {str(e)}")
            # Return a failed result with error information
            return RiskAssessmentResult(
                should_trade=False,
                confidence=0.0,
                reasoning=f"Risk assessment failed due to error: {str(e)}",
                risk_metrics=RiskMetrics(
                    max_profit=0.0,
                    max_loss=0.0,
                    breakeven=0.0,
                    prob_profit=0.0,
                    return_on_capital=0.0
                ),
                contracts=[],
                recommendation="Unable to complete risk assessment"
            )
    
    def _create_contracts_summary(self, contracts: List[Dict[str, Any]], current_price: float) -> str:
        """
        Create a summary of proposed contracts.
        
        Args:
            contracts: List of contract dictionaries
            current_price: Current stock price
        
        Returns:
            Formatted summary string
        """
        if not contracts:
            return "No specific contracts proposed."
        
        summary_lines = []
        for i, contract in enumerate(contracts, 1):
            action = contract.get('action', 'N/A')
            contract_type = contract.get('type', 'N/A')
            strike = contract.get('strike', 0)
            expiration = contract.get('expiration', 'N/A')
            quantity = contract.get('quantity', 1)
            premium = contract.get('premium_credit', 0)
            
            # Calculate moneyness
            if contract_type == 'PUT':
                moneyness = "ITM" if strike > current_price else "OTM"
            else:  # CALL
                moneyness = "ITM" if strike < current_price else "OTM"
            
            summary_lines.append(
                f"{i}. {action} {quantity} {contract_type} ${strike:.2f} exp {expiration} "
                f"({moneyness}, Premium: ${premium:.2f})"
            )
        
        return "\n".join(summary_lines)
    
    def calculate_risk_metrics(self, strategy_name: str, contracts: List[Dict[str, Any]], 
                               current_price: float, iv_rank: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate risk metrics programmatically for a given strategy.
        
        Args:
            strategy_name: Name of the options strategy
            contracts: List of contract dictionaries
            current_price: Current stock price
            iv_rank: Implied volatility rank (optional)
        
        Returns:
            Dictionary with calculated risk metrics
        """
        if not contracts:
            return {
                'max_profit': 0.0,
                'max_loss': 0.0,
                'breakeven': 0.0,
                'prob_profit': 0.0,
                'return_on_capital': 0.0,
                'should_trade': False
            }
        
        # Calculate based on strategy type
        if strategy_name == "Cash-Secured Put":
            return self._calculate_cash_secured_put_metrics(contracts, current_price, iv_rank)
        elif strategy_name == "Iron Condor":
            return self._calculate_iron_condor_metrics(contracts, current_price, iv_rank)
        elif strategy_name == "Credit Put Spread":
            return self._calculate_credit_spread_metrics(contracts, current_price, iv_rank)
        elif strategy_name == "Covered Call":
            return self._calculate_covered_call_metrics(contracts, current_price, iv_rank)
        else:
            # Generic calculation
            return self._calculate_generic_metrics(contracts, current_price, iv_rank)
    
    def _calculate_cash_secured_put_metrics(self, contracts: List[Dict[str, Any]], 
                                           current_price: float, iv_rank: Optional[float]) -> Dict[str, Any]:
        """Calculate metrics for cash-secured put strategy"""
        contract = contracts[0]  # Single put contract
        strike = contract.get('strike', 0)
        premium = contract.get('premium_credit', 0)
        quantity = contract.get('quantity', 1)
        
        # Max profit = premium received
        max_profit = premium * 100 * quantity
        
        # Max loss = (strike - premium) * 100 * quantity
        max_loss = (strike - premium) * 100 * quantity
        
        # Breakeven = strike - premium
        breakeven = strike - premium
        
        # Probability of profit (simplified - based on distance to breakeven)
        distance_pct = abs(current_price - breakeven) / current_price
        prob_profit = min(0.5 + (distance_pct * 2), 0.85)  # Cap at 85%
        
        # Return on capital
        capital_required = strike * 100 * quantity
        return_on_capital = (max_profit / capital_required * 100) if capital_required > 0 else 0
        
        # Decision
        should_trade = (
            return_on_capital >= self.MIN_RETURN_ON_CAPITAL and
            prob_profit >= self.MIN_PROB_PROFIT and
            max_loss / max_profit <= self.MAX_RISK_REWARD_RATIO
        )
        
        return {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'prob_profit': prob_profit,
            'return_on_capital': return_on_capital,
            'should_trade': should_trade
        }
    
    def _calculate_iron_condor_metrics(self, contracts: List[Dict[str, Any]], 
                                      current_price: float, iv_rank: Optional[float]) -> Dict[str, Any]:
        """Calculate metrics for iron condor strategy"""
        # Simplified calculation - assumes 4 legs
        total_premium = sum(c.get('premium_credit', 0) for c in contracts)
        quantity = contracts[0].get('quantity', 1) if contracts else 1
        
        # Max profit = net premium received
        max_profit = total_premium * 100 * quantity
        
        # Max loss = width of spread - premium (simplified)
        # Assuming $5 wide spreads
        spread_width = 5.0
        max_loss = (spread_width - total_premium) * 100 * quantity
        
        # Breakeven points (two for iron condor)
        put_strike = min(c.get('strike', 0) for c in contracts if c.get('type') == 'PUT')
        call_strike = max(c.get('strike', 0) for c in contracts if c.get('type') == 'CALL')
        breakeven = [put_strike + total_premium, call_strike - total_premium]
        
        # Probability of profit (higher for iron condor if price is centered)
        prob_profit = 0.65 if abs(current_price - (put_strike + call_strike) / 2) < spread_width else 0.50
        
        # Return on capital
        capital_required = max_loss
        return_on_capital = (max_profit / capital_required * 100) if capital_required > 0 else 0
        
        should_trade = (
            return_on_capital >= self.MIN_RETURN_ON_CAPITAL and
            prob_profit >= self.MIN_PROB_PROFIT
        )
        
        return {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'prob_profit': prob_profit,
            'return_on_capital': return_on_capital,
            'should_trade': should_trade
        }
    
    def _calculate_credit_spread_metrics(self, contracts: List[Dict[str, Any]], 
                                        current_price: float, iv_rank: Optional[float]) -> Dict[str, Any]:
        """Calculate metrics for credit spread strategy"""
        # Assumes 2 legs (sell and buy)
        net_premium = sum(c.get('premium_credit', 0) for c in contracts)
        quantity = contracts[0].get('quantity', 1) if contracts else 1
        
        # Max profit = net premium
        max_profit = net_premium * 100 * quantity
        
        # Max loss = spread width - premium
        strikes = [c.get('strike', 0) for c in contracts]
        spread_width = abs(max(strikes) - min(strikes))
        max_loss = (spread_width - net_premium) * 100 * quantity
        
        # Breakeven
        short_strike = max(strikes)  # Assuming put spread
        breakeven = short_strike - net_premium
        
        # Probability of profit
        distance_pct = abs(current_price - breakeven) / current_price
        prob_profit = min(0.5 + (distance_pct * 2), 0.80)
        
        # Return on capital
        capital_required = max_loss
        return_on_capital = (max_profit / capital_required * 100) if capital_required > 0 else 0
        
        should_trade = (
            return_on_capital >= self.MIN_RETURN_ON_CAPITAL and
            prob_profit >= self.MIN_PROB_PROFIT
        )
        
        return {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'prob_profit': prob_profit,
            'return_on_capital': return_on_capital,
            'should_trade': should_trade
        }
    
    def _calculate_covered_call_metrics(self, contracts: List[Dict[str, Any]], 
                                       current_price: float, iv_rank: Optional[float]) -> Dict[str, Any]:
        """Calculate metrics for covered call strategy"""
        contract = contracts[0]  # Single call contract
        strike = contract.get('strike', 0)
        premium = contract.get('premium_credit', 0)
        quantity = contract.get('quantity', 1)
        
        # Max profit = premium + (strike - current_price) if called away
        max_profit = (premium + max(0, strike - current_price)) * 100 * quantity
        
        # Max loss = unlimited (stock ownership risk), but limited by strike
        max_loss = (current_price - premium) * 100 * quantity
        
        # Breakeven = current_price - premium
        breakeven = current_price - premium
        
        # Probability of profit (high for covered calls)
        prob_profit = 0.70
        
        # Return on capital
        capital_required = current_price * 100 * quantity
        return_on_capital = (max_profit / capital_required * 100) if capital_required > 0 else 0
        
        should_trade = return_on_capital >= self.MIN_RETURN_ON_CAPITAL
        
        return {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'prob_profit': prob_profit,
            'return_on_capital': return_on_capital,
            'should_trade': should_trade
        }
    
    def _calculate_generic_metrics(self, contracts: List[Dict[str, Any]], 
                                   current_price: float, iv_rank: Optional[float]) -> Dict[str, Any]:
        """Calculate generic metrics when strategy is unknown"""
        total_premium = sum(c.get('premium_credit', 0) for c in contracts)
        quantity = contracts[0].get('quantity', 1) if contracts else 1
        
        return {
            'max_profit': total_premium * 100 * quantity,
            'max_loss': total_premium * 100 * quantity * 2,  # Estimate
            'breakeven': current_price,
            'prob_profit': 0.50,
            'return_on_capital': 5.0,  # Estimate
            'should_trade': False  # Conservative default
        }
