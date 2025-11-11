from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Union, Literal
from datetime import date, datetime, timezone
from enum import Enum
import re

class TickerResult(BaseModel):
    """Model for ticker search results"""
    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Company name")
    exchange: str = Field(default="NYSE", description="Stock exchange")
    
    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        if not re.match(r'^[A-Z]{1,5}$', v):
            raise ValueError('Ticker must be 1-5 uppercase letters')
        return v

class TickerSearchResponse(BaseModel):
    """Response model for ticker search endpoint"""
    results: List[TickerResult] = Field(..., description="List of matching tickers")
    count: int = Field(..., description="Number of results returned")

class TradeAnalysisRequest(BaseModel):
    """Request model for trade analysis"""
    ticker: str = Field(..., description="Stock ticker to analyze")
    
    @field_validator('ticker')
    @classmethod
    def validate_ticker_format(cls, v):
        # Normalize the ticker first
        normalized = v.upper().strip()
        # Then validate the format
        if not re.match(r'^[A-Z]{1,5}$', normalized):
            raise ValueError('Ticker must be 1-5 uppercase letters')
        return normalized

class ActionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class ContractType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"

class Contract(BaseModel):
    """Model for options contract details"""
    action: ActionType = Field(..., description="Buy or sell action")
    type: ContractType = Field(..., description="Call or put option")
    strike: float = Field(..., gt=0, description="Strike price")
    expiration: date = Field(..., description="Expiration date")
    quantity: int = Field(..., gt=0, description="Number of contracts")
    premium_credit: Optional[float] = Field(None, description="Premium received (for credit strategies)")
    
    @field_validator('expiration')
    @classmethod
    def validate_expiration_future(cls, v):
        if v <= date.today():
            raise ValueError('Expiration date must be in the future')
        return v

class RiskMetrics(BaseModel):
    """Model for trade risk assessment metrics"""
    max_profit: float = Field(..., description="Maximum potential profit")
    max_loss: float = Field(..., description="Maximum potential loss")
    breakeven: Union[float, List[float]] = Field(..., description="Breakeven price(s)")
    prob_profit: float = Field(..., ge=0, le=1, description="Probability of profit (0-1)")
    return_on_capital: float = Field(..., description="Return on capital percentage")
    
    @field_validator('prob_profit')
    @classmethod
    def validate_probability(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Probability must be between 0 and 1')
        return v

class ReasoningStep(BaseModel):
    """Model for individual analysis reasoning steps"""
    step: str = Field(..., description="Name of the analysis step")
    passed: bool = Field(..., description="Whether this step passed criteria")
    reasoning: str = Field(..., description="Detailed reasoning for this step")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score for this step")
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v

class TradeRecommendation(BaseModel):
    """Model for final trade recommendation"""
    should_trade: bool = Field(..., description="Whether to execute the trade")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence score")
    strategy: Optional[str] = Field(None, description="Recommended options strategy")
    contracts: List[Contract] = Field(default_factory=list, description="Specific contract recommendations")
    risk_metrics: Optional[RiskMetrics] = Field(None, description="Risk assessment metrics")
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list, description="Step-by-step analysis reasoning")
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v

class TradeAnalysisResponse(BaseModel):
    """Response model for trade analysis endpoint"""
    ticker: str = Field(..., description="Analyzed ticker symbol")
    company_name: str = Field(..., description="Company name")
    recommendation: TradeRecommendation = Field(..., description="Trade recommendation details")
    analysis_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When analysis was performed")
    
class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(default="healthy", description="Service status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Health check timestamp")
    services: dict = Field(default_factory=dict, description="Status of dependent services")

class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retry")