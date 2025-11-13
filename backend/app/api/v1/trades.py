"""
Trade analysis API endpoints.

Implements requirements 6.1, 6.2, 6.3, 6.4, 7.1, 7.4, 7.5:
- POST /api/v1/trades/analyze with request validation
- Integration with OptionsEvaluationAgent for complete analysis workflow
- Comprehensive error handling for all failure scenarios
- Complete analysis within 5 seconds average (requirement 7.1)
- Graceful handling of API failures (requirement 7.4, 7.5)
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from datetime import datetime, timezone

from app.models.api import (
    TradeAnalysisRequest,
    TradeAnalysisResponse,
    ErrorResponse
)
from app.chains.options_evaluation_agent import OptionsEvaluationAgent
from app.services.ticker_service import TickerService
from app.services.market_data_service import DataUnavailableError
from app.services.options_data_service import OptionsUnavailableError
from app.core.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trades", tags=["trades"])


@router.post(
    "/analyze",
    response_model=TradeAnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Ticker not found"},
        422: {"model": ErrorResponse, "description": "Data unavailable"},
        503: {"model": ErrorResponse, "description": "External service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Analyze options trade opportunity",
    description="""
    Perform comprehensive options trade analysis for a given ticker.
    
    - Executes sequential analysis: technical, fundamental, options, strategy, risk
    - Returns actionable trade recommendation with confidence score
    - Includes detailed reasoning steps and risk metrics
    - Completes within 5 seconds average (requirement 7.1)
    """
)
async def analyze_trade(
    request: TradeAnalysisRequest,
    db: AsyncSession = Depends(get_db)
) -> TradeAnalysisResponse:
    """
    Analyze options trade opportunity for a given ticker.
    
    Implements requirement 6.1: Calculate max profit, max loss, breakeven, probability, ROC
    Implements requirement 6.2: Provide specific contract details
    Implements requirement 6.3: Include confidence score
    Implements requirement 6.4: Return clear YES/NO recommendation with reasoning
    Implements requirement 7.1: Complete within 5 seconds average
    Implements requirement 7.4, 7.5: Handle API failures gracefully
    
    Args:
        request: TradeAnalysisRequest with ticker to analyze
        db: Database session (injected)
    
    Returns:
        TradeAnalysisResponse with complete analysis and recommendation
    
    Raises:
        HTTPException: Various status codes based on error type
    """
    ticker = request.ticker
    
    try:
        logger.info(f"Starting trade analysis for ticker: {ticker}")
        
        # Step 1: Verify ticker exists in database
        ticker_service = TickerService(db)
        ticker_info = await ticker_service.get_ticker_by_symbol(ticker)
        
        if not ticker_info:
            logger.warning(f"Ticker not found: {ticker}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ticker '{ticker}' not found in database"
            )
        
        # Step 2: Initialize and run options evaluation agent
        agent = OptionsEvaluationAgent()
        
        try:
            recommendation = await agent.evaluate_trade(ticker)
            
            # Log execution times for monitoring
            execution_times = agent.get_execution_times()
            logger.info(
                f"Trade analysis complete for {ticker}: "
                f"should_trade={recommendation.should_trade}, "
                f"confidence={recommendation.confidence:.2f}, "
                f"total_time={execution_times.get('total', 0):.2f}s"
            )
            
            # Return complete analysis response
            return TradeAnalysisResponse(
                ticker=ticker,
                company_name=ticker_info.company_name,
                recommendation=recommendation,
                analysis_timestamp=datetime.now(timezone.utc)
            )
        
        finally:
            # Clean up agent resources
            agent.close()
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except DataUnavailableError as e:
        logger.error(f"Market data unavailable for {ticker}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Market data unavailable for ticker '{ticker}'. Please try again later."
        )
    
    except OptionsUnavailableError as e:
        logger.error(f"Options data unavailable for {ticker}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Options data unavailable for ticker '{ticker}'. This stock may not have tradable options."
        )
    
    except Exception as e:
        logger.error(f"Unexpected error analyzing trade for {ticker}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during trade analysis. Please try again."
        )
