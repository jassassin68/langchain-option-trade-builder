"""
Ticker search API endpoints.

Implements requirements 1.1, 1.2, 1.3, 1.4, 7.2:
- GET /api/v1/tickers/search with query parameter validation
- Autocomplete with 1+ characters
- Match ticker symbols and company names
- Return formatted results with proper error handling
- Fast response time (<300ms per requirement 7.2)
"""

from fastapi import APIRouter, Query, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated
import logging

from app.models.api import TickerSearchResponse, ErrorResponse
from app.services.ticker_service import TickerService
from app.core.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tickers", tags=["tickers"])


@router.get(
    "/search",
    response_model=TickerSearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Search for stock tickers",
    description="""
    Search for stock tickers with autocomplete functionality.
    
    - Matches both ticker symbols (exact and partial) and company names (fuzzy match)
    - Returns results in format suitable for "TICKER - Company Name" display
    - Debounced on frontend with 300ms delay (requirement 1.4)
    - Responds within 300ms (requirement 7.2)
    """
)
async def search_tickers(
    q: Annotated[str, Query(
        min_length=1,
        max_length=50,
        description="Search query (minimum 1 character)",
        example="AAPL"
    )],
    limit: Annotated[int, Query(
        ge=1,
        le=50,
        description="Maximum number of results to return (default 10)",
        example=10
    )] = 10,
    db: AsyncSession = Depends(get_db)
) -> TickerSearchResponse:
    """
    Search for stock tickers with autocomplete functionality.
    
    Implements requirement 1.1: Display autocomplete dropdown when user types 1+ characters
    Implements requirement 1.2: Match ticker symbols and company names
    Implements requirement 1.3: Show format "TICKER - Company Name" and limit to 10 suggestions
    Implements requirement 7.2: Respond within 300ms
    
    Args:
        q: Search query string (minimum 1 character)
        limit: Maximum number of results (default 10, max 50)
        db: Database session (injected)
    
    Returns:
        TickerSearchResponse with matching results
    
    Raises:
        HTTPException: 400 if query is invalid, 500 if search fails
    """
    try:
        # Validate query
        query = q.strip()
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query cannot be empty"
            )
        
        # Create ticker service and perform search
        ticker_service = TickerService(db)
        results = await ticker_service.search_tickers(query, limit=limit)
        
        logger.info(f"Ticker search for '{query}' returned {len(results)} results")
        
        return TickerSearchResponse(
            results=results,
            count=len(results)
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error searching tickers for query '{q}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while searching for tickers"
        )
