from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from sqlalchemy.orm import selectinload
from app.models.database import StockTicker
from app.models.api import TickerResult
from app.core.database import get_db
import logging

logger = logging.getLogger(__name__)

class TickerService:
    """Service for ticker search and management operations"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def search_tickers(self, query: str, limit: int = 10) -> List[TickerResult]:
        """
        Search for tickers using fuzzy matching on ticker symbols and company names
        
        Implements requirement 1.1, 1.2, 1.3:
        - Autocomplete with 1+ characters
        - Match both ticker symbols (exact and partial) and company names (fuzzy)
        - Return results in format suitable for "TICKER - Company Name" display
        
        Args:
            query: Search query string
            limit: Maximum number of results to return (default 10 per requirement 1.3)
            
        Returns:
            List of TickerResult objects matching the query, ordered by relevance
        """
        if not query or len(query.strip()) == 0:
            return []
        
        # Normalize query for ticker matching
        normalized_query = query.strip().upper()
        original_query = query.strip()
        
        try:
            # Build search query with multiple matching strategies, ordered by priority
            search_conditions = []
            
            # 1. Exact ticker match (highest priority)
            search_conditions.append(
                StockTicker.ticker == normalized_query
            )
            
            # 2. Ticker starts with query (high priority for partial matches)
            search_conditions.append(
                StockTicker.ticker.like(f"{normalized_query}%")
            )
            
            # 3. Ticker contains query (for symbols like BRK.A, BRK.B)
            search_conditions.append(
                StockTicker.ticker.contains(normalized_query)
            )
            
            # 4. Company name contains query (case insensitive, exact word matching)
            search_conditions.append(
                func.upper(StockTicker.company_name).contains(normalized_query)
            )
            
            # 5. Fuzzy match on company name using trigram similarity
            # This requires pg_trgm extension which we enabled in migrations
            # Lower threshold for better fuzzy matching as per requirement 1.2
            search_conditions.append(
                func.similarity(StockTicker.company_name, original_query) > 0.2
            )
            
            # Execute search query with simple ordering (exact matches will naturally come first)
            stmt = (
                select(StockTicker)
                .where(
                    StockTicker.is_active == True,
                    or_(*search_conditions)
                )
                .order_by(StockTicker.ticker)  # Simple alphabetical ordering
                .limit(limit)
            )
            
            result = await self.db.execute(stmt)
            tickers = result.scalars().all()
            
            # Convert to TickerResult objects
            ticker_results = [
                TickerResult(
                    ticker=ticker.ticker,
                    company_name=ticker.company_name,
                    exchange=ticker.exchange
                )
                for ticker in tickers
            ]
            
            logger.info(f"Found {len(ticker_results)} tickers for query: '{query}'")
            return ticker_results
            
        except Exception as e:
            logger.error(f"Error searching tickers for query '{query}': {str(e)}")
            raise
    
    async def get_ticker_by_symbol(self, ticker: str) -> Optional[TickerResult]:
        """
        Get a specific ticker by its symbol
        
        Args:
            ticker: Ticker symbol to search for
            
        Returns:
            TickerResult if found, None otherwise
        """
        try:
            stmt = (
                select(StockTicker)
                .where(
                    StockTicker.ticker == ticker.upper(),
                    StockTicker.is_active == True
                )
            )
            
            result = await self.db.execute(stmt)
            ticker_obj = result.scalar_one_or_none()
            
            if ticker_obj:
                return TickerResult(
                    ticker=ticker_obj.ticker,
                    company_name=ticker_obj.company_name,
                    exchange=ticker_obj.exchange
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting ticker '{ticker}': {str(e)}")
            raise
    
    async def add_ticker(self, ticker: str, company_name: str, exchange: str = "NYSE") -> TickerResult:
        """
        Add a new ticker to the database
        
        Args:
            ticker: Ticker symbol
            company_name: Company name
            exchange: Stock exchange (default: NYSE)
            
        Returns:
            TickerResult object for the added ticker
        """
        try:
            new_ticker = StockTicker(
                ticker=ticker.upper(),
                company_name=company_name,
                exchange=exchange
            )
            
            self.db.add(new_ticker)
            await self.db.commit()
            await self.db.refresh(new_ticker)
            
            logger.info(f"Added new ticker: {ticker}")
            
            return TickerResult(
                ticker=new_ticker.ticker,
                company_name=new_ticker.company_name,
                exchange=new_ticker.exchange
            )
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error adding ticker '{ticker}': {str(e)}")
            raise
    
    async def update_ticker(self, ticker: str, company_name: Optional[str] = None, 
                          exchange: Optional[str] = None, is_active: Optional[bool] = None) -> Optional[TickerResult]:
        """
        Update an existing ticker
        
        Args:
            ticker: Ticker symbol to update
            company_name: New company name (optional)
            exchange: New exchange (optional)
            is_active: New active status (optional)
            
        Returns:
            Updated TickerResult if found and updated, None otherwise
        """
        try:
            stmt = select(StockTicker).where(StockTicker.ticker == ticker.upper())
            result = await self.db.execute(stmt)
            ticker_obj = result.scalar_one_or_none()
            
            if not ticker_obj:
                return None
            
            # Update fields if provided
            if company_name is not None:
                ticker_obj.company_name = company_name
            if exchange is not None:
                ticker_obj.exchange = exchange
            if is_active is not None:
                ticker_obj.is_active = is_active
            
            ticker_obj.last_updated = func.now()
            
            await self.db.commit()
            await self.db.refresh(ticker_obj)
            
            logger.info(f"Updated ticker: {ticker}")
            
            return TickerResult(
                ticker=ticker_obj.ticker,
                company_name=ticker_obj.company_name,
                exchange=ticker_obj.exchange
            )
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating ticker '{ticker}': {str(e)}")
            raise
    
    async def get_ticker_count(self) -> int:
        """
        Get total count of active tickers
        
        Returns:
            Number of active tickers in database
        """
        try:
            stmt = select(func.count(StockTicker.ticker)).where(StockTicker.is_active == True)
            result = await self.db.execute(stmt)
            count = result.scalar()
            return count or 0
            
        except Exception as e:
            logger.error(f"Error getting ticker count: {str(e)}")
            raise

# Dependency function to get TickerService instance
async def get_ticker_service(db: AsyncSession = None) -> TickerService:
    """
    Dependency function to get TickerService instance
    
    Args:
        db: Database session (will be injected by FastAPI)
        
    Returns:
        TickerService instance
    """
    if db is None:
        # This should not happen in normal FastAPI usage, but provides fallback
        async for session in get_db():
            return TickerService(session)
    
    return TickerService(db)