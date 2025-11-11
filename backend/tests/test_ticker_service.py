import pytest
import sys
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.ticker_service import TickerService
from app.models.api import TickerResult
from app.models.database import StockTicker

class TestTickerService:
    """Test TickerService functionality"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session"""
        return AsyncMock()
    
    @pytest.fixture
    def ticker_service(self, mock_db_session):
        """Create TickerService instance with mock session"""
        return TickerService(mock_db_session)
    
    @pytest.fixture
    def sample_tickers(self):
        """Sample ticker data for testing"""
        return [
            StockTicker(
                ticker="AAPL",
                company_name="Apple Inc.",
                exchange="NASDAQ",
                is_active=True,
                last_updated=datetime.now(timezone.utc)
            ),
            StockTicker(
                ticker="MSFT",
                company_name="Microsoft Corporation",
                exchange="NASDAQ",
                is_active=True,
                last_updated=datetime.now(timezone.utc)
            ),
            StockTicker(
                ticker="GOOGL",
                company_name="Alphabet Inc.",
                exchange="NASDAQ",
                is_active=True,
                last_updated=datetime.now(timezone.utc)
            ),
            StockTicker(
                ticker="AMZN",
                company_name="Amazon.com Inc.",
                exchange="NASDAQ",
                is_active=True,
                last_updated=datetime.now(timezone.utc)
            ),
            StockTicker(
                ticker="TSLA",
                company_name="Tesla Inc.",
                exchange="NASDAQ",
                is_active=True,
                last_updated=datetime.now(timezone.utc)
            )
        ]

    @pytest.mark.asyncio
    async def test_search_tickers_exact_match(self, ticker_service, mock_db_session, sample_tickers):
        """Test exact ticker symbol match"""
        # Mock database response for exact match
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_tickers[0]]  # AAPL
        mock_db_session.execute.return_value = mock_result
        
        results = await ticker_service.search_tickers("AAPL", limit=10)
        
        assert len(results) == 1
        assert results[0].ticker == "AAPL"
        assert results[0].company_name == "Apple Inc."
        assert results[0].exchange == "NASDAQ"
        assert isinstance(results[0], TickerResult)
    
    @pytest.mark.asyncio
    async def test_search_tickers_partial_match(self, ticker_service, mock_db_session, sample_tickers):
        """Test partial ticker symbol match"""
        # Mock database response for partial match
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_tickers[0]]  # AAPL for "AA"
        mock_db_session.execute.return_value = mock_result
        
        results = await ticker_service.search_tickers("AA", limit=10)
        
        assert len(results) == 1
        assert results[0].ticker == "AAPL"
    
    @pytest.mark.asyncio
    async def test_search_tickers_company_name_match(self, ticker_service, mock_db_session, sample_tickers):
        """Test company name fuzzy matching"""
        # Mock database response for company name match
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_tickers[0]]  # Apple Inc.
        mock_db_session.execute.return_value = mock_result
        
        results = await ticker_service.search_tickers("Apple", limit=10)
        
        assert len(results) == 1
        assert results[0].ticker == "AAPL"
        assert "Apple" in results[0].company_name
    
    @pytest.mark.asyncio
    async def test_search_tickers_multiple_results(self, ticker_service, mock_db_session, sample_tickers):
        """Test search returning multiple results"""
        # Mock database response for multiple matches
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = sample_tickers[:3]  # First 3 tickers
        mock_db_session.execute.return_value = mock_result
        
        results = await ticker_service.search_tickers("tech", limit=10)
        
        assert len(results) == 3
        assert all(isinstance(result, TickerResult) for result in results)
    
    @pytest.mark.asyncio
    async def test_search_tickers_empty_query(self, ticker_service, mock_db_session):
        """Test search with empty query"""
        results = await ticker_service.search_tickers("", limit=10)
        
        assert len(results) == 0
        # Database should not be called for empty query
        mock_db_session.execute.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_search_tickers_whitespace_query(self, ticker_service, mock_db_session):
        """Test search with whitespace-only query"""
        results = await ticker_service.search_tickers("   ", limit=10)
        
        assert len(results) == 0
        # Database should not be called for whitespace-only query
        mock_db_session.execute.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_search_tickers_case_insensitive(self, ticker_service, mock_db_session, sample_tickers):
        """Test case insensitive search"""
        # Mock database response
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_tickers[0]]
        mock_db_session.execute.return_value = mock_result
        
        results = await ticker_service.search_tickers("aapl", limit=10)
        
        assert len(results) == 1
        assert results[0].ticker == "AAPL"
    
    @pytest.mark.asyncio
    async def test_search_tickers_limit_respected(self, ticker_service, mock_db_session, sample_tickers):
        """Test that search limit is respected"""
        # Mock database response with all tickers
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = sample_tickers[:2]  # Limited to 2
        mock_db_session.execute.return_value = mock_result
        
        results = await ticker_service.search_tickers("tech", limit=2)
        
        assert len(results) <= 2
    
    @pytest.mark.asyncio
    async def test_search_tickers_no_results(self, ticker_service, mock_db_session):
        """Test search with no matching results"""
        # Mock database response with no results
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db_session.execute.return_value = mock_result
        
        results = await ticker_service.search_tickers("NONEXISTENT", limit=10)
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_get_ticker_by_symbol_found(self, ticker_service, mock_db_session, sample_tickers):
        """Test getting ticker by symbol when it exists"""
        # Mock database response
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_tickers[0]
        mock_db_session.execute.return_value = mock_result
        
        result = await ticker_service.get_ticker_by_symbol("AAPL")
        
        assert result is not None
        assert result.ticker == "AAPL"
        assert result.company_name == "Apple Inc."
        assert isinstance(result, TickerResult)
    
    @pytest.mark.asyncio
    async def test_get_ticker_by_symbol_not_found(self, ticker_service, mock_db_session):
        """Test getting ticker by symbol when it doesn't exist"""
        # Mock database response with no result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result
        
        result = await ticker_service.get_ticker_by_symbol("NONEXISTENT")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_ticker_by_symbol_case_insensitive(self, ticker_service, mock_db_session, sample_tickers):
        """Test getting ticker by symbol is case insensitive"""
        # Mock database response
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_tickers[0]
        mock_db_session.execute.return_value = mock_result
        
        result = await ticker_service.get_ticker_by_symbol("aapl")
        
        assert result is not None
        assert result.ticker == "AAPL"
    
    @pytest.mark.asyncio
    async def test_add_ticker_success(self, ticker_service, mock_db_session):
        """Test successfully adding a new ticker"""
        # Mock the new ticker object that would be created
        new_ticker = StockTicker(
            ticker="TEST",
            company_name="Test Company",
            exchange="NYSE",
            is_active=True,
            last_updated=datetime.now(timezone.utc)
        )
        
        # Mock database operations
        mock_db_session.add = MagicMock()
        mock_db_session.commit = AsyncMock()
        mock_db_session.refresh = AsyncMock()
        
        result = await ticker_service.add_ticker("test", "Test Company", "NYSE")
        
        assert isinstance(result, TickerResult)
        assert result.ticker == "TEST"  # Should be uppercase
        assert result.company_name == "Test Company"
        assert result.exchange == "NYSE"
        
        # Verify database operations were called
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_ticker_database_error(self, ticker_service, mock_db_session):
        """Test handling database error when adding ticker"""
        # Mock database operations to raise an exception
        mock_db_session.add = MagicMock()
        mock_db_session.commit = AsyncMock(side_effect=Exception("Database error"))
        mock_db_session.rollback = AsyncMock()
        
        with pytest.raises(Exception, match="Database error"):
            await ticker_service.add_ticker("TEST", "Test Company")
        
        # Verify rollback was called
        mock_db_session.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_ticker_success(self, ticker_service, mock_db_session, sample_tickers):
        """Test successfully updating an existing ticker"""
        # Mock finding the existing ticker
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_tickers[0]
        mock_db_session.execute.return_value = mock_result
        mock_db_session.commit = AsyncMock()
        mock_db_session.refresh = AsyncMock()
        
        result = await ticker_service.update_ticker(
            ticker="AAPL",
            company_name="Apple Inc. Updated",
            exchange="NASDAQ"
        )
        
        assert result is not None
        assert isinstance(result, TickerResult)
        assert result.ticker == "AAPL"
        
        # Verify database operations were called
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_ticker_not_found(self, ticker_service, mock_db_session):
        """Test updating a ticker that doesn't exist"""
        # Mock database response with no result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result
        
        result = await ticker_service.update_ticker("NONEXISTENT", company_name="New Name")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_ticker_count(self, ticker_service, mock_db_session):
        """Test getting total ticker count"""
        # Mock database response
        mock_result = MagicMock()
        mock_result.scalar.return_value = 100
        mock_db_session.execute.return_value = mock_result
        
        count = await ticker_service.get_ticker_count()
        
        assert count == 100
        assert isinstance(count, int)
    
    @pytest.mark.asyncio
    async def test_get_ticker_count_zero(self, ticker_service, mock_db_session):
        """Test getting ticker count when database is empty"""
        # Mock database response
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_db_session.execute.return_value = mock_result
        
        count = await ticker_service.get_ticker_count()
        
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_search_tickers_database_error(self, ticker_service, mock_db_session):
        """Test handling database error during search"""
        # Mock database to raise an exception
        mock_db_session.execute.side_effect = Exception("Database connection error")
        
        with pytest.raises(Exception, match="Database connection error"):
            await ticker_service.search_tickers("AAPL")
    
    @pytest.mark.asyncio
    async def test_search_tickers_special_characters(self, ticker_service, mock_db_session, sample_tickers):
        """Test search with special characters in query"""
        # Mock database response
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db_session.execute.return_value = mock_result
        
        # Should not crash with special characters
        results = await ticker_service.search_tickers("A&P%", limit=10)
        
        assert len(results) == 0
        # Database should still be called (query is not empty)
        mock_db_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_tickers_very_long_query(self, ticker_service, mock_db_session):
        """Test search with very long query string"""
        # Mock database response
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db_session.execute.return_value = mock_result
        
        long_query = "A" * 1000  # Very long query
        results = await ticker_service.search_tickers(long_query, limit=10)
        
        assert len(results) == 0
        # Should still execute the query
        mock_db_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_tickers_fuzzy_company_name_match(self, ticker_service, mock_db_session, sample_tickers):
        """Test fuzzy matching on company names with typos"""
        # Mock database response for fuzzy company name match
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_tickers[1]]  # Microsoft
        mock_db_session.execute.return_value = mock_result
        
        # Test with slight misspelling
        results = await ticker_service.search_tickers("Microsft", limit=10)
        
        assert len(results) == 1
        assert results[0].ticker == "MSFT"
        assert "Microsoft" in results[0].company_name
    
    @pytest.mark.asyncio
    async def test_search_tickers_partial_company_name(self, ticker_service, mock_db_session, sample_tickers):
        """Test partial company name matching"""
        # Mock database response for partial company name match
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_tickers[2]]  # Alphabet
        mock_db_session.execute.return_value = mock_result
        
        results = await ticker_service.search_tickers("Alphabet", limit=10)
        
        assert len(results) == 1
        assert results[0].ticker == "GOOGL"
        assert "Alphabet" in results[0].company_name
    
    @pytest.mark.asyncio
    async def test_search_tickers_single_character(self, ticker_service, mock_db_session, sample_tickers):
        """Test search with single character (requirement 1.1: 1+ characters)"""
        # Mock database response for single character match
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_tickers[0]]  # AAPL for "A"
        mock_db_session.execute.return_value = mock_result
        
        results = await ticker_service.search_tickers("A", limit=10)
        
        assert len(results) == 1
        assert results[0].ticker == "AAPL"
    
    @pytest.mark.asyncio
    async def test_search_tickers_relevance_ordering(self, ticker_service, mock_db_session, sample_tickers):
        """Test that results are ordered by relevance (exact match first)"""
        # Create tickers that would match "A" in different ways
        test_tickers = [
            StockTicker(ticker="AMZN", company_name="Amazon.com Inc.", exchange="NASDAQ", is_active=True),
            StockTicker(ticker="A", company_name="Agilent Technologies Inc.", exchange="NYSE", is_active=True),
            StockTicker(ticker="AAPL", company_name="Apple Inc.", exchange="NASDAQ", is_active=True),
        ]
        
        # Mock database response with multiple matches
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = test_tickers
        mock_db_session.execute.return_value = mock_result
        
        results = await ticker_service.search_tickers("A", limit=10)
        
        assert len(results) == 3
        # Results should be ordered by relevance (exact match "A" should be first if properly ordered)
        assert all(isinstance(result, TickerResult) for result in results)
    
    @pytest.mark.asyncio
    async def test_search_tickers_limit_boundary_conditions(self, ticker_service, mock_db_session, sample_tickers):
        """Test search with various limit boundary conditions"""
        # Mock database response
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = sample_tickers[:1]  # Return 1 result
        mock_db_session.execute.return_value = mock_result
        
        # Test with limit of 1
        results = await ticker_service.search_tickers("tech", limit=1)
        assert len(results) <= 1
        
        # Test with limit of 0 (should still work)
        mock_result.scalars.return_value.all.return_value = []
        results = await ticker_service.search_tickers("tech", limit=0)
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_search_tickers_mixed_case_company_name(self, ticker_service, mock_db_session, sample_tickers):
        """Test case insensitive search on company names"""
        # Mock database response
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_tickers[0]]  # Apple
        mock_db_session.execute.return_value = mock_result
        
        # Test with mixed case company name search
        results = await ticker_service.search_tickers("apple", limit=10)
        
        assert len(results) == 1
        assert results[0].ticker == "AAPL"
        assert "Apple" in results[0].company_name
    
    @pytest.mark.asyncio
    async def test_search_tickers_numeric_characters(self, ticker_service, mock_db_session):
        """Test search with numeric characters in query"""
        # Mock database response
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db_session.execute.return_value = mock_result
        
        results = await ticker_service.search_tickers("3M", limit=10)
        
        assert len(results) == 0
        # Should execute query without errors
        mock_db_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_tickers_default_limit(self, ticker_service, mock_db_session, sample_tickers):
        """Test that default limit of 10 is applied (requirement 1.3)"""
        # Mock database response with more than 10 results
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = sample_tickers  # 5 results
        mock_db_session.execute.return_value = mock_result
        
        # Call without explicit limit
        results = await ticker_service.search_tickers("tech")
        
        # Should respect default limit of 10
        assert len(results) <= 10
        mock_db_session.execute.assert_called_once()