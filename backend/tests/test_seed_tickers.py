import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.seed_tickers import seed_tickers, NYSE_TICKERS
from app.services.ticker_service import TickerService
from app.models.database import StockTicker

class TestSeedTickers:
    """Test ticker seeding functionality"""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock database session"""
        return AsyncMock()
    
    @pytest.fixture
    def mock_ticker_service(self, mock_session):
        """Create a mock ticker service"""
        service = TickerService(mock_session)
        service.get_ticker_by_symbol = AsyncMock()
        service.add_ticker = AsyncMock()
        service.update_ticker = AsyncMock()
        return service
    
    @pytest.mark.asyncio
    async def test_seed_tickers_with_new_tickers(self, mock_session):
        """Test seeding with all new tickers"""
        # Mock ticker service methods
        with patch('scripts.seed_tickers.TickerService') as MockTickerService:
            mock_service = AsyncMock()
            MockTickerService.return_value = mock_service
            
            # Mock that no tickers exist (all are new)
            mock_service.get_ticker_by_symbol.return_value = None
            mock_service.add_ticker.return_value = AsyncMock()
            
            # Test with a small subset of tickers
            test_tickers = [
                ("AAPL", "Apple Inc.", "NASDAQ"),
                ("MSFT", "Microsoft Corporation", "NASDAQ"),
                ("GOOGL", "Alphabet Inc.", "NASDAQ")
            ]
            
            await seed_tickers(mock_session, test_tickers)
            
            # Verify that add_ticker was called for each ticker
            assert mock_service.add_ticker.call_count == 3
            
            # Verify the calls were made with correct parameters
            mock_service.add_ticker.assert_any_call(
                ticker="AAPL",
                company_name="Apple Inc.",
                exchange="NASDAQ"
            )
            mock_service.add_ticker.assert_any_call(
                ticker="MSFT", 
                company_name="Microsoft Corporation",
                exchange="NASDAQ"
            )
            mock_service.add_ticker.assert_any_call(
                ticker="GOOGL",
                company_name="Alphabet Inc.",
                exchange="NASDAQ"
            )
    
    @pytest.mark.asyncio
    async def test_seed_tickers_with_existing_tickers(self, mock_session):
        """Test seeding with existing tickers that need updates"""
        with patch('scripts.seed_tickers.TickerService') as MockTickerService:
            mock_service = AsyncMock()
            MockTickerService.return_value = mock_service
            
            # Mock existing ticker with different company name
            existing_ticker = StockTicker(
                ticker="AAPL",
                company_name="Apple Computer Inc.",  # Old name
                exchange="NASDAQ",
                is_active=True
            )
            
            mock_service.get_ticker_by_symbol.return_value = existing_ticker
            mock_service.update_ticker.return_value = AsyncMock()
            
            test_tickers = [("AAPL", "Apple Inc.", "NASDAQ")]  # New name
            
            await seed_tickers(mock_session, test_tickers)
            
            # Verify update was called
            mock_service.update_ticker.assert_called_once_with(
                ticker="AAPL",
                company_name="Apple Inc.",
                exchange="NASDAQ",
                is_active=True
            )
            
            # Verify add was not called
            mock_service.add_ticker.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_seed_tickers_with_unchanged_existing_tickers(self, mock_session):
        """Test seeding with existing tickers that don't need updates"""
        with patch('scripts.seed_tickers.TickerService') as MockTickerService:
            mock_service = AsyncMock()
            MockTickerService.return_value = mock_service
            
            # Mock existing ticker with same data
            existing_ticker = StockTicker(
                ticker="AAPL",
                company_name="Apple Inc.",
                exchange="NASDAQ",
                is_active=True
            )
            
            mock_service.get_ticker_by_symbol.return_value = existing_ticker
            
            test_tickers = [("AAPL", "Apple Inc.", "NASDAQ")]  # Same data
            
            await seed_tickers(mock_session, test_tickers)
            
            # Verify neither add nor update was called
            mock_service.add_ticker.assert_not_called()
            mock_service.update_ticker.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_seed_tickers_handles_errors(self, mock_session):
        """Test that seeding handles errors gracefully"""
        with patch('scripts.seed_tickers.TickerService') as MockTickerService:
            mock_service = AsyncMock()
            MockTickerService.return_value = mock_service
            
            # Mock that get_ticker_by_symbol raises an exception
            mock_service.get_ticker_by_symbol.side_effect = Exception("Database error")
            
            test_tickers = [("AAPL", "Apple Inc.", "NASDAQ")]
            
            # Should not raise exception, but handle it gracefully
            await seed_tickers(mock_session, test_tickers)
            
            # Verify the method was called despite the error
            mock_service.get_ticker_by_symbol.assert_called_once()
    
    def test_nyse_tickers_data_format(self):
        """Test that NYSE_TICKERS data is properly formatted"""
        # Verify we have ticker data
        assert len(NYSE_TICKERS) > 0
        
        # Verify each ticker entry has the correct format
        for ticker_data in NYSE_TICKERS:
            assert len(ticker_data) == 3  # ticker, company_name, exchange
            ticker, company_name, exchange = ticker_data
            
            # Verify ticker format
            assert isinstance(ticker, str)
            assert len(ticker) >= 1 and len(ticker) <= 5
            assert ticker.isupper()
            
            # Verify company name
            assert isinstance(company_name, str)
            assert len(company_name) > 0
            
            # Verify exchange
            assert isinstance(exchange, str)
            assert exchange in ["NYSE", "NASDAQ", "CBOE"]
    
    def test_nyse_tickers_contains_major_stocks(self):
        """Test that NYSE_TICKERS contains expected major stocks"""
        ticker_symbols = [ticker[0] for ticker in NYSE_TICKERS]
        
        # Check for some major stocks that should be included
        expected_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "JNJ"]
        
        for expected_ticker in expected_tickers:
            assert expected_ticker in ticker_symbols, f"Expected ticker {expected_ticker} not found in NYSE_TICKERS"
    
    def test_nyse_tickers_no_duplicates(self):
        """Test that NYSE_TICKERS contains no duplicate ticker symbols"""
        ticker_symbols = [ticker[0] for ticker in NYSE_TICKERS]
        
        # Check for duplicates
        unique_tickers = set(ticker_symbols)
        assert len(ticker_symbols) == len(unique_tickers), "Duplicate ticker symbols found in NYSE_TICKERS"
    
    def test_nyse_tickers_covers_different_sectors(self):
        """Test that NYSE_TICKERS covers different market sectors"""
        company_names = [ticker[1].lower() for ticker in NYSE_TICKERS]
        
        # Check for representation from different sectors
        tech_keywords = ["apple", "microsoft", "alphabet", "amazon", "tesla", "nvidia"]
        financial_keywords = ["jpmorgan", "bank", "goldman", "morgan stanley"]
        healthcare_keywords = ["johnson", "pfizer", "abbott", "merck"]
        
        tech_count = sum(1 for name in company_names if any(keyword in name for keyword in tech_keywords))
        financial_count = sum(1 for name in company_names if any(keyword in name for keyword in financial_keywords))
        healthcare_count = sum(1 for name in company_names if any(keyword in name for keyword in healthcare_keywords))
        
        # Verify we have representation from multiple sectors
        assert tech_count > 0, "No technology companies found"
        assert financial_count > 0, "No financial companies found"
        assert healthcare_count > 0, "No healthcare companies found"