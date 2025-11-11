import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.market_data_service import (
    MarketDataService,
    StockQuote,
    TechnicalData,
    FundamentalData,
    DataUnavailableError,
    RateLimitError
)


class TestMarketDataService:
    """Test MarketDataService functionality"""
    
    @pytest.fixture
    def market_service(self):
        """Create MarketDataService instance"""
        return MarketDataService()
    
    @pytest.fixture
    def mock_stock_info(self):
        """Mock stock info data"""
        return {
            'currentPrice': 150.50,
            'regularMarketPrice': 150.50,
            'volume': 1000000,
            'regularMarketVolume': 1000000,
            'marketCap': 2500000000000,
            'beta': 1.2,
            'trailingPE': 25.5,
            'forwardPE': 22.0,
            'debtToEquity': 150.0
        }
    
    @pytest.fixture
    def mock_historical_data(self):
        """Mock historical price data"""
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        data = {
            'Close': [100 + i * 0.5 for i in range(252)],
            'Volume': [1000000 + i * 1000 for i in range(252)]
        }
        return pd.DataFrame(data, index=dates)
    
    @pytest.mark.asyncio
    async def test_get_stock_quote_success(self, market_service, mock_stock_info):
        """Test successfully fetching stock quote"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.info = mock_stock_info
            
            quote = await market_service.get_stock_quote("AAPL")
            
            assert isinstance(quote, StockQuote)
            assert quote.ticker == "AAPL"
            assert quote.price == 150.50
            assert quote.volume == 1000000
            assert quote.market_cap == 2500000000000
            assert quote.beta == 1.2
    
    @pytest.mark.asyncio
    async def test_get_stock_quote_no_data(self, market_service):
        """Test handling when no quote data is available"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.info = {}
            
            with pytest.raises(DataUnavailableError, match="No quote data available"):
                await market_service.get_stock_quote("INVALID")
    
    @pytest.mark.asyncio
    async def test_get_stock_quote_missing_price(self, market_service):
        """Test handling when price data is missing"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.info = {'volume': 1000000}
            
            with pytest.raises(DataUnavailableError, match="No quote data available"):
                await market_service.get_stock_quote("AAPL")
    
    @pytest.mark.asyncio
    async def test_get_technical_indicators_success(self, market_service, mock_stock_info, mock_historical_data):
        """Test successfully fetching technical indicators"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.info = mock_stock_info
            mock_ticker.return_value.history.return_value = mock_historical_data
            
            technical = await market_service.get_technical_indicators("AAPL")
            
            assert isinstance(technical, TechnicalData)
            assert technical.ticker == "AAPL"
            assert technical.price > 0
            assert technical.volume > 0
            assert technical.ma_50 is not None
            assert technical.ma_200 is not None
            assert technical.rsi is not None
            assert technical.beta == 1.2
    
    @pytest.mark.asyncio
    async def test_get_technical_indicators_no_history(self, market_service):
        """Test handling when no historical data is available"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            
            with pytest.raises(DataUnavailableError, match="No historical data available"):
                await market_service.get_technical_indicators("INVALID")
    
    @pytest.mark.asyncio
    async def test_get_fundamental_data_success(self, market_service, mock_stock_info):
        """Test successfully fetching fundamental data"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.info = mock_stock_info
            mock_ticker.return_value.earnings_dates = pd.DataFrame(
                index=pd.DatetimeIndex([datetime.now() + timedelta(days=30)])
            )
            mock_ticker.return_value.news = []
            
            fundamental = await market_service.get_fundamental_data("AAPL")
            
            assert isinstance(fundamental, FundamentalData)
            assert fundamental.ticker == "AAPL"
            assert fundamental.market_cap == 2500000000000
            assert fundamental.pe_ratio == 25.5
            assert fundamental.debt_to_equity == 150.0
    
    @pytest.mark.asyncio
    async def test_get_fundamental_data_no_info(self, market_service):
        """Test handling when no fundamental data is available"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.info = None
            
            with pytest.raises(DataUnavailableError, match="No fundamental data available"):
                await market_service.get_fundamental_data("INVALID")
    
    @pytest.mark.asyncio
    async def test_calculate_rsi(self, market_service, mock_historical_data):
        """Test RSI calculation"""
        rsi = market_service._calculate_rsi(mock_historical_data['Close'], period=14)
        
        assert rsi is not None
        assert 0 <= rsi <= 100
    
    @pytest.mark.asyncio
    async def test_calculate_rsi_insufficient_data(self, market_service):
        """Test RSI calculation with insufficient data"""
        short_data = pd.Series([100, 101, 102])
        rsi = market_service._calculate_rsi(short_data, period=14)
        
        assert rsi is None
    
    @pytest.mark.asyncio
    async def test_estimate_iv_rank(self, market_service, mock_historical_data):
        """Test IV rank estimation"""
        iv_rank = market_service._estimate_iv_rank(mock_historical_data)
        
        assert iv_rank is not None
        assert 0 <= iv_rank <= 100
    
    @pytest.mark.asyncio
    async def test_estimate_iv_rank_insufficient_data(self, market_service):
        """Test IV rank estimation with insufficient data"""
        short_hist = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200]
        })
        iv_rank = market_service._estimate_iv_rank(short_hist)
        
        assert iv_rank is None
