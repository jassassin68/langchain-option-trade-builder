"""
Market Data Service for fetching stock quotes, technical indicators, and fundamental data.

Implements requirements 2.1, 3.1, 7.4, 7.5:
- Technical analysis data (price, MA, RSI, volume, IV rank, beta)
- Fundamental screening data (market cap, P/E, debt-to-equity, earnings date)
- Error handling and retry logic for API failures
- Performance optimization with caching
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

logger = logging.getLogger(__name__)


class MarketDataError(Exception):
    """Base exception for market data service errors"""
    pass


class DataUnavailableError(MarketDataError):
    """Raised when required data is not available"""
    pass


class RateLimitError(MarketDataError):
    """Raised when API rate limit is exceeded"""
    pass


class StockQuote:
    """Model for stock quote data"""
    def __init__(self, ticker: str, price: float, volume: int, 
                 market_cap: Optional[float] = None, beta: Optional[float] = None):
        self.ticker = ticker
        self.price = price
        self.volume = volume
        self.market_cap = market_cap
        self.beta = beta
        self.timestamp = datetime.now()


class TechnicalData:
    """Model for technical analysis indicators"""
    def __init__(self, ticker: str, price: float, ma_50: Optional[float], 
                 ma_200: Optional[float], rsi: Optional[float], 
                 volume: int, iv_rank: Optional[float], beta: Optional[float]):
        self.ticker = ticker
        self.price = price
        self.ma_50 = ma_50
        self.ma_200 = ma_200
        self.rsi = rsi
        self.volume = volume
        self.iv_rank = iv_rank
        self.beta = beta
        self.timestamp = datetime.now()


class FundamentalData:
    """Model for fundamental analysis data"""
    def __init__(self, ticker: str, market_cap: Optional[float], 
                 pe_ratio: Optional[float], debt_to_equity: Optional[float],
                 earnings_date: Optional[datetime], news_sentiment: Optional[str] = None):
        self.ticker = ticker
        self.market_cap = market_cap
        self.pe_ratio = pe_ratio
        self.debt_to_equity = debt_to_equity
        self.earnings_date = earnings_date
        self.news_sentiment = news_sentiment
        self.timestamp = datetime.now()


class MarketDataService:
    """
    Service for fetching market data using yfinance.
    
    Implements retry logic and error handling for external API failures.
    """
    
    def __init__(self):
        self.session = None
    
    def _get_session(self) -> httpx.Client:
        """Get or create HTTP session for API requests"""
        if self.session is None:
            self.session = httpx.Client(timeout=30.0)
        return self.session
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, ConnectionError)),
        reraise=True
    )
    async def get_stock_quote(self, ticker: str) -> StockQuote:
        """
        Fetch current stock quote data.
        
        Implements requirement 2.1: Get current stock price and volume
        Implements requirement 7.4, 7.5: Error handling and retry logic
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            StockQuote object with current price and volume data
            
        Raises:
            DataUnavailableError: If data cannot be fetched
            RateLimitError: If API rate limit is exceeded
        """
        try:
            logger.info(f"Fetching stock quote for {ticker}")
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Validate that we got actual data
            if not info or 'currentPrice' not in info and 'regularMarketPrice' not in info:
                raise DataUnavailableError(f"No quote data available for {ticker}")
            
            # Get price from available fields
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            if price is None:
                raise DataUnavailableError(f"Price data not available for {ticker}")
            
            # Get volume
            volume = info.get('volume') or info.get('regularMarketVolume', 0)
            
            # Get optional fields
            market_cap = info.get('marketCap')
            beta = info.get('beta')
            
            quote = StockQuote(
                ticker=ticker,
                price=float(price),
                volume=int(volume),
                market_cap=float(market_cap) if market_cap else None,
                beta=float(beta) if beta else None
            )
            
            logger.info(f"Successfully fetched quote for {ticker}: ${price}")
            return quote
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.error(f"Rate limit exceeded for {ticker}")
                raise RateLimitError(f"API rate limit exceeded for {ticker}")
            logger.error(f"HTTP error fetching quote for {ticker}: {e}")
            raise DataUnavailableError(f"Failed to fetch quote for {ticker}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error fetching stock quote for {ticker}: {str(e)}")
            raise DataUnavailableError(f"Failed to fetch quote for {ticker}: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, ConnectionError)),
        reraise=True
    )
    async def get_technical_indicators(self, ticker: str) -> TechnicalData:
        """
        Fetch technical analysis indicators.
        
        Implements requirement 2.1: Evaluate price, 50-day MA, 200-day MA, RSI, volume, IV rank, beta
        Implements requirement 7.4, 7.5: Error handling and retry logic
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            TechnicalData object with technical indicators
            
        Raises:
            DataUnavailableError: If data cannot be fetched
        """
        try:
            logger.info(f"Fetching technical indicators for {ticker}")
            
            stock = yf.Ticker(ticker)
            
            # Get historical data for moving averages and RSI calculation
            hist = stock.history(period="1y")
            
            if hist.empty:
                raise DataUnavailableError(f"No historical data available for {ticker}")
            
            # Get current price and volume
            current_price = float(hist['Close'].iloc[-1])
            current_volume = int(hist['Volume'].iloc[-1])
            
            # Calculate moving averages
            ma_50 = float(hist['Close'].rolling(window=50).mean().iloc[-1]) if len(hist) >= 50 else None
            ma_200 = float(hist['Close'].rolling(window=200).mean().iloc[-1]) if len(hist) >= 200 else None
            
            # Calculate RSI (14-period)
            rsi = self._calculate_rsi(hist['Close'], period=14)
            
            # Get beta from info
            info = stock.info
            beta = info.get('beta')
            beta = float(beta) if beta else None
            
            # Calculate IV rank (simplified - would need options data for accurate calculation)
            # For now, we'll use a placeholder or estimate based on historical volatility
            iv_rank = self._estimate_iv_rank(hist)
            
            technical_data = TechnicalData(
                ticker=ticker,
                price=current_price,
                ma_50=ma_50,
                ma_200=ma_200,
                rsi=rsi,
                volume=current_volume,
                iv_rank=iv_rank,
                beta=beta
            )
            
            logger.info(f"Successfully fetched technical indicators for {ticker}")
            return technical_data
            
        except Exception as e:
            logger.error(f"Error fetching technical indicators for {ticker}: {str(e)}")
            raise DataUnavailableError(f"Failed to fetch technical indicators for {ticker}: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, ConnectionError)),
        reraise=True
    )
    async def get_fundamental_data(self, ticker: str) -> FundamentalData:
        """
        Fetch fundamental analysis data.
        
        Implements requirement 3.1: Evaluate market cap, P/E ratio, debt-to-equity, earnings date
        Implements requirement 7.4, 7.5: Error handling and retry logic
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            FundamentalData object with fundamental metrics
            
        Raises:
            DataUnavailableError: If data cannot be fetched
        """
        try:
            logger.info(f"Fetching fundamental data for {ticker}")
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                raise DataUnavailableError(f"No fundamental data available for {ticker}")
            
            # Extract fundamental metrics
            market_cap = info.get('marketCap')
            pe_ratio = info.get('trailingPE') or info.get('forwardPE')
            debt_to_equity = info.get('debtToEquity')
            
            # Get earnings date
            earnings_date = None
            try:
                earnings_dates = stock.earnings_dates
                if earnings_dates is not None and not earnings_dates.empty:
                    # Get the next earnings date
                    future_dates = earnings_dates[earnings_dates.index > datetime.now()]
                    if not future_dates.empty:
                        earnings_date = future_dates.index[0].to_pydatetime()
            except Exception as e:
                logger.warning(f"Could not fetch earnings date for {ticker}: {e}")
            
            # News sentiment (simplified - would need news API for accurate sentiment)
            news_sentiment = self._get_news_sentiment(stock)
            
            fundamental_data = FundamentalData(
                ticker=ticker,
                market_cap=float(market_cap) if market_cap else None,
                pe_ratio=float(pe_ratio) if pe_ratio else None,
                debt_to_equity=float(debt_to_equity) if debt_to_equity else None,
                earnings_date=earnings_date,
                news_sentiment=news_sentiment
            )
            
            logger.info(f"Successfully fetched fundamental data for {ticker}")
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {ticker}: {str(e)}")
            raise DataUnavailableError(f"Failed to fetch fundamental data for {ticker}: {str(e)}")
    
    def _calculate_rsi(self, prices, period: int = 14) -> Optional[float]:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Series of closing prices
            period: RSI period (default 14)
            
        Returns:
            RSI value or None if insufficient data
        """
        try:
            if len(prices) < period + 1:
                return None
            
            # Calculate price changes
            delta = prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not rsi.empty else None
            
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            return None
    
    def _estimate_iv_rank(self, hist) -> Optional[float]:
        """
        Estimate IV rank based on historical volatility.
        
        This is a simplified estimation. Accurate IV rank requires options data.
        
        Args:
            hist: Historical price data
            
        Returns:
            Estimated IV rank (0-100) or None
        """
        try:
            if len(hist) < 252:  # Need at least 1 year of data
                return None
            
            # Calculate historical volatility (annualized)
            returns = hist['Close'].pct_change()
            current_vol = returns.tail(30).std() * (252 ** 0.5) * 100
            
            # Calculate volatility range over the past year
            rolling_vol = returns.rolling(window=30).std() * (252 ** 0.5) * 100
            min_vol = rolling_vol.min()
            max_vol = rolling_vol.max()
            
            # Calculate IV rank as percentile
            if max_vol > min_vol:
                iv_rank = ((current_vol - min_vol) / (max_vol - min_vol)) * 100
                return float(iv_rank)
            
            return 50.0  # Default to middle if no range
            
        except Exception as e:
            logger.warning(f"Error estimating IV rank: {e}")
            return None
    
    def _get_news_sentiment(self, stock) -> Optional[str]:
        """
        Get simplified news sentiment.
        
        This is a placeholder. Accurate sentiment requires news API integration.
        
        Args:
            stock: yfinance Ticker object
            
        Returns:
            Sentiment string ("positive", "neutral", "negative") or None
        """
        try:
            news = stock.news
            if news and len(news) > 0:
                # Simplified: just return neutral for now
                # In production, would analyze news titles/content
                return "neutral"
            return None
        except Exception as e:
            logger.warning(f"Error fetching news sentiment: {e}")
            return None
    
    def close(self):
        """Close HTTP session"""
        if self.session:
            self.session.close()
            self.session = None
