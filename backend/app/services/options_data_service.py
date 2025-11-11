"""
Options Data Service for fetching options chain data and calculating Greeks.

Implements requirements 4.1, 4.2, 4.3, 4.4, 4.6:
- Options chain fetching for next 2 expiration cycles
- Open interest and bid-ask spread validation
- Days to expiration constraints (30-60 days)
- Greeks calculation methods
- Data validation for options contract quality checks
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx
import math

logger = logging.getLogger(__name__)


class OptionsDataError(Exception):
    """Base exception for options data service errors"""
    pass


class OptionsUnavailableError(OptionsDataError):
    """Raised when options data is not available"""
    pass


class OptionsContract:
    """Model for options contract data"""
    def __init__(self, ticker: str, contract_type: str, strike: float, 
                 expiration: date, bid: float, ask: float, last: float,
                 volume: int, open_interest: int, implied_volatility: Optional[float] = None):
        self.ticker = ticker
        self.contract_type = contract_type  # 'CALL' or 'PUT'
        self.strike = strike
        self.expiration = expiration
        self.bid = bid
        self.ask = ask
        self.last = last
        self.volume = volume
        self.open_interest = open_interest
        self.implied_volatility = implied_volatility
        self.mid_price = (bid + ask) / 2 if bid and ask else last
        self.bid_ask_spread = ask - bid if bid and ask else 0
        self.spread_percentage = (self.bid_ask_spread / self.mid_price * 100) if self.mid_price > 0 else 0
        self.days_to_expiration = (expiration - date.today()).days


class OptionsChain:
    """Model for complete options chain data"""
    def __init__(self, ticker: str, expiration_dates: List[date], 
                 calls: List[OptionsContract], puts: List[OptionsContract]):
        self.ticker = ticker
        self.expiration_dates = expiration_dates
        self.calls = calls
        self.puts = puts
        self.timestamp = datetime.now()


class Greeks:
    """Model for options Greeks"""
    def __init__(self, delta: Optional[float] = None, gamma: Optional[float] = None,
                 theta: Optional[float] = None, vega: Optional[float] = None,
                 rho: Optional[float] = None):
        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega
        self.rho = rho


class OptionsDataService:
    """
    Service for fetching options data and calculating Greeks.
    
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
    async def get_options_chain(self, ticker: str, max_expirations: int = 2) -> OptionsChain:
        """
        Fetch options chain data for the next expiration cycles.
        
        Implements requirement 4.1: Analyze options chain data for next 2 expiration cycles
        
        Args:
            ticker: Stock ticker symbol
            max_expirations: Maximum number of expiration dates to fetch (default 2)
            
        Returns:
            OptionsChain object with calls and puts data
            
        Raises:
            OptionsUnavailableError: If options data cannot be fetched
        """
        try:
            logger.info(f"Fetching options chain for {ticker}")
            
            stock = yf.Ticker(ticker)
            
            # Get available expiration dates
            expirations = stock.options
            if not expirations or len(expirations) == 0:
                raise OptionsUnavailableError(f"No options available for {ticker}")
            
            # Limit to requested number of expirations
            expirations_to_fetch = expirations[:max_expirations]
            
            all_calls = []
            all_puts = []
            expiration_dates = []
            
            for exp_date_str in expirations_to_fetch:
                try:
                    # Parse expiration date
                    exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                    expiration_dates.append(exp_date)
                    
                    # Get options chain for this expiration
                    opt_chain = stock.option_chain(exp_date_str)
                    
                    # Process calls
                    if opt_chain.calls is not None and not opt_chain.calls.empty:
                        for _, row in opt_chain.calls.iterrows():
                            contract = OptionsContract(
                                ticker=ticker,
                                contract_type='CALL',
                                strike=float(row['strike']),
                                expiration=exp_date,
                                bid=float(row.get('bid', 0)),
                                ask=float(row.get('ask', 0)),
                                last=float(row.get('lastPrice', 0)),
                                volume=int(row.get('volume', 0)),
                                open_interest=int(row.get('openInterest', 0)),
                                implied_volatility=float(row.get('impliedVolatility', 0)) if row.get('impliedVolatility') else None
                            )
                            all_calls.append(contract)
                    
                    # Process puts
                    if opt_chain.puts is not None and not opt_chain.puts.empty:
                        for _, row in opt_chain.puts.iterrows():
                            contract = OptionsContract(
                                ticker=ticker,
                                contract_type='PUT',
                                strike=float(row['strike']),
                                expiration=exp_date,
                                bid=float(row.get('bid', 0)),
                                ask=float(row.get('ask', 0)),
                                last=float(row.get('lastPrice', 0)),
                                volume=int(row.get('volume', 0)),
                                open_interest=int(row.get('openInterest', 0)),
                                implied_volatility=float(row.get('impliedVolatility', 0)) if row.get('impliedVolatility') else None
                            )
                            all_puts.append(contract)
                    
                except Exception as e:
                    logger.warning(f"Error processing expiration {exp_date_str}: {e}")
                    continue
            
            if not all_calls and not all_puts:
                raise OptionsUnavailableError(f"No valid options contracts found for {ticker}")
            
            options_chain = OptionsChain(
                ticker=ticker,
                expiration_dates=expiration_dates,
                calls=all_calls,
                puts=all_puts
            )
            
            logger.info(f"Successfully fetched options chain for {ticker}: {len(all_calls)} calls, {len(all_puts)} puts")
            return options_chain
            
        except Exception as e:
            logger.error(f"Error fetching options chain for {ticker}: {str(e)}")
            raise OptionsUnavailableError(f"Failed to fetch options chain for {ticker}: {str(e)}")
    
    def validate_contract_quality(self, contract: OptionsContract) -> Dict[str, Any]:
        """
        Validate options contract quality based on liquidity and spread criteria.
        
        Implements requirements 4.2, 4.3, 4.4:
        - Open interest >= 100 contracts
        - Bid-ask spread <= 5% of mid price
        - Days to expiration between 30-60 days
        
        Args:
            contract: OptionsContract to validate
            
        Returns:
            Dictionary with validation results and reasons
        """
        validation = {
            'is_valid': True,
            'reasons': []
        }
        
        # Check open interest (requirement 4.2)
        if contract.open_interest < 100:
            validation['is_valid'] = False
            validation['reasons'].append(f"Low open interest: {contract.open_interest} (minimum 100)")
        
        # Check bid-ask spread (requirement 4.3)
        if contract.spread_percentage > 5.0:
            validation['is_valid'] = False
            validation['reasons'].append(f"Wide bid-ask spread: {contract.spread_percentage:.2f}% (maximum 5%)")
        
        # Check days to expiration (requirement 4.4)
        if contract.days_to_expiration < 30:
            validation['is_valid'] = False
            validation['reasons'].append(f"Too close to expiration: {contract.days_to_expiration} days (minimum 30)")
        elif contract.days_to_expiration > 60:
            validation['is_valid'] = False
            validation['reasons'].append(f"Too far from expiration: {contract.days_to_expiration} days (maximum 60)")
        
        if validation['is_valid']:
            validation['reasons'].append("Contract meets all quality criteria")
        
        return validation
    
    def filter_quality_contracts(self, contracts: List[OptionsContract]) -> List[OptionsContract]:
        """
        Filter contracts to only include those meeting quality criteria.
        
        Args:
            contracts: List of OptionsContract objects
            
        Returns:
            Filtered list of quality contracts
        """
        quality_contracts = []
        
        for contract in contracts:
            validation = self.validate_contract_quality(contract)
            if validation['is_valid']:
                quality_contracts.append(contract)
        
        logger.info(f"Filtered {len(quality_contracts)} quality contracts from {len(contracts)} total")
        return quality_contracts
    
    def calculate_greeks(self, contract: OptionsContract, stock_price: float, 
                        risk_free_rate: float = 0.05) -> Greeks:
        """
        Calculate options Greeks using Black-Scholes model.
        
        This is a simplified implementation. For production, consider using
        a dedicated options pricing library.
        
        Args:
            contract: OptionsContract to calculate Greeks for
            stock_price: Current stock price
            risk_free_rate: Risk-free interest rate (default 5%)
            
        Returns:
            Greeks object with calculated values
        """
        try:
            if not contract.implied_volatility or contract.implied_volatility == 0:
                logger.warning(f"Cannot calculate Greeks without implied volatility")
                return Greeks()
            
            # Time to expiration in years
            T = contract.days_to_expiration / 365.0
            
            if T <= 0:
                return Greeks()
            
            # Black-Scholes parameters
            S = stock_price
            K = contract.strike
            r = risk_free_rate
            sigma = contract.implied_volatility
            
            # Calculate d1 and d2
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            # Standard normal CDF approximation
            def norm_cdf(x):
                return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
            
            # Standard normal PDF
            def norm_pdf(x):
                return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
            
            # Calculate Greeks
            if contract.contract_type == 'CALL':
                delta = norm_cdf(d1)
                theta = (-(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) 
                        - r * K * math.exp(-r * T) * norm_cdf(d2)) / 365
            else:  # PUT
                delta = norm_cdf(d1) - 1
                theta = (-(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) 
                        + r * K * math.exp(-r * T) * norm_cdf(-d2)) / 365
            
            gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))
            vega = S * norm_pdf(d1) * math.sqrt(T) / 100  # Divided by 100 for 1% change
            
            if contract.contract_type == 'CALL':
                rho = K * T * math.exp(-r * T) * norm_cdf(d2) / 100
            else:  # PUT
                rho = -K * T * math.exp(-r * T) * norm_cdf(-d2) / 100
            
            return Greeks(
                delta=round(delta, 4),
                gamma=round(gamma, 4),
                theta=round(theta, 4),
                vega=round(vega, 4),
                rho=round(rho, 4)
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Greeks: {e}")
            return Greeks()
    
    def get_atm_contracts(self, contracts: List[OptionsContract], 
                         stock_price: float, num_strikes: int = 5) -> List[OptionsContract]:
        """
        Get at-the-money and near-the-money contracts.
        
        Args:
            contracts: List of OptionsContract objects
            stock_price: Current stock price
            num_strikes: Number of strikes around ATM to return
            
        Returns:
            List of contracts near the money
        """
        if not contracts:
            return []
        
        # Sort by distance from stock price
        sorted_contracts = sorted(contracts, key=lambda c: abs(c.strike - stock_price))
        
        # Return the closest strikes
        return sorted_contracts[:num_strikes]
    
    def close(self):
        """Close HTTP session"""
        if self.session:
            self.session.close()
            self.session = None
