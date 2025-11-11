import pytest
import sys
import os
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.options_data_service import (
    OptionsDataService,
    OptionsContract,
    OptionsChain,
    Greeks,
    OptionsUnavailableError
)


class TestOptionsDataService:
    """Test OptionsDataService functionality"""
    
    @pytest.fixture
    def options_service(self):
        """Create OptionsDataService instance"""
        return OptionsDataService()
    
    @pytest.fixture
    def sample_contract(self):
        """Create a sample options contract"""
        return OptionsContract(
            ticker="AAPL",
            contract_type="CALL",
            strike=150.0,
            expiration=date.today() + timedelta(days=45),
            bid=5.0,
            ask=5.2,
            last=5.1,
            volume=1000,
            open_interest=500,
            implied_volatility=0.25
        )
    
    @pytest.fixture
    def mock_options_chain_data(self):
        """Mock options chain data from yfinance"""
        calls_data = pd.DataFrame({
            'strike': [145.0, 150.0, 155.0],
            'bid': [7.0, 5.0, 3.0],
            'ask': [7.2, 5.2, 3.2],
            'lastPrice': [7.1, 5.1, 3.1],
            'volume': [1500, 1000, 800],
            'openInterest': [600, 500, 400],
            'impliedVolatility': [0.26, 0.25, 0.24]
        })
        
        puts_data = pd.DataFrame({
            'strike': [145.0, 150.0, 155.0],
            'bid': [3.0, 5.0, 7.0],
            'ask': [3.2, 5.2, 7.2],
            'lastPrice': [3.1, 5.1, 7.1],
            'volume': [800, 1000, 1500],
            'openInterest': [400, 500, 600],
            'impliedVolatility': [0.24, 0.25, 0.26]
        })
        
        mock_chain = MagicMock()
        mock_chain.calls = calls_data
        mock_chain.puts = puts_data
        
        return mock_chain
    
    @pytest.mark.asyncio
    async def test_get_options_chain_success(self, options_service, mock_options_chain_data):
        """Test successfully fetching options chain"""
        with patch('yfinance.Ticker') as mock_ticker:
            exp_date = (date.today() + timedelta(days=45)).strftime('%Y-%m-%d')
            mock_ticker.return_value.options = [exp_date]
            mock_ticker.return_value.option_chain.return_value = mock_options_chain_data
            
            chain = await options_service.get_options_chain("AAPL")
            
            assert isinstance(chain, OptionsChain)
            assert chain.ticker == "AAPL"
            assert len(chain.calls) == 3
            assert len(chain.puts) == 3
            assert len(chain.expiration_dates) == 1
    
    @pytest.mark.asyncio
    async def test_get_options_chain_no_options(self, options_service):
        """Test handling when no options are available"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.options = []
            
            with pytest.raises(OptionsUnavailableError, match="No options available"):
                await options_service.get_options_chain("INVALID")
    
    @pytest.mark.asyncio
    async def test_get_options_chain_multiple_expirations(self, options_service, mock_options_chain_data):
        """Test fetching multiple expiration dates"""
        with patch('yfinance.Ticker') as mock_ticker:
            exp_date1 = (date.today() + timedelta(days=30)).strftime('%Y-%m-%d')
            exp_date2 = (date.today() + timedelta(days=60)).strftime('%Y-%m-%d')
            mock_ticker.return_value.options = [exp_date1, exp_date2]
            mock_ticker.return_value.option_chain.return_value = mock_options_chain_data
            
            chain = await options_service.get_options_chain("AAPL", max_expirations=2)
            
            assert len(chain.expiration_dates) == 2
            assert len(chain.calls) == 6  # 3 strikes * 2 expirations
            assert len(chain.puts) == 6
    
    def test_validate_contract_quality_valid(self, options_service, sample_contract):
        """Test validation of a quality contract"""
        validation = options_service.validate_contract_quality(sample_contract)
        
        assert validation['is_valid'] is True
        assert "meets all quality criteria" in validation['reasons'][0]
    
    def test_validate_contract_quality_low_open_interest(self, options_service):
        """Test validation fails for low open interest"""
        contract = OptionsContract(
            ticker="AAPL",
            contract_type="CALL",
            strike=150.0,
            expiration=date.today() + timedelta(days=45),
            bid=5.0,
            ask=5.2,
            last=5.1,
            volume=1000,
            open_interest=50,  # Below 100 threshold
            implied_volatility=0.25
        )
        
        validation = options_service.validate_contract_quality(contract)
        
        assert validation['is_valid'] is False
        assert any("Low open interest" in reason for reason in validation['reasons'])
    
    def test_validate_contract_quality_wide_spread(self, options_service):
        """Test validation fails for wide bid-ask spread"""
        contract = OptionsContract(
            ticker="AAPL",
            contract_type="CALL",
            strike=150.0,
            expiration=date.today() + timedelta(days=45),
            bid=5.0,
            ask=6.0,  # 20% spread
            last=5.5,
            volume=1000,
            open_interest=500,
            implied_volatility=0.25
        )
        
        validation = options_service.validate_contract_quality(contract)
        
        assert validation['is_valid'] is False
        assert any("Wide bid-ask spread" in reason for reason in validation['reasons'])
    
    def test_validate_contract_quality_too_close_expiration(self, options_service):
        """Test validation fails for contracts too close to expiration"""
        contract = OptionsContract(
            ticker="AAPL",
            contract_type="CALL",
            strike=150.0,
            expiration=date.today() + timedelta(days=20),  # Less than 30 days
            bid=5.0,
            ask=5.2,
            last=5.1,
            volume=1000,
            open_interest=500,
            implied_volatility=0.25
        )
        
        validation = options_service.validate_contract_quality(contract)
        
        assert validation['is_valid'] is False
        assert any("Too close to expiration" in reason for reason in validation['reasons'])
    
    def test_validate_contract_quality_too_far_expiration(self, options_service):
        """Test validation fails for contracts too far from expiration"""
        contract = OptionsContract(
            ticker="AAPL",
            contract_type="CALL",
            strike=150.0,
            expiration=date.today() + timedelta(days=90),  # More than 60 days
            bid=5.0,
            ask=5.2,
            last=5.1,
            volume=1000,
            open_interest=500,
            implied_volatility=0.25
        )
        
        validation = options_service.validate_contract_quality(contract)
        
        assert validation['is_valid'] is False
        assert any("Too far from expiration" in reason for reason in validation['reasons'])
    
    def test_filter_quality_contracts(self, options_service):
        """Test filtering contracts by quality"""
        contracts = [
            OptionsContract(
                ticker="AAPL", contract_type="CALL", strike=150.0,
                expiration=date.today() + timedelta(days=45),
                bid=5.0, ask=5.2, last=5.1, volume=1000,
                open_interest=500, implied_volatility=0.25
            ),
            OptionsContract(
                ticker="AAPL", contract_type="CALL", strike=155.0,
                expiration=date.today() + timedelta(days=45),
                bid=3.0, ask=3.2, last=3.1, volume=500,
                open_interest=50,  # Too low
                implied_volatility=0.24
            ),
            OptionsContract(
                ticker="AAPL", contract_type="CALL", strike=160.0,
                expiration=date.today() + timedelta(days=45),
                bid=2.0, ask=2.1, last=2.05, volume=800,
                open_interest=300, implied_volatility=0.23
            )
        ]
        
        quality = options_service.filter_quality_contracts(contracts)
        
        assert len(quality) == 2  # First and third contracts pass
        assert quality[0].strike == 150.0
        assert quality[1].strike == 160.0
    
    def test_calculate_greeks_call(self, options_service, sample_contract):
        """Test Greeks calculation for call option"""
        greeks = options_service.calculate_greeks(sample_contract, stock_price=150.0)
        
        assert isinstance(greeks, Greeks)
        assert greeks.delta is not None
        assert 0 < greeks.delta <= 1  # Call delta is positive
        assert greeks.gamma is not None
        assert greeks.theta is not None
        assert greeks.vega is not None
        assert greeks.rho is not None
    
    def test_calculate_greeks_put(self, options_service):
        """Test Greeks calculation for put option"""
        contract = OptionsContract(
            ticker="AAPL",
            contract_type="PUT",
            strike=150.0,
            expiration=date.today() + timedelta(days=45),
            bid=5.0,
            ask=5.2,
            last=5.1,
            volume=1000,
            open_interest=500,
            implied_volatility=0.25
        )
        
        greeks = options_service.calculate_greeks(contract, stock_price=150.0)
        
        assert isinstance(greeks, Greeks)
        assert greeks.delta is not None
        assert -1 <= greeks.delta < 0  # Put delta is negative
    
    def test_calculate_greeks_no_iv(self, options_service):
        """Test Greeks calculation without implied volatility"""
        contract = OptionsContract(
            ticker="AAPL",
            contract_type="CALL",
            strike=150.0,
            expiration=date.today() + timedelta(days=45),
            bid=5.0,
            ask=5.2,
            last=5.1,
            volume=1000,
            open_interest=500,
            implied_volatility=None
        )
        
        greeks = options_service.calculate_greeks(contract, stock_price=150.0)
        
        assert greeks.delta is None
        assert greeks.gamma is None
    
    def test_get_atm_contracts(self, options_service):
        """Test getting at-the-money contracts"""
        contracts = [
            OptionsContract(
                ticker="AAPL", contract_type="CALL", strike=140.0,
                expiration=date.today() + timedelta(days=45),
                bid=10.0, ask=10.2, last=10.1, volume=1000,
                open_interest=500, implied_volatility=0.25
            ),
            OptionsContract(
                ticker="AAPL", contract_type="CALL", strike=150.0,
                expiration=date.today() + timedelta(days=45),
                bid=5.0, ask=5.2, last=5.1, volume=1000,
                open_interest=500, implied_volatility=0.25
            ),
            OptionsContract(
                ticker="AAPL", contract_type="CALL", strike=160.0,
                expiration=date.today() + timedelta(days=45),
                bid=2.0, ask=2.2, last=2.1, volume=1000,
                open_interest=500, implied_volatility=0.25
            )
        ]
        
        atm_contracts = options_service.get_atm_contracts(contracts, stock_price=150.0, num_strikes=2)
        
        assert len(atm_contracts) == 2
        assert atm_contracts[0].strike == 150.0  # Closest to stock price
    
    def test_options_contract_calculations(self):
        """Test OptionsContract calculated properties"""
        contract = OptionsContract(
            ticker="AAPL",
            contract_type="CALL",
            strike=150.0,
            expiration=date.today() + timedelta(days=45),
            bid=5.0,
            ask=5.2,
            last=5.1,
            volume=1000,
            open_interest=500,
            implied_volatility=0.25
        )
        
        assert contract.mid_price == 5.1
        assert abs(contract.bid_ask_spread - 0.2) < 0.01  # Allow for floating point precision
        assert abs(contract.spread_percentage - 3.92) < 0.1  # ~3.92%
        assert contract.days_to_expiration == 45
