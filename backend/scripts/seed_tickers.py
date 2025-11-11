#!/usr/bin/env python3
"""
Ticker data seeding script for NYSE symbols

This script populates the database with a comprehensive list of NYSE ticker symbols
and their corresponding company names for the options trade evaluator.
"""

import asyncio
import sys
import os
from typing import List, Tuple

# Add the parent directory to the path so we can import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import AsyncSessionLocal, engine
from app.models.database import StockTicker
from app.services.ticker_service import TickerService
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample NYSE ticker data - in production this would come from a financial data API
NYSE_TICKERS = [
    # Major Tech Companies
    ("AAPL", "Apple Inc.", "NASDAQ"),
    ("MSFT", "Microsoft Corporation", "NASDAQ"),
    ("GOOGL", "Alphabet Inc.", "NASDAQ"),
    ("GOOG", "Alphabet Inc. Class A", "NASDAQ"),
    ("AMZN", "Amazon.com Inc.", "NASDAQ"),
    ("META", "Meta Platforms Inc.", "NASDAQ"),
    ("TSLA", "Tesla Inc.", "NASDAQ"),
    ("NVDA", "NVIDIA Corporation", "NASDAQ"),
    ("NFLX", "Netflix Inc.", "NASDAQ"),
    ("ADBE", "Adobe Inc.", "NASDAQ"),
    
    # Financial Services
    ("JPM", "JPMorgan Chase & Co.", "NYSE"),
    ("BAC", "Bank of America Corporation", "NYSE"),
    ("WFC", "Wells Fargo & Company", "NYSE"),
    ("GS", "The Goldman Sachs Group Inc.", "NYSE"),
    ("MS", "Morgan Stanley", "NYSE"),
    ("C", "Citigroup Inc.", "NYSE"),
    ("AXP", "American Express Company", "NYSE"),
    ("BLK", "BlackRock Inc.", "NYSE"),
    ("SCHW", "The Charles Schwab Corporation", "NYSE"),
    ("USB", "U.S. Bancorp", "NYSE"),
    
    # Healthcare & Pharmaceuticals
    ("JNJ", "Johnson & Johnson", "NYSE"),
    ("PFE", "Pfizer Inc.", "NYSE"),
    ("UNH", "UnitedHealth Group Incorporated", "NYSE"),
    ("ABBV", "AbbVie Inc.", "NYSE"),
    ("MRK", "Merck & Co. Inc.", "NYSE"),
    ("TMO", "Thermo Fisher Scientific Inc.", "NYSE"),
    ("ABT", "Abbott Laboratories", "NYSE"),
    ("LLY", "Eli Lilly and Company", "NYSE"),
    ("MDT", "Medtronic plc", "NYSE"),
    ("BMY", "Bristol-Myers Squibb Company", "NYSE"),
    
    # Consumer Goods & Retail
    ("WMT", "Walmart Inc.", "NYSE"),
    ("PG", "The Procter & Gamble Company", "NYSE"),
    ("KO", "The Coca-Cola Company", "NYSE"),
    ("PEP", "PepsiCo Inc.", "NASDAQ"),
    ("COST", "Costco Wholesale Corporation", "NASDAQ"),
    ("HD", "The Home Depot Inc.", "NYSE"),
    ("MCD", "McDonald's Corporation", "NYSE"),
    ("NKE", "NIKE Inc.", "NYSE"),
    ("SBUX", "Starbucks Corporation", "NASDAQ"),
    ("TGT", "Target Corporation", "NYSE"),
    
    # Industrial & Manufacturing
    ("BA", "The Boeing Company", "NYSE"),
    ("CAT", "Caterpillar Inc.", "NYSE"),
    ("GE", "General Electric Company", "NYSE"),
    ("MMM", "3M Company", "NYSE"),
    ("HON", "Honeywell International Inc.", "NASDAQ"),
    ("UPS", "United Parcel Service Inc.", "NYSE"),
    ("LMT", "Lockheed Martin Corporation", "NYSE"),
    ("RTX", "Raytheon Technologies Corporation", "NYSE"),
    ("DE", "Deere & Company", "NYSE"),
    ("FDX", "FedEx Corporation", "NYSE"),
    
    # Energy & Utilities
    ("XOM", "Exxon Mobil Corporation", "NYSE"),
    ("CVX", "Chevron Corporation", "NYSE"),
    ("COP", "ConocoPhillips", "NYSE"),
    ("SLB", "Schlumberger Limited", "NYSE"),
    ("EOG", "EOG Resources Inc.", "NYSE"),
    ("KMI", "Kinder Morgan Inc.", "NYSE"),
    ("OXY", "Occidental Petroleum Corporation", "NYSE"),
    ("PSX", "Phillips 66", "NYSE"),
    ("VLO", "Valero Energy Corporation", "NYSE"),
    ("MPC", "Marathon Petroleum Corporation", "NYSE"),
    
    # Telecommunications & Media
    ("VZ", "Verizon Communications Inc.", "NYSE"),
    ("T", "AT&T Inc.", "NYSE"),
    ("CMCSA", "Comcast Corporation", "NASDAQ"),
    ("DIS", "The Walt Disney Company", "NYSE"),
    ("TMUS", "T-Mobile US Inc.", "NASDAQ"),
    ("CHTR", "Charter Communications Inc.", "NASDAQ"),
    ("VIA", "ViacomCBS Inc.", "NASDAQ"),
    ("FOXA", "Fox Corporation", "NASDAQ"),
    ("FOX", "Fox Corporation Class B", "NASDAQ"),
    ("DISH", "DISH Network Corporation", "NASDAQ"),
    
    # Real Estate & REITs
    ("AMT", "American Tower Corporation", "NYSE"),
    ("PLD", "Prologis Inc.", "NYSE"),
    ("CCI", "Crown Castle International Corp.", "NYSE"),
    ("EQIX", "Equinix Inc.", "NASDAQ"),
    ("SPG", "Simon Property Group Inc.", "NYSE"),
    ("O", "Realty Income Corporation", "NYSE"),
    ("WELL", "Welltower Inc.", "NYSE"),
    ("AVB", "AvalonBay Communities Inc.", "NYSE"),
    ("EQR", "Equity Residential", "NYSE"),
    ("DLR", "Digital Realty Trust Inc.", "NYSE"),
    
    # Additional Popular Trading Stocks
    ("TSLA", "Tesla Inc.", "NASDAQ"),
    ("AMD", "Advanced Micro Devices Inc.", "NASDAQ"),
    ("INTC", "Intel Corporation", "NASDAQ"),
    ("CRM", "Salesforce Inc.", "NYSE"),
    ("ORCL", "Oracle Corporation", "NYSE"),
    ("IBM", "International Business Machines Corporation", "NYSE"),
    ("CSCO", "Cisco Systems Inc.", "NASDAQ"),
    ("QCOM", "QUALCOMM Incorporated", "NASDAQ"),
    ("TXN", "Texas Instruments Incorporated", "NASDAQ"),
    ("AVGO", "Broadcom Inc.", "NASDAQ"),
    
    # ETFs and Popular Options Trading Symbols
    ("SPY", "SPDR S&P 500 ETF Trust", "NYSE"),
    ("QQQ", "Invesco QQQ Trust", "NASDAQ"),
    ("IWM", "iShares Russell 2000 ETF", "NYSE"),
    ("EEM", "iShares MSCI Emerging Markets ETF", "NYSE"),
    ("GLD", "SPDR Gold Shares", "NYSE"),
    ("SLV", "iShares Silver Trust", "NYSE"),
    ("TLT", "iShares 20+ Year Treasury Bond ETF", "NASDAQ"),
    ("VIX", "CBOE Volatility Index", "CBOE"),
    ("USO", "United States Oil Fund", "NYSE"),
    ("XLE", "Energy Select Sector SPDR Fund", "NYSE"),
]

async def seed_tickers(session: AsyncSession, tickers: List[Tuple[str, str, str]]) -> None:
    """
    Seed the database with ticker data
    
    Args:
        session: Database session
        tickers: List of (ticker, company_name, exchange) tuples
    """
    ticker_service = TickerService(session)
    
    logger.info(f"Starting to seed {len(tickers)} tickers...")
    
    added_count = 0
    updated_count = 0
    error_count = 0
    
    for ticker_symbol, company_name, exchange in tickers:
        try:
            # Check if ticker already exists
            existing_ticker = await ticker_service.get_ticker_by_symbol(ticker_symbol)
            
            if existing_ticker:
                # Update existing ticker if company name or exchange changed
                if (existing_ticker.company_name != company_name or 
                    existing_ticker.exchange != exchange):
                    
                    await ticker_service.update_ticker(
                        ticker=ticker_symbol,
                        company_name=company_name,
                        exchange=exchange,
                        is_active=True
                    )
                    updated_count += 1
                    logger.info(f"Updated ticker: {ticker_symbol}")
                else:
                    logger.debug(f"Ticker {ticker_symbol} already exists and is up to date")
            else:
                # Add new ticker
                await ticker_service.add_ticker(
                    ticker=ticker_symbol,
                    company_name=company_name,
                    exchange=exchange
                )
                added_count += 1
                logger.info(f"Added ticker: {ticker_symbol}")
                
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing ticker {ticker_symbol}: {str(e)}")
    
    logger.info(f"Seeding completed: {added_count} added, {updated_count} updated, {error_count} errors")

async def main():
    """Main function to run the seeding script"""
    logger.info("Starting ticker seeding script...")
    
    try:
        # Create database session
        async with AsyncSessionLocal() as session:
            await seed_tickers(session, NYSE_TICKERS)
            
        logger.info("Ticker seeding completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during ticker seeding: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())