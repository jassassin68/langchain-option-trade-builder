"""
Main API v1 router that combines all endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1 import tickers, trades, health

# Create main v1 router
api_router = APIRouter(prefix="/api/v1")

# Include all sub-routers
api_router.include_router(tickers.router)
api_router.include_router(trades.router)
api_router.include_router(health.router)

__all__ = ["api_router"]
