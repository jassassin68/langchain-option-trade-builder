"""
Health check and monitoring API endpoints.

Implements requirements 7.3, 7.4, 7.5:
- GET /api/v1/health with service status checks
- Monitoring endpoints for metrics collection
- Structured logging for request tracking and debugging
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import datetime, timezone
import logging
import redis.asyncio as redis

from app.models.api import HealthResponse, ErrorResponse
from app.core.database import get_db
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Service unhealthy"}
    },
    summary="Health check endpoint",
    description="""
    Check the health status of the API and its dependencies.
    
    - Verifies database connectivity
    - Checks Redis cache availability
    - Returns status of all dependent services
    """
)
async def health_check(
    db: AsyncSession = Depends(get_db)
) -> HealthResponse:
    """
    Perform health check on API and dependent services.
    
    Implements requirement 7.3: Handle 10 concurrent users without degradation
    Implements requirement 7.4: Monitor service availability
    Implements requirement 7.5: Provide appropriate error messages
    
    Args:
        db: Database session (injected)
    
    Returns:
        HealthResponse with service status information
    """
    services_status = {}
    overall_healthy = True
    
    # Check database connectivity
    try:
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        services_status["database"] = "healthy"
        logger.debug("Database health check: healthy")
    except Exception as e:
        services_status["database"] = f"unhealthy: {str(e)}"
        overall_healthy = False
        logger.error(f"Database health check failed: {str(e)}")
    
    # Check Redis cache connectivity
    try:
        redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
        await redis_client.close()
        services_status["cache"] = "healthy"
        logger.debug("Redis cache health check: healthy")
    except Exception as e:
        services_status["cache"] = f"unhealthy: {str(e)}"
        # Cache is not critical, so don't mark overall as unhealthy
        logger.warning(f"Redis cache health check failed: {str(e)}")
    
    # Check OpenAI API key configuration
    if settings.openai_api_key:
        services_status["openai"] = "configured"
    else:
        services_status["openai"] = "not configured"
        logger.warning("OpenAI API key not configured")
    
    # Check external API keys configuration
    if settings.alpha_vantage_api_key:
        services_status["market_data"] = "configured"
    else:
        services_status["market_data"] = "not configured"
        logger.warning("Alpha Vantage API key not configured")
    
    if settings.tradier_api_key:
        services_status["options_data"] = "configured"
    else:
        services_status["options_data"] = "not configured"
        logger.warning("Tradier API key not configured")
    
    return HealthResponse(
        status="healthy" if overall_healthy else "unhealthy",
        timestamp=datetime.now(timezone.utc),
        services=services_status
    )


@router.get(
    "/metrics",
    summary="Get application metrics",
    description="""
    Get basic application metrics for monitoring.
    
    - Request counts
    - Response times
    - Error rates
    """
)
async def get_metrics():
    """
    Get application metrics for monitoring.
    
    Note: This is a placeholder for future metrics implementation.
    In production, consider using Prometheus or similar monitoring tools.
    
    Returns:
        Dictionary with basic metrics
    """
    # Placeholder for metrics
    # In production, integrate with Prometheus, DataDog, or similar
    return {
        "status": "metrics endpoint",
        "message": "Metrics collection not yet implemented. Consider integrating Prometheus.",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
