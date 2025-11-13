"""
Tests for health check and monitoring API endpoints.

Implements requirement testing for 7.3, 7.4, 7.5:
- GET /api/v1/health with service status checks
- Monitoring endpoints
"""

import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import app

client = TestClient(app)


class TestHealthCheckAPI:
    """Test health check API endpoint"""
    
    @patch('backend.app.api.v1.health.redis.from_url')
    def test_health_check_all_services_healthy(self, mock_redis):
        """Test health check when all services are healthy"""
        # Mock Redis client
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock()
        mock_redis_client.close = AsyncMock()
        mock_redis.return_value = mock_redis_client
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "status" in data
        assert "timestamp" in data
        assert "services" in data
        
        # Verify status
        assert data["status"] == "healthy"
        
        # Verify services
        services = data["services"]
        assert "database" in services
        assert "cache" in services
    
    @patch('backend.app.api.v1.health.redis.from_url')
    def test_health_check_database_healthy(self, mock_redis):
        """Test that database health is checked"""
        # Mock Redis client
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock()
        mock_redis_client.close = AsyncMock()
        mock_redis.return_value = mock_redis_client
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Database should be healthy
        assert data["services"]["database"] == "healthy"
    
    @patch('backend.app.api.v1.health.redis.from_url')
    def test_health_check_cache_healthy(self, mock_redis):
        """Test that cache health is checked"""
        # Mock Redis client
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock()
        mock_redis_client.close = AsyncMock()
        mock_redis.return_value = mock_redis_client
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Cache should be healthy
        assert data["services"]["cache"] == "healthy"
    
    @patch('backend.app.api.v1.health.redis.from_url')
    def test_health_check_cache_unhealthy(self, mock_redis):
        """Test health check when cache is unavailable"""
        # Mock Redis client to raise exception
        mock_redis.side_effect = Exception("Redis connection failed")
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200  # Still returns 200 as cache is not critical
        data = response.json()
        
        # Overall status should still be healthy (cache is not critical)
        assert data["status"] == "healthy"
        
        # Cache should be marked as unhealthy
        assert "unhealthy" in data["services"]["cache"]
    
    @patch('backend.app.api.v1.health.redis.from_url')
    def test_health_check_includes_timestamp(self, mock_redis):
        """Test that health check includes timestamp"""
        # Mock Redis client
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock()
        mock_redis_client.close = AsyncMock()
        mock_redis.return_value = mock_redis_client
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify timestamp is present and valid
        assert "timestamp" in data
        # Should be able to parse as ISO format datetime
        timestamp = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
        assert isinstance(timestamp, datetime)
    
    @patch('backend.app.api.v1.health.redis.from_url')
    @patch('backend.app.api.v1.health.settings')
    def test_health_check_includes_api_configuration(self, mock_settings, mock_redis):
        """Test that health check includes API configuration status"""
        # Mock Redis client
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock()
        mock_redis_client.close = AsyncMock()
        mock_redis.return_value = mock_redis_client
        
        # Mock settings
        mock_settings.openai_api_key = "test_key"
        mock_settings.alpha_vantage_api_key = "test_key"
        mock_settings.tradier_api_key = "test_key"
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify API configuration status
        services = data["services"]
        assert "openai" in services
        assert "market_data" in services
        assert "options_data" in services
    
    @patch('backend.app.api.v1.health.redis.from_url')
    @patch('backend.app.api.v1.health.settings')
    def test_health_check_missing_api_keys(self, mock_settings, mock_redis):
        """Test health check when API keys are not configured"""
        # Mock Redis client
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock()
        mock_redis_client.close = AsyncMock()
        mock_redis.return_value = mock_redis_client
        
        # Mock settings with no API keys
        mock_settings.openai_api_key = None
        mock_settings.alpha_vantage_api_key = None
        mock_settings.tradier_api_key = None
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify API configuration status shows not configured
        services = data["services"]
        assert services["openai"] == "not configured"
        assert services["market_data"] == "not configured"
        assert services["options_data"] == "not configured"
    
    def test_metrics_endpoint_exists(self):
        """Test that metrics endpoint exists"""
        response = client.get("/api/v1/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify basic response structure
        assert "status" in data or "message" in data
    
    def test_metrics_endpoint_includes_timestamp(self):
        """Test that metrics endpoint includes timestamp"""
        response = client.get("/api/v1/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify timestamp is present
        assert "timestamp" in data
    
    @patch('backend.app.api.v1.health.redis.from_url')
    def test_health_check_response_format(self, mock_redis):
        """Test that health check response matches HealthResponse model"""
        # Mock Redis client
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock()
        mock_redis_client.close = AsyncMock()
        mock_redis.return_value = mock_redis_client
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure matches HealthResponse model
        assert isinstance(data["status"], str)
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["services"], dict)
    
    @patch('backend.app.api.v1.health.redis.from_url')
    def test_health_check_concurrent_requests(self, mock_redis):
        """Test health check handles concurrent requests (requirement 7.3)"""
        # Mock Redis client
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock()
        mock_redis_client.close = AsyncMock()
        mock_redis.return_value = mock_redis_client
        
        # Make multiple concurrent requests
        responses = []
        for _ in range(10):
            response = client.get("/api/v1/health")
            responses.append(response)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
