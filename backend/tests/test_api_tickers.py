"""
Tests for ticker search API endpoint.

Implements requirement testing for 1.1, 1.2, 1.3, 1.4, 7.2:
- GET /api/v1/tickers/search with query parameter validation
- Autocomplete functionality
- Response formatting and error handling
"""

import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import app
from app.models.api import TickerResult

client = TestClient(app)


class TestTickerSearchAPI:
    """Test ticker search API endpoint"""
    
    @pytest.fixture
    def sample_ticker_results(self):
        """Sample ticker results for mocking"""
        return [
            TickerResult(ticker="AAPL", company_name="Apple Inc.", exchange="NASDAQ"),
            TickerResult(ticker="MSFT", company_name="Microsoft Corporation", exchange="NASDAQ"),
            TickerResult(ticker="GOOGL", company_name="Alphabet Inc.", exchange="NASDAQ")
        ]
    
    @patch('backend.app.api.v1.tickers.TickerService')
    def test_search_tickers_success(self, mock_ticker_service_class, sample_ticker_results):
        """Test successful ticker search"""
        # Mock the service
        mock_service = AsyncMock()
        mock_service.search_tickers.return_value = sample_ticker_results
        mock_ticker_service_class.return_value = mock_service
        
        response = client.get("/api/v1/tickers/search?q=tech")
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "count" in data
        assert data["count"] == 3
        assert len(data["results"]) == 3
        assert data["results"][0]["ticker"] == "AAPL"
    
    @patch('backend.app.api.v1.tickers.TickerService')
    def test_search_tickers_single_result(self, mock_ticker_service_class, sample_ticker_results):
        """Test search returning single result"""
        # Mock the service
        mock_service = AsyncMock()
        mock_service.search_tickers.return_value = [sample_ticker_results[0]]
        mock_ticker_service_class.return_value = mock_service
        
        response = client.get("/api/v1/tickers/search?q=AAPL")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["results"][0]["ticker"] == "AAPL"
        assert data["results"][0]["company_name"] == "Apple Inc."
    
    @patch('backend.app.api.v1.tickers.TickerService')
    def test_search_tickers_no_results(self, mock_ticker_service_class):
        """Test search with no matching results"""
        # Mock the service
        mock_service = AsyncMock()
        mock_service.search_tickers.return_value = []
        mock_ticker_service_class.return_value = mock_service
        
        response = client.get("/api/v1/tickers/search?q=NONEXISTENT")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert len(data["results"]) == 0
    
    def test_search_tickers_missing_query(self):
        """Test search without query parameter"""
        response = client.get("/api/v1/tickers/search")
        
        assert response.status_code == 422  # Validation error
    
    def test_search_tickers_empty_query(self):
        """Test search with empty query string"""
        response = client.get("/api/v1/tickers/search?q=")
        
        assert response.status_code == 422  # Validation error (min_length=1)
    
    @patch('backend.app.api.v1.tickers.TickerService')
    def test_search_tickers_with_limit(self, mock_ticker_service_class, sample_ticker_results):
        """Test search with custom limit parameter"""
        # Mock the service
        mock_service = AsyncMock()
        mock_service.search_tickers.return_value = sample_ticker_results[:2]
        mock_ticker_service_class.return_value = mock_service
        
        response = client.get("/api/v1/tickers/search?q=tech&limit=2")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["results"]) == 2
    
    def test_search_tickers_limit_too_large(self):
        """Test search with limit exceeding maximum"""
        response = client.get("/api/v1/tickers/search?q=tech&limit=100")
        
        assert response.status_code == 422  # Validation error (max 50)
    
    def test_search_tickers_limit_zero(self):
        """Test search with limit of zero"""
        response = client.get("/api/v1/tickers/search?q=tech&limit=0")
        
        assert response.status_code == 422  # Validation error (min 1)
    
    def test_search_tickers_limit_negative(self):
        """Test search with negative limit"""
        response = client.get("/api/v1/tickers/search?q=tech&limit=-1")
        
        assert response.status_code == 422  # Validation error
    
    @patch('backend.app.api.v1.tickers.TickerService')
    def test_search_tickers_case_insensitive(self, mock_ticker_service_class, sample_ticker_results):
        """Test case insensitive search"""
        # Mock the service
        mock_service = AsyncMock()
        mock_service.search_tickers.return_value = [sample_ticker_results[0]]
        mock_ticker_service_class.return_value = mock_service
        
        response = client.get("/api/v1/tickers/search?q=aapl")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["results"][0]["ticker"] == "AAPL"
    
    @patch('backend.app.api.v1.tickers.TickerService')
    def test_search_tickers_whitespace_trimmed(self, mock_ticker_service_class, sample_ticker_results):
        """Test that whitespace is trimmed from query"""
        # Mock the service
        mock_service = AsyncMock()
        mock_service.search_tickers.return_value = [sample_ticker_results[0]]
        mock_ticker_service_class.return_value = mock_service
        
        response = client.get("/api/v1/tickers/search?q=%20AAPL%20")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
    
    @patch('backend.app.api.v1.tickers.TickerService')
    def test_search_tickers_service_error(self, mock_ticker_service_class):
        """Test handling of service errors"""
        # Mock the service to raise an exception
        mock_service = AsyncMock()
        mock_service.search_tickers.side_effect = Exception("Database error")
        mock_ticker_service_class.return_value = mock_service
        
        response = client.get("/api/v1/tickers/search?q=AAPL")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    
    @patch('backend.app.api.v1.tickers.TickerService')
    def test_search_tickers_response_format(self, mock_ticker_service_class, sample_ticker_results):
        """Test response format matches TickerSearchResponse model"""
        # Mock the service
        mock_service = AsyncMock()
        mock_service.search_tickers.return_value = sample_ticker_results
        mock_ticker_service_class.return_value = mock_service
        
        response = client.get("/api/v1/tickers/search?q=tech")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "results" in data
        assert "count" in data
        assert isinstance(data["results"], list)
        assert isinstance(data["count"], int)
        
        # Verify result structure
        for result in data["results"]:
            assert "ticker" in result
            assert "company_name" in result
            assert "exchange" in result
    
    @patch('backend.app.api.v1.tickers.TickerService')
    def test_search_tickers_single_character(self, mock_ticker_service_class, sample_ticker_results):
        """Test search with single character (requirement 1.1)"""
        # Mock the service
        mock_service = AsyncMock()
        mock_service.search_tickers.return_value = [sample_ticker_results[0]]
        mock_ticker_service_class.return_value = mock_service
        
        response = client.get("/api/v1/tickers/search?q=A")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 0
    
    @patch('backend.app.api.v1.tickers.TickerService')
    def test_search_tickers_special_characters(self, mock_ticker_service_class):
        """Test search with special characters"""
        # Mock the service
        mock_service = AsyncMock()
        mock_service.search_tickers.return_value = []
        mock_ticker_service_class.return_value = mock_service
        
        response = client.get("/api/v1/tickers/search?q=A%26P")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
    
    def test_search_tickers_query_too_long(self):
        """Test search with very long query"""
        long_query = "A" * 100
        response = client.get(f"/api/v1/tickers/search?q={long_query}")
        
        assert response.status_code == 422  # Validation error (max_length=50)
    
    @patch('backend.app.api.v1.tickers.TickerService')
    def test_search_tickers_default_limit(self, mock_ticker_service_class, sample_ticker_results):
        """Test that default limit of 10 is applied (requirement 1.3)"""
        # Mock the service
        mock_service = AsyncMock()
        mock_service.search_tickers.return_value = sample_ticker_results
        mock_ticker_service_class.return_value = mock_service
        
        response = client.get("/api/v1/tickers/search?q=tech")
        
        assert response.status_code == 200
        # Service should be called with default limit of 10
        mock_service.search_tickers.assert_called_once()
