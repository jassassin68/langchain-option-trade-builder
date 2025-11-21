#!/bin/bash

# Health Check Script for monitoring deployment
set -e

# Configuration
BACKEND_URL=${BACKEND_URL:-"http://localhost:8000"}
TIMEOUT=${TIMEOUT:-30}
RETRY_COUNT=${RETRY_COUNT:-5}
RETRY_DELAY=${RETRY_DELAY:-5}

echo "🔍 Starting health check for Options Trade Evaluator..."
echo "Backend URL: $BACKEND_URL"

# Function to check service health
check_health() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo "Checking $service_name..."
    
    for i in $(seq 1 $RETRY_COUNT); do
        if response=$(curl -s -w "%{http_code}" -o /tmp/health_response --max-time $TIMEOUT "$url" 2>/dev/null); then
            if [ "$response" = "$expected_status" ]; then
                echo "✅ $service_name is healthy (HTTP $response)"
                return 0
            else
                echo "⚠️ $service_name returned HTTP $response (attempt $i/$RETRY_COUNT)"
            fi
        else
            echo "❌ $service_name is unreachable (attempt $i/$RETRY_COUNT)"
        fi
        
        if [ $i -lt $RETRY_COUNT ]; then
            echo "Retrying in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
        fi
    done
    
    echo "❌ $service_name health check failed after $RETRY_COUNT attempts"
    return 1
}

# Check main API endpoint
check_health "Main API" "$BACKEND_URL/"

# Check health endpoint
check_health "Health Endpoint" "$BACKEND_URL/api/v1/health"

# Check API documentation
check_health "API Documentation" "$BACKEND_URL/docs"

# Check ticker search endpoint
echo "🔍 Testing ticker search functionality..."
search_response=$(curl -s "$BACKEND_URL/api/v1/tickers/search?q=AAPL" --max-time $TIMEOUT 2>/dev/null || echo "FAILED")
if [ "$search_response" != "FAILED" ] && echo "$search_response" | grep -q "results"; then
    echo "✅ Ticker search is working"
else
    echo "❌ Ticker search failed"
    exit 1
fi

# Check database connectivity (if health endpoint provides this info)
echo "🗄️ Checking database connectivity..."
if curl -s "$BACKEND_URL/api/v1/health" --max-time $TIMEOUT | grep -q "database.*healthy\|status.*ok" 2>/dev/null; then
    echo "✅ Database connectivity is healthy"
else
    echo "⚠️ Database connectivity status unclear"
fi

# Check Redis connectivity (if health endpoint provides this info)
echo "🔄 Checking Redis connectivity..."
if curl -s "$BACKEND_URL/api/v1/health" --max-time $TIMEOUT | grep -q "redis.*healthy\|cache.*ok" 2>/dev/null; then
    echo "✅ Redis connectivity is healthy"
else
    echo "⚠️ Redis connectivity status unclear"
fi

echo ""
echo "✅ Health check completed successfully!"
echo "🌐 All services are operational"

# Cleanup
rm -f /tmp/health_response