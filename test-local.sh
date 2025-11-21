#!/bin/bash

# Quick Local Docker Test Script
set -e

echo "🧪 Testing Docker Setup Locally"
echo "================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "✅ Docker is running"

# Check if .env.local exists, if not create it
if [ ! -f .env.local ]; then
    echo "📝 Creating .env.local for testing..."
    cat > .env.local << EOF
# Test Environment Configuration
POSTGRES_DB=options_db_test
POSTGRES_USER=postgres
POSTGRES_PASSWORD=testpassword
POSTGRES_PORT=5432

REDIS_PORT=6379
REDIS_PASSWORD=

BACKEND_PORT=8000
DEBUG=true

APP_NAME=Options Trade Evaluator (Test)

TICKER_CACHE_TTL=300
ANALYSIS_CACHE_TTL=180
MARKET_DATA_CACHE_TTL=60

# Dummy API keys for testing (replace with real ones if needed)
OPENAI_API_KEY=test_key_not_real
ALPHA_VANTAGE_API_KEY=test_key_not_real
TRADIER_API_KEY=test_key_not_real
EOF
    echo "✅ Created .env.local with test configuration"
else
    echo "✅ Using existing .env.local"
fi

# Load environment variables
export $(cat .env.local | grep -v '^#' | xargs)

echo "🚀 Starting Docker services..."
docker-compose --env-file .env.local up --build -d

echo "⏳ Waiting for services to start..."
sleep 15

echo "🔍 Checking service status..."
docker-compose ps

echo ""
echo "🧪 Running health checks..."

# Wait for backend to be ready
max_attempts=30
attempt=1
while [ $attempt -le $max_attempts ]; do
    if curl -f http://localhost:${BACKEND_PORT:-8000}/api/v1/health > /dev/null 2>&1; then
        echo "✅ Backend is healthy!"
        break
    else
        echo "⏳ Waiting for backend... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ Backend failed to start within expected time"
    echo "📋 Backend logs:"
    docker-compose logs --tail=20 backend
    exit 1
fi

# Test endpoints
echo ""
echo "🔗 Testing API endpoints..."

# Test root endpoint
if curl -f http://localhost:${BACKEND_PORT:-8000}/ > /dev/null 2>&1; then
    echo "✅ Root endpoint working"
else
    echo "❌ Root endpoint failed"
fi

# Test health endpoint
if curl -f http://localhost:${BACKEND_PORT:-8000}/api/v1/health > /dev/null 2>&1; then
    echo "✅ Health endpoint working"
else
    echo "❌ Health endpoint failed"
fi

# Test database connection
echo ""
echo "🗄️ Testing database connection..."
if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL connection failed"
fi

# Test Redis connection
echo ""
echo "🔄 Testing Redis connection..."
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is responding"
else
    echo "❌ Redis connection failed"
fi

echo ""
echo "🎉 Local Docker test completed!"
echo ""
echo "📊 Service URLs:"
echo "  Backend API: http://localhost:${BACKEND_PORT:-8000}"
echo "  API Docs: http://localhost:${BACKEND_PORT:-8000}/docs"
echo "  Health Check: http://localhost:${BACKEND_PORT:-8000}/api/v1/health"
echo ""
echo "🔧 Useful commands:"
echo "  View logs: docker-compose logs -f backend"
echo "  Stop services: docker-compose down"
echo "  Monitor: ./scripts/monitor.sh"
echo ""
echo "💡 To test with real API keys, edit .env.local and restart services"