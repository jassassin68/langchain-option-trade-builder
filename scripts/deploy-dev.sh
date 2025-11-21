#!/bin/bash

# Development Deployment Script
set -e

echo "🚀 Starting development deployment..."

# Load development environment
if [ -f .env.development ]; then
    export $(cat .env.development | grep -v '^#' | xargs)
    echo "✅ Loaded development environment variables"
else
    echo "❌ .env.development file not found"
    exit 1
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build and start services
echo "🔨 Building and starting services..."
docker-compose --env-file .env.development up --build -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

# Run database migrations if needed
echo "🗄️ Running database migrations..."
docker-compose exec backend python -c "
import asyncio
from app.database.connection import init_db
asyncio.run(init_db())
print('Database initialized successfully')
"

echo "✅ Development deployment completed!"
echo "🌐 Backend API: http://localhost:${BACKEND_PORT:-8000}"
echo "📚 API Documentation: http://localhost:${BACKEND_PORT:-8000}/docs"
echo "🔍 Health Check: http://localhost:${BACKEND_PORT:-8000}/api/v1/health"