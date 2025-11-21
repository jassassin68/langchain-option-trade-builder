#!/bin/bash

# Production Deployment Script
set -e

echo "🚀 Starting production deployment..."

# Check if production environment file exists
if [ ! -f .env.production ]; then
    echo "❌ .env.production file not found"
    echo "Please create .env.production with production configuration"
    exit 1
fi

# Load production environment
export $(cat .env.production | grep -v '^#' | xargs)
echo "✅ Loaded production environment variables"

# Validate required environment variables
required_vars=("POSTGRES_PASSWORD" "REDIS_PASSWORD" "OPENAI_API_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ] || [ "${!var}" = "CHANGE_ME_"* ]; then
        echo "❌ Required environment variable $var is not set or uses default value"
        exit 1
    fi
done

# Create backup of current deployment
echo "💾 Creating backup..."
timestamp=$(date +%Y%m%d_%H%M%S)
docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U $POSTGRES_USER $POSTGRES_DB > "backups/backup_${timestamp}.sql" || echo "⚠️ Backup failed or no existing database"

# Pull latest images
echo "📥 Pulling latest images..."
docker-compose -f docker-compose.prod.yml pull

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down

# Build and start services
echo "🔨 Building and starting production services..."
docker-compose -f docker-compose.prod.yml --env-file .env.production up --build -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
docker-compose -f docker-compose.prod.yml ps

# Run database migrations
echo "🗄️ Running database migrations..."
docker-compose -f docker-compose.prod.yml exec backend python -c "
import asyncio
from app.database.connection import init_db
asyncio.run(init_db())
print('Database initialized successfully')
"

# Verify deployment
echo "🧪 Verifying deployment..."
health_check_url="http://localhost:${BACKEND_PORT:-8000}/api/v1/health"
if curl -f $health_check_url > /dev/null 2>&1; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

echo "✅ Production deployment completed successfully!"
echo "🌐 Backend API: http://localhost:${BACKEND_PORT:-8000}"
echo "📊 Monitor logs: docker-compose -f docker-compose.prod.yml logs -f"