@echo off
REM Development Deployment Script for Windows
setlocal enabledelayedexpansion

echo 🚀 Starting development deployment...

REM Check if development environment file exists
if not exist .env.development (
    echo ❌ .env.development file not found
    exit /b 1
)

echo ✅ Found development environment file

REM Stop existing containers
echo 🛑 Stopping existing containers...
docker-compose down

REM Build and start services
echo 🔨 Building and starting services...
docker-compose --env-file .env.development up --build -d

REM Wait for services to be healthy
echo ⏳ Waiting for services to be healthy...
timeout /t 10 /nobreak > nul

REM Check service health
echo 🔍 Checking service health...
docker-compose ps

REM Run database migrations if needed
echo 🗄️ Running database migrations...
docker-compose exec backend python -c "import asyncio; from app.database.connection import init_db; asyncio.run(init_db()); print('Database initialized successfully')"

echo ✅ Development deployment completed!
echo 🌐 Backend API: http://localhost:8000
echo 📚 API Documentation: http://localhost:8000/docs
echo 🔍 Health Check: http://localhost:8000/api/v1/health

pause