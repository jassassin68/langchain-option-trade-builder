@echo off
REM Quick Local Docker Test Script for Windows
setlocal enabledelayedexpansion

echo 🧪 Testing Docker Setup Locally
echo ================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker and try again.
    exit /b 1
)

echo ✅ Docker is running

REM Check if .env.local exists, if not create it
if not exist .env.local (
    echo 📝 Creating .env.local for testing...
    (
        echo # Test Environment Configuration
        echo POSTGRES_DB=options_db_test
        echo POSTGRES_USER=postgres
        echo POSTGRES_PASSWORD=testpassword
        echo POSTGRES_PORT=5432
        echo.
        echo REDIS_PORT=6379
        echo REDIS_PASSWORD=
        echo.
        echo BACKEND_PORT=8000
        echo DEBUG=true
        echo.
        echo APP_NAME=Options Trade Evaluator ^(Test^)
        echo.
        echo TICKER_CACHE_TTL=300
        echo ANALYSIS_CACHE_TTL=180
        echo MARKET_DATA_CACHE_TTL=60
        echo.
        echo # Dummy API keys for testing ^(replace with real ones if needed^)
        echo OPENAI_API_KEY=test_key_not_real
        echo ALPHA_VANTAGE_API_KEY=test_key_not_real
        echo TRADIER_API_KEY=test_key_not_real
    ) > .env.local
    echo ✅ Created .env.local with test configuration
) else (
    echo ✅ Using existing .env.local
)

echo 🚀 Starting Docker services...
docker-compose --env-file .env.local up --build -d

echo ⏳ Waiting for services to start...
timeout /t 15 /nobreak >nul

echo 🔍 Checking service status...
docker-compose ps

echo.
echo 🧪 Running health checks...

REM Wait for backend to be ready
set max_attempts=30
set attempt=1

:wait_loop
curl -f http://localhost:8000/api/v1/health >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Backend is healthy!
    goto :health_check_done
) else (
    echo ⏳ Waiting for backend... ^(attempt !attempt!/!max_attempts!^)
    timeout /t 2 /nobreak >nul
    set /a attempt+=1
    if !attempt! leq !max_attempts! goto :wait_loop
)

echo ❌ Backend failed to start within expected time
echo 📋 Backend logs:
docker-compose logs --tail=20 backend
exit /b 1

:health_check_done

REM Test endpoints
echo.
echo 🔗 Testing API endpoints...

REM Test root endpoint
curl -f http://localhost:8000/ >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Root endpoint working
) else (
    echo ❌ Root endpoint failed
)

REM Test health endpoint
curl -f http://localhost:8000/api/v1/health >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Health endpoint working
) else (
    echo ❌ Health endpoint failed
)

REM Test database connection
echo.
echo 🗄️ Testing database connection...
docker-compose exec -T postgres pg_isready -U postgres >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ PostgreSQL is ready
) else (
    echo ❌ PostgreSQL connection failed
)

REM Test Redis connection
echo.
echo 🔄 Testing Redis connection...
docker-compose exec -T redis redis-cli ping >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Redis is responding
) else (
    echo ❌ Redis connection failed
)

echo.
echo 🎉 Local Docker test completed!
echo.
echo 📊 Service URLs:
echo   Backend API: http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo   Health Check: http://localhost:8000/api/v1/health
echo.
echo 🔧 Useful commands:
echo   View logs: docker-compose logs -f backend
echo   Stop services: docker-compose down
echo   Monitor: scripts\monitor.sh
echo.
echo 💡 To test with real API keys, edit .env.local and restart services

pause