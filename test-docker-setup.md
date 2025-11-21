# Testing Docker Setup Locally

## Prerequisites

Make sure you have:
- Docker and Docker Compose installed
- Environment variables configured
- API keys available (optional for basic testing)

## Step 1: Set Up Environment Variables

Create a local environment file:

```bash
# Copy the development environment template
cp .env.development .env.local

# Edit .env.local with your settings
# For basic testing, you can use these minimal settings:
```

**Minimal .env.local for testing:**
```bash
# Database Configuration
POSTGRES_DB=options_db_test
POSTGRES_USER=postgres
POSTGRES_PASSWORD=testpassword
POSTGRES_PORT=5432

# Redis Configuration
REDIS_PORT=6379
REDIS_PASSWORD=

# Backend Configuration
BACKEND_PORT=8000
DEBUG=true

# Application Configuration
APP_NAME=Options Trade Evaluator (Test)

# Cache TTL Settings (shorter for testing)
TICKER_CACHE_TTL=300
ANALYSIS_CACHE_TTL=180
MARKET_DATA_CACHE_TTL=60

# API Keys (optional for basic testing - can be dummy values)
OPENAI_API_KEY=test_key_not_real
ALPHA_VANTAGE_API_KEY=test_key_not_real
TRADIER_API_KEY=test_key_not_real
```

## Step 2: Test Backend Services

### Option A: Using the deployment script (Recommended)

**Windows:**
```cmd
scripts\deploy-dev.bat
```

**Linux/macOS:**
```bash
chmod +x scripts/deploy-dev.sh
./scripts/deploy-dev.sh
```

### Option B: Manual Docker Compose

```bash
# Load environment variables
export $(cat .env.local | grep -v '^#' | xargs)

# Start services
docker-compose --env-file .env.local up --build -d

# Check status
docker-compose ps
```

## Step 3: Verify Backend Services

### Check Service Status
```bash
# View running containers
docker-compose ps

# Check logs
docker-compose logs backend
docker-compose logs postgres
docker-compose logs redis
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Root endpoint
curl http://localhost:8000/

# API documentation (open in browser)
# http://localhost:8000/docs
```

### Using the health check script

**Windows:**
```cmd
scripts\health-check.bat
```

**Linux/macOS:**
```bash
chmod +x scripts/health-check.sh
./scripts/health-check.sh
```

## Step 4: Test Database Connectivity

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d options_db_test

# Inside PostgreSQL, run:
\l                    # List databases
\dt                   # List tables (if any)
\q                    # Quit
```

## Step 5: Test Redis Connectivity

```bash
# Connect to Redis
docker-compose exec redis redis-cli

# Inside Redis, run:
ping                  # Should return PONG
set test "hello"      # Set a test key
get test              # Should return "hello"
exit                  # Exit Redis CLI
```

## Step 6: Monitor Services

Use the monitoring script:
```bash
# Linux/macOS
chmod +x scripts/monitor.sh
./scripts/monitor.sh

# This will show:
# - Container status
# - Resource usage
# - Health checks
# - Interactive commands (r=refresh, l=logs, h=health, q=quit)
```

## Step 7: Test Production Configuration

To test the production Docker setup:

```bash
# Copy production environment
cp .env.production .env.production.local

# Edit .env.production.local with secure values
# Then run:
docker-compose -f docker-compose.prod.yml --env-file .env.production.local up --build -d

# Test with Nginx proxy
curl http://localhost/api/v1/health
```

## Troubleshooting Common Issues

### Port Conflicts
If you get port conflicts:
```bash
# Check what's using the ports
netstat -tulpn | grep :8000
netstat -tulpn | grep :5432
netstat -tulpn | grep :6379

# Change ports in .env.local:
BACKEND_PORT=8001
POSTGRES_PORT=5433
REDIS_PORT=6380
```

### Permission Issues
```bash
# Fix Docker permissions (Linux)
sudo usermod -aG docker $USER
# Then logout and login again

# Or run with sudo
sudo docker-compose up
```

### Database Connection Issues
```bash
# Check if PostgreSQL is ready
docker-compose exec postgres pg_isready -U postgres

# Reset database
docker-compose down -v  # This removes volumes
docker-compose up -d
```

### Memory Issues
```bash
# Check Docker resources
docker system df
docker stats

# Clean up if needed
docker system prune -f
docker volume prune -f
```

## Testing Specific Features

### Test Ticker Search
```bash
# Test ticker search endpoint
curl "http://localhost:8000/api/v1/tickers/search?q=AAPL"
```

### Test Analysis Endpoint
```bash
# Test analysis endpoint (requires valid API keys)
curl -X POST "http://localhost:8000/api/v1/analysis" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "strategy": "covered_call"}'
```

## Cleanup

When you're done testing:

```bash
# Stop services
docker-compose down

# Remove volumes (this deletes data)
docker-compose down -v

# Remove images (optional)
docker-compose down --rmi all
```

## Performance Testing

### Load Testing with curl
```bash
# Simple load test
for i in {1..10}; do
  curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/api/v1/health"
done
```

Create `curl-format.txt`:
```
     time_namelookup:  %{time_namelookup}\n
        time_connect:  %{time_connect}\n
     time_appconnect:  %{time_appconnect}\n
    time_pretransfer:  %{time_pretransfer}\n
       time_redirect:  %{time_redirect}\n
  time_starttransfer:  %{time_starttransfer}\n
                     ----------\n
          time_total:  %{time_total}\n
```

## Next Steps

Once backend testing is successful:
1. Test frontend deployment (see frontend/DEPLOYMENT.md)
2. Test full-stack integration
3. Run end-to-end tests
4. Performance and security testing