# Options Trade Evaluator

AI-powered options trading analysis and recommendations using LangChain and modern web technologies.

## Overview

Options Trade Evaluator is an intelligent trading assistant that helps traders make informed decisions about options trades. The system analyzes stocks through multiple lenses — technical indicators, fundamental metrics, options data, and risk assessment — to provide actionable trade recommendations with confidence scores.

**Key Features:**
- Real-time ticker search with autocomplete
- Multi-stage AI analysis using LangChain agents
- Comprehensive risk metrics (max profit/loss, breakeven, ROI)
- Specific contract recommendations with strike prices and expiration dates
- Clear YES/NO recommendations with detailed reasoning

**Target Users:** Options traders seeking data-driven insights and risk analysis before entering trades.

## Project Structure

```
├── frontend/          # Next.js frontend application
├── backend/           # FastAPI backend application
├── docker-compose.yml # Local development environment
└── README.md         # This file
```

## Technology Stack

### Frontend
- Next.js 14 with App Router
- TypeScript
- TailwindCSS
- React Query for state management

### Backend
- FastAPI with Python 3.11+
- LangChain for LLM orchestration
- SQLAlchemy with AsyncPG
- Redis for caching
- PostgreSQL database

## Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- Docker and Docker Compose (for local development)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd options-trade-evaluator
   ```

2. **Set up environment variables**
   ```bash
   # Backend
   cp backend/.env.example backend/.env
   # Edit backend/.env with your API keys
   
   # Frontend
   cp frontend/.env.local.example frontend/.env.local
   ```

3. **Start services with Docker Compose**
   ```bash
   docker-compose up -d postgres redis
   ```

4. **Install and run backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

5. **Install and run frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

### API Documentation

Once the backend is running, visit:
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Frontend Application

The frontend will be available at: http://localhost:3000

## API Endpoints

### Health & Monitoring
- `GET /health` - Health check with service status
- `GET /api/v1/metrics` - Application metrics (placeholder)

### Ticker Search
- `GET /api/v1/tickers/search?q={query}&limit={limit}`
  - Search for stock tickers with autocomplete
  - **Parameters:**
    - `q` (required): Search query (min 1 character)
    - `limit` (optional): Max results (default 10, max 50)
  - **Response:** List of matching tickers with company names
  - **Performance:** <300ms response time

### Trade Analysis
- `POST /api/v1/trades/analyze`
  - Perform comprehensive options trade analysis
  - **Request Body:**
    ```json
    {
      "ticker": "AAPL"
    }
    ```
  - **Response:** Complete analysis with recommendation, confidence score, risk metrics, and specific contract details
  - **Performance:** ~5 seconds average (includes multiple API calls and LLM analysis)
  - **Error Codes:**
    - `400` - Invalid request
    - `404` - Ticker not found
    - `422` - Data unavailable
    - `503` - External service unavailable

## Environment Variables

### Backend (.env)
Copy `backend/.env.example` to `backend/.env` and configure:

- `DATABASE_URL` - PostgreSQL connection string
  - Format: `postgresql+asyncpg://user:password@host:port/database`
  - Example: `postgresql+asyncpg://postgres:postgres@localhost:5432/options_db`
  
- `REDIS_URL` - Redis connection string for caching
  - Format: `redis://host:port/db`
  - Example: `redis://localhost:6379/0`
  
- `OPENAI_API_KEY` - OpenAI API key for LLM analysis (required)
  - Get from: https://platform.openai.com/api-keys
  - Used for: LangChain agent reasoning and analysis
  
- `ALPHA_VANTAGE_API_KEY` - Market data API key (required)
  - Get from: https://www.alphavantage.co/support/#api-key
  - Used for: Stock prices, technical indicators
  - Free tier: 25 requests/day
  
- `TRADIER_API_KEY` - Options data API key (required)
  - Get from: https://developer.tradier.com/
  - Used for: Options chains, Greeks, pricing
  - Sandbox available for testing

### Frontend (.env.local)
Copy `frontend/.env.local.example` to `frontend/.env.local` and configure:

- `NEXT_PUBLIC_API_URL` - Backend API URL
  - Development: `http://localhost:8000`
  - Production: Your deployed backend URL

## Code Structure

### Backend (`/backend`)
```
backend/
├── app/
│   ├── api/v1/          # API endpoints
│   │   ├── health.py    # Health check endpoints
│   │   ├── tickers.py   # Ticker search endpoints
│   │   └── trades.py    # Trade analysis endpoints
│   ├── chains/          # LangChain analysis chains
│   │   ├── options_evaluation_agent.py  # Main orchestration agent
│   │   ├── technical_analysis_chain.py
│   │   ├── fundamental_screening_chain.py
│   │   ├── options_analysis_chain.py
│   │   ├── strategy_selection_chain.py
│   │   └── risk_assessment_chain.py
│   ├── core/            # Core configuration
│   │   ├── config.py    # Settings and environment
│   │   └── database.py  # Database connection
│   ├── models/          # Data models
│   │   ├── api.py       # API request/response models
│   │   └── database.py  # SQLAlchemy models
│   └── services/        # Business logic
│       ├── ticker_service.py
│       ├── market_data_service.py
│       ├── options_data_service.py
│       └── cache_service.py
├── migrations/          # Database migrations
├── tests/              # Test suite
├── main.py             # FastAPI application entry
└── requirements.txt    # Python dependencies
```

### Frontend (`/frontend`)
```
frontend/
├── src/
│   ├── app/            # Next.js App Router
│   │   ├── page.tsx    # Main page
│   │   └── layout.tsx  # Root layout
│   ├── components/     # React components
│   │   ├── TickerSearch.tsx      # Autocomplete search
│   │   ├── AnalysisResult.tsx    # Results display
│   │   ├── LoadingSpinner.tsx
│   │   ├── ErrorMessage.tsx
│   │   └── ErrorBoundary.tsx
│   ├── hooks/          # Custom React hooks
│   │   ├── useApi.ts   # API integration
│   │   └── useDebounce.ts
│   ├── lib/            # Utilities
│   │   └── api.ts      # API client
│   └── types/          # TypeScript types
│       └── index.ts
├── __tests__/          # Test files
└── package.json        # Node dependencies
```

## Development

### Running Tests
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Database Migrations
The initial schema is automatically applied when using Docker Compose. For manual setup:
```bash
psql -h localhost -U postgres -d options_db -f backend/migrations/001_initial_schema.sql
```

### Common Development Commands
```bash
# Backend
cd backend
pip install -r requirements.txt          # Install dependencies
uvicorn main:app --reload                # Run dev server
pytest                                   # Run tests
pytest -v                                # Verbose test output
pytest tests/test_api_health.py          # Run specific test

# Frontend
cd frontend
npm install                              # Install dependencies
npm run dev                              # Run dev server
npm run build                            # Production build
npm test                                 # Run tests
npm run lint                             # Run linter
```

## Architecture

The application follows a microservices architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Ticker Search│  │   Analysis   │  │    Results   │          │
│  │  Component   │  │   Display    │  │   Display    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST API
┌────────────────────────────┴────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Options Evaluation Agent (LangChain)           │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │Technical │ │Fundamental│ │ Options  │ │   Risk   │   │   │
│  │  │ Analysis │→│ Screening │→│ Analysis │→│Assessment│   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Ticker     │  │ Market Data  │  │Options Data  │          │
│  │   Service    │  │   Service    │  │   Service    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│   PostgreSQL   │  │     Redis       │  │  External APIs │
│   (Tickers)    │  │    (Cache)      │  │ Alpha Vantage  │
│                │  │                 │  │    Tradier     │
└────────────────┘  └─────────────────┘  └────────────────┘
```

**Component Responsibilities:**
- **Frontend**: User interface, form validation, result visualization
- **Backend API**: Request handling, orchestration, error management
- **LangChain Agent**: Sequential analysis workflow with LLM reasoning
- **Services**: Data fetching, caching, database operations
- **Database**: Ticker storage and query optimization
- **Cache**: API response caching to reduce external API calls

## Docker Commands

### Starting Services
```bash
# Start all services (PostgreSQL, Redis, Backend, Frontend)
docker-compose up -d

# Start specific services only
docker-compose up -d postgres redis

# View logs
docker-compose logs -f                   # All services
docker-compose logs -f backend           # Backend only
docker-compose logs -f frontend          # Frontend only

# Check service status
docker-compose ps
```

### Stopping and Cleanup
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clears database)
docker-compose down -v

# Rebuild containers after code changes
docker-compose up -d --build

# Remove all containers and images
docker-compose down --rmi all
```

### Troubleshooting Docker
```bash
# View container logs
docker-compose logs backend --tail=100

# Access container shell
docker-compose exec backend bash
docker-compose exec postgres psql -U postgres -d options_db

# Restart specific service
docker-compose restart backend

# Check container resource usage
docker stats
```

## Troubleshooting

### Common Issues

**Backend won't start:**
- Check `.env` file exists and has all required variables
- Verify PostgreSQL is running: `docker-compose ps postgres`
- Check database connection: `docker-compose logs postgres`
- Ensure port 8000 is not in use: `netstat -ano | findstr :8000` (Windows)

**Frontend won't start:**
- Check `.env.local` exists with `NEXT_PUBLIC_API_URL`
- Verify Node.js version: `node --version` (requires 18+)
- Clear Next.js cache: `rm -rf frontend/.next`
- Ensure port 3000 is not in use

**Database connection errors:**
- Verify PostgreSQL is running: `docker-compose ps postgres`
- Check connection string in `backend/.env`
- Test connection: `docker-compose exec postgres psql -U postgres -d options_db`
- Reset database: `docker-compose down -v && docker-compose up -d postgres`

**API key errors:**
- Verify all API keys are set in `backend/.env`
- Check OpenAI key: https://platform.openai.com/api-keys
- Check Alpha Vantage key: https://www.alphavantage.co/support/#api-key
- Check Tradier key: https://developer.tradier.com/
- Note: Free tier rate limits may cause errors

**Redis connection errors:**
- Verify Redis is running: `docker-compose ps redis`
- Check Redis URL in `backend/.env`
- Test connection: `docker-compose exec redis redis-cli ping`

**Slow API responses:**
- Check external API rate limits (Alpha Vantage: 25/day free tier)
- Verify Redis cache is working: `docker-compose logs redis`
- Monitor backend logs: `docker-compose logs -f backend`

**CORS errors:**
- Verify `NEXT_PUBLIC_API_URL` in frontend `.env.local`
- Check CORS settings in `backend/main.py`
- Ensure frontend URL is in allowed origins

## Deployment

For production deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Contributing

1. Follow the existing code style and structure
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all services start successfully with Docker Compose