# Options Trade Evaluator

AI-powered options trading analysis and recommendations using LangChain and modern web technologies.

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

## Environment Variables

### Backend (.env)
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `OPENAI_API_KEY`: OpenAI API key for LLM
- `ALPHA_VANTAGE_API_KEY`: Market data API key
- `TRADIER_API_KEY`: Options data API key

### Frontend (.env.local)
- `NEXT_PUBLIC_API_URL`: Backend API URL

## Development

### Running Tests
```bash
# Backend tests
cd backend
pytest

# Frontend tests (when implemented)
cd frontend
npm test
```

### Database Migrations
The initial schema is automatically applied when using Docker Compose. For manual setup:
```bash
psql -h localhost -U postgres -d options_db -f backend/migrations/001_initial_schema.sql
```

## Architecture

The application follows a microservices architecture with:
- **Frontend**: React-based UI with TypeScript
- **Backend**: FastAPI with LangChain agent orchestration
- **Database**: PostgreSQL with Redis caching
- **External APIs**: Market data and options data providers

## Contributing

1. Follow the existing code style and structure
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all services start successfully with Docker Compose