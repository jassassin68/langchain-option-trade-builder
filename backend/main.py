from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Options Trade Evaluator API...")
    yield
    # Shutdown
    print("Shutting down Options Trade Evaluator API...")

app = FastAPI(
    title="Options Trade Evaluator API",
    description="AI-powered options trading analysis and recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Options Trade Evaluator API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "options-trade-evaluator"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)