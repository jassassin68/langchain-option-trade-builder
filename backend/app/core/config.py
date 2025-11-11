from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    # Database - Supabase compatible
    database_url: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/options_db")
    supabase_url: Optional[str] = os.getenv("SUPABASE_URL")
    supabase_key: Optional[str] = os.getenv("SUPABASE_ANON_KEY")
    
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # OpenAI
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # External APIs
    alpha_vantage_api_key: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
    tradier_api_key: Optional[str] = os.getenv("TRADIER_API_KEY")
    
    # Application
    app_name: str = "Options Trade Evaluator"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Cache TTL (seconds)
    ticker_cache_ttl: int = 3600  # 1 hour
    analysis_cache_ttl: int = 1800  # 30 minutes
    market_data_cache_ttl: int = 300  # 5 minutes
    
    model_config = ConfigDict(env_file=".env")

settings = Settings()