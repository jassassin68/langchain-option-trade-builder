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
    
    # OpenAI (kept for backward compatibility/rollback)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # OpenRouter Configuration
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    app_url: str = os.getenv("APP_URL", "http://localhost:3000")
    
    # LLM Model Configuration
    primary_model: str = os.getenv("PRIMARY_MODEL", "x-ai/grok-4.1-fast:free")
    fallback_model: str = os.getenv("FALLBACK_MODEL", "mistralai/mistral-small-3.2-24b-instruct:free")
    
    # LLM Request Configuration
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "30"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4000"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))
    
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