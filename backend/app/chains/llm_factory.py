"""
LLM Factory for creating and managing LLM instances with OpenRouter integration.

Implements requirements 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 4.2, 4.3, 4.4, 4.5:
- Configure OpenRouter as LLM provider with API credentials
- Use Grok 4.1 Fast as primary model for all analysis chains
- Provide centralized LLM Factory to manage model instantiation
- Configure consistent temperature, max_tokens, and timeout across models
- Include HTTP-Referer and X-Title headers for OpenRouter ranking
"""

import logging
from langchain_openai import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory for creating LLM instances with OpenRouter integration.
    
    Provides static methods to create primary and fallback LLM instances
    with consistent configuration for temperature, max_tokens, and timeout.
    """
    
    @staticmethod
    def create_primary_llm() -> ChatOpenAI:
        """
        Create primary LLM instance (Grok 4.1 Fast free tier).
        
        Configures ChatOpenAI instance with:
        - Model: x-ai/grok-4.1-fast:free
        - OpenRouter base URL and API key
        - Temperature: 0.0 (deterministic outputs)
        - Max tokens: 4000 (comprehensive responses)
        - Timeout: 30 seconds
        - HTTP-Referer and X-Title headers for OpenRouter ranking
        
        Returns:
            ChatOpenAI instance configured for x-ai/grok-4.1-fast:free
        
        Raises:
            ValueError: If OPENROUTER_API_KEY is not configured
        """
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured")
        
        logger.info(f"Creating primary LLM instance: {settings.primary_model}")
        
        return ChatOpenAI(
            model=settings.primary_model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            request_timeout=settings.llm_timeout,
            model_kwargs={
                "headers": {
                    "HTTP-Referer": settings.app_url,
                    "X-Title": settings.app_name
                }
            }
        )
    
    @staticmethod
    def create_fallback_llm() -> ChatOpenAI:
        """
        Create fallback LLM instance (Mistral Small 3.2 24B Instruct free tier).
        
        Configures ChatOpenAI instance with:
        - Model: mistralai/mistral-small-3.2-24b-instruct:free
        - OpenRouter base URL and API key
        - Temperature: 0.0 (deterministic outputs)
        - Max tokens: 4000 (comprehensive responses)
        - Timeout: 30 seconds
        - HTTP-Referer and X-Title headers for OpenRouter ranking
        
        Returns:
            ChatOpenAI instance configured for mistralai/mistral-small-3.2-24b-instruct:free
        
        Raises:
            ValueError: If OPENROUTER_API_KEY is not configured
        """
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured")
        
        logger.info(f"Creating fallback LLM instance: {settings.fallback_model}")
        
        return ChatOpenAI(
            model=settings.fallback_model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            request_timeout=settings.llm_timeout,
            model_kwargs={
                "headers": {
                    "HTTP-Referer": settings.app_url,
                    "X-Title": settings.app_name
                }
            }
        )
    
    @staticmethod
    def create_llm_with_fallback() -> ChatOpenAI:
        """
        Create LLM with automatic fallback on initialization failure.
        
        Attempts to create primary LLM instance. If initialization fails,
        automatically falls back to creating fallback LLM instance.
        
        This method implements initialization-time fallback logic to ensure
        the application can start even if the primary model is unavailable.
        
        Returns:
            ChatOpenAI instance (primary if successful, fallback if primary fails)
        
        Raises:
            Exception: If both primary and fallback LLM initialization fail
        """
        try:
            logger.info("Attempting to create primary LLM instance")
            return LLMFactory.create_primary_llm()
        except Exception as e:
            logger.warning(f"Failed to initialize primary LLM: {e}")
            logger.info(f"Falling back to {settings.fallback_model}")
            
            try:
                return LLMFactory.create_fallback_llm()
            except Exception as fallback_error:
                logger.error(f"Failed to initialize fallback LLM: {fallback_error}")
                raise Exception(
                    f"Both primary and fallback LLM initialization failed. "
                    f"Primary: {str(e)}, Fallback: {str(fallback_error)}"
                )
