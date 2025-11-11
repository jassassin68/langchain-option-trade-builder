from sqlalchemy import Column, String, Boolean, DateTime, Text, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.core.database import Base
import uuid

class StockTicker(Base):
    __tablename__ = "stock_tickers"
    
    ticker = Column(String(10), primary_key=True)
    company_name = Column(Text, nullable=False)
    exchange = Column(String(10), default="NYSE")
    is_active = Column(Boolean, default=True)
    last_updated = Column(DateTime(timezone=True), server_default=func.now())
    
    # Create indexes for performance
    __table_args__ = (
        Index('idx_company_name_gin', 'company_name', postgresql_using='gin', 
              postgresql_ops={'company_name': 'gin_trgm_ops'}),
        Index('idx_ticker', 'ticker'),
    )

class AnalysisCache(Base):
    __tablename__ = "analysis_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(10), nullable=False)
    analysis_data = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    
    # Create indexes for performance
    __table_args__ = (
        Index('idx_ticker_cache', 'ticker', 'expires_at'),
    )