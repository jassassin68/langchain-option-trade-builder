-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Stock tickers table
CREATE TABLE IF NOT EXISTS stock_tickers (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name TEXT NOT NULL,
    exchange VARCHAR(10) DEFAULT 'NYSE',
    is_active BOOLEAN DEFAULT true,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_company_name_gin ON stock_tickers USING GIN(company_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_ticker ON stock_tickers(ticker);

-- Analysis cache table
CREATE TABLE IF NOT EXISTS analysis_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(10) NOT NULL,
    analysis_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- Create indexes for cache performance
CREATE INDEX IF NOT EXISTS idx_ticker_cache ON analysis_cache(ticker, expires_at);

-- Insert sample NYSE tickers for testing
INSERT INTO stock_tickers (ticker, company_name, exchange) VALUES
    ('AAPL', 'Apple Inc.', 'NASDAQ'),
    ('MSFT', 'Microsoft Corporation', 'NASDAQ'),
    ('GOOGL', 'Alphabet Inc.', 'NASDAQ'),
    ('AMZN', 'Amazon.com Inc.', 'NASDAQ'),
    ('TSLA', 'Tesla Inc.', 'NASDAQ'),
    ('META', 'Meta Platforms Inc.', 'NASDAQ'),
    ('NVDA', 'NVIDIA Corporation', 'NASDAQ'),
    ('JPM', 'JPMorgan Chase & Co.', 'NYSE'),
    ('JNJ', 'Johnson & Johnson', 'NYSE'),
    ('V', 'Visa Inc.', 'NYSE')
ON CONFLICT (ticker) DO NOTHING;