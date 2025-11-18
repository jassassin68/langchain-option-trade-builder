import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactNode } from 'react';
import { useTickerSearch, useTradeAnalysis, useHealthCheck, useQueryError } from '../useApi';

// Mock the entire API module
jest.mock('@/lib/api', () => ({
  api: {
    searchTickers: jest.fn(),
    analyzeTrade: jest.fn(),
    healthCheck: jest.fn(),
  },
  ApiError: class MockApiError extends Error {
    constructor(
      message: string,
      public status?: number,
      public code?: string,
      public retryAfter?: number
    ) {
      super(message);
      this.name = 'ApiError';
    }
  },
}));

// Import the mocked api after mocking
import { api, ApiError } from '@/lib/api';
const mockedApi = api as jest.Mocked<typeof api>;

// Test wrapper with QueryClient
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });

  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useApi hooks', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('useTickerSearch', () => {
    it('should search tickers successfully', async () => {
      const mockResponse = {
        results: [
          { ticker: 'AAPL', company_name: 'Apple Inc.', exchange: 'NASDAQ' },
        ],
        count: 1,
      };

      mockedApi.searchTickers.mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useTickerSearch('AAPL'), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockResponse);
      expect(mockedApi.searchTickers).toHaveBeenCalledWith('AAPL');
    });

    it('should not query when query is empty', () => {
      const { result } = renderHook(() => useTickerSearch(''), {
        wrapper: createWrapper(),
      });

      // Query should be disabled when query is empty
      expect(result.current.fetchStatus).toBe('idle');
      expect(mockedApi.searchTickers).not.toHaveBeenCalled();
    });

    it('should handle search errors', async () => {
      const error = new ApiError('Search failed', 404);
      mockedApi.searchTickers.mockRejectedValue(error);

      const { result } = renderHook(() => useTickerSearch('INVALID'), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      }, { timeout: 3000 });

      expect(result.current.error).toEqual(error);
    });

    it('should be disabled when enabled is false', () => {
      const { result } = renderHook(() => useTickerSearch('AAPL', false), {
        wrapper: createWrapper(),
      });

      expect(result.current.fetchStatus).toBe('idle');
      expect(mockedApi.searchTickers).not.toHaveBeenCalled();
    });
  });

  describe('useTradeAnalysis', () => {
    it('should analyze trade successfully', async () => {
      const mockResponse = {
        ticker: 'AAPL',
        company_name: 'Apple Inc.',
        recommendation: {
          should_trade: true,
          confidence: 0.85,
          strategy: 'cash-secured-put',
          contracts: [],
          reasoning_steps: [],
        },
      };

      mockedApi.analyzeTrade.mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useTradeAnalysis(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({ ticker: 'AAPL' });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockResponse);
      expect(mockedApi.analyzeTrade).toHaveBeenCalledWith({ ticker: 'AAPL' });
    });

    it('should handle analysis errors', async () => {
      const error = new ApiError('Analysis failed', 422, 'INSUFFICIENT_DATA');
      mockedApi.analyzeTrade.mockRejectedValue(error);

      const { result } = renderHook(() => useTradeAnalysis(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({ ticker: 'INVALID' });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      }, { timeout: 3000 });

      expect(result.current.error).toEqual(error);
    });
  });

  describe('useHealthCheck', () => {
    it('should perform health check successfully', async () => {
      const mockResponse = { status: 'healthy' };
      mockedApi.healthCheck.mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useHealthCheck(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockResponse);
      expect(mockedApi.healthCheck).toHaveBeenCalled();
    });

    it('should be disabled when enabled is false', () => {
      const { result } = renderHook(() => useHealthCheck(false), {
        wrapper: createWrapper(),
      });

      expect(result.current.fetchStatus).toBe('idle');
      expect(mockedApi.healthCheck).not.toHaveBeenCalled();
    });
  });

  describe('useQueryError', () => {
    it('should format ApiError correctly', () => {
      const { result } = renderHook(() => useQueryError());

      const networkError = new ApiError('Network failed', 0, 'NETWORK_ERROR');
      const rateLimitError = new ApiError('Rate limited', 429, 'RATE_LIMIT', 60);
      const tickerError = new ApiError('Not found', 404, 'TICKER_NOT_FOUND');
      const genericError = new ApiError('Generic error', 500);

      expect(result.current.formatError(networkError)).toBe(
        'Unable to connect to the server. Please check your internet connection.'
      );
      expect(result.current.formatError(rateLimitError)).toBe(
        'Too many requests. Please wait 60 seconds before trying again.'
      );
      expect(result.current.formatError(tickerError)).toBe(
        'The ticker symbol was not found. Please check the symbol and try again.'
      );
      expect(result.current.formatError(genericError)).toBe('Generic error');
    });

    it('should identify retryable errors', () => {
      const { result } = renderHook(() => useQueryError());

      const networkError = new ApiError('Network failed', 0, 'NETWORK_ERROR');
      const serverError = new ApiError('Server error', 500);
      const clientError = new ApiError('Bad request', 400);

      expect(result.current.isRetryableError(networkError)).toBe(true);
      expect(result.current.isRetryableError(serverError)).toBe(true);
      expect(result.current.isRetryableError(clientError)).toBe(false);
    });

    it('should handle generic errors', () => {
      const { result } = renderHook(() => useQueryError());

      const genericError = new Error('Generic error');
      const unknownError = 'String error';

      expect(result.current.formatError(genericError)).toBe('Generic error');
      expect(result.current.formatError(unknownError)).toBe(
        'An unexpected error occurred. Please try again.'
      );
    });
  });
});