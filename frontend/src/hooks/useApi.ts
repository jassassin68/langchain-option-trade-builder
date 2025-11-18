import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '@/lib/api';
import { TickerSearchResponse, TradeAnalysisRequest, TradeAnalysisResult } from '@/types';

// Query keys for React Query
export const queryKeys = {
  tickerSearch: (query: string) => ['ticker-search', query] as const,
  tradeAnalysis: (ticker: string) => ['trade-analysis', ticker] as const,
  healthCheck: () => ['health-check'] as const,
};

/**
 * Hook for searching stock tickers with debounced queries
 */
export const useTickerSearch = (query: string, enabled: boolean = true) => {
  return useQuery({
    queryKey: queryKeys.tickerSearch(query),
    queryFn: () => api.searchTickers(query),
    enabled: enabled && query.length > 0,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes (formerly cacheTime)
    retry: (failureCount, error) => {
      // Don't retry on client errors (4xx)
      if (error instanceof ApiError && error.status && error.status >= 400 && error.status < 500) {
        return false;
      }
      // Retry up to 2 times for other errors
      return failureCount < 2;
    },
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000), // Exponential backoff
  });
};

/**
 * Hook for trade analysis with mutation
 */
export const useTradeAnalysis = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: TradeAnalysisRequest) => api.analyzeTrade(request),
    retry: (failureCount, error) => {
      // Don't retry on client errors (4xx) or specific server errors
      if (error instanceof ApiError) {
        if (error.status && error.status >= 400 && error.status < 500) {
          return false;
        }
        // Don't retry if server explicitly says not to
        if (error.code === 'NO_RETRY') {
          return false;
        }
      }
      // Retry up to 1 time for analysis (it's expensive)
      return failureCount < 1;
    },
    retryDelay: (attemptIndex) => {
      // Use retry_after from server if available
      const error = attemptIndex > 0 ? undefined : undefined; // Get from context if needed
      return 2000; // Default 2 second delay
    },
    onSuccess: (data, variables) => {
      // Cache the successful analysis result
      queryClient.setQueryData(
        queryKeys.tradeAnalysis(variables.ticker),
        data
      );
    },
    onError: (error) => {
      console.error('Trade analysis mutation failed:', error);
    },
  });
};

/**
 * Hook for getting cached trade analysis results
 */
export const useTradeAnalysisQuery = (ticker: string, enabled: boolean = false) => {
  return useQuery({
    queryKey: queryKeys.tradeAnalysis(ticker),
    queryFn: () => api.analyzeTrade({ ticker }),
    enabled: enabled && ticker.length > 0,
    staleTime: 30 * 60 * 1000, // 30 minutes - analysis results are expensive
    gcTime: 60 * 60 * 1000, // 1 hour
    retry: false, // Don't auto-retry queries, use mutation for that
  });
};

/**
 * Hook for health check monitoring
 */
export const useHealthCheck = (enabled: boolean = true) => {
  return useQuery({
    queryKey: queryKeys.healthCheck(),
    queryFn: () => api.healthCheck(),
    enabled,
    staleTime: 30 * 1000, // 30 seconds
    gcTime: 60 * 1000, // 1 minute
    retry: 3,
    retryDelay: 1000,
    refetchInterval: 60 * 1000, // Check every minute
    refetchIntervalInBackground: false,
  });
};

/**
 * Utility hook to invalidate all queries (useful for error recovery)
 */
export const useInvalidateQueries = () => {
  const queryClient = useQueryClient();

  return {
    invalidateTickerSearch: (query?: string) => {
      if (query) {
        queryClient.invalidateQueries({ queryKey: queryKeys.tickerSearch(query) });
      } else {
        queryClient.invalidateQueries({ queryKey: ['ticker-search'] });
      }
    },
    invalidateTradeAnalysis: (ticker?: string) => {
      if (ticker) {
        queryClient.invalidateQueries({ queryKey: queryKeys.tradeAnalysis(ticker) });
      } else {
        queryClient.invalidateQueries({ queryKey: ['trade-analysis'] });
      }
    },
    invalidateAll: () => {
      queryClient.invalidateQueries();
    },
  };
};

/**
 * Hook for managing query error states
 */
export const useQueryError = () => {
  const formatError = (error: unknown): string => {
    if (error instanceof ApiError) {
      switch (error.code) {
        case 'NETWORK_ERROR':
          return 'Unable to connect to the server. Please check your internet connection.';
        case 'RATE_LIMIT':
          return `Too many requests. Please wait ${error.retryAfter || 60} seconds before trying again.`;
        case 'TICKER_NOT_FOUND':
          return 'The ticker symbol was not found. Please check the symbol and try again.';
        case 'INSUFFICIENT_DATA':
          return 'Insufficient data available for analysis. Please try a different ticker.';
        default:
          return error.message || 'An unexpected error occurred.';
      }
    }
    
    if (error instanceof Error) {
      return error.message;
    }
    
    return 'An unexpected error occurred. Please try again.';
  };

  const isRetryableError = (error: unknown): boolean => {
    if (error instanceof ApiError) {
      // Network errors and 5xx server errors are retryable
      return error.code === 'NETWORK_ERROR' || (error.status !== undefined && error.status >= 500);
    }
    return false;
  };

  return { formatError, isRetryableError };
};