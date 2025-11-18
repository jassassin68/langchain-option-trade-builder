'use client';

import { useState, useCallback } from 'react';
import TickerSearch from '@/components/TickerSearch';
import AnalysisResult from '@/components/AnalysisResult';
import ErrorBoundary from '@/components/ErrorBoundary';
import { AnalysisLoadingSpinner } from '@/components/LoadingSpinner';
import { AnalysisErrorMessage } from '@/components/ErrorMessage';
import { TradeAnalysisResult } from '@/types';
import { api, ApiError } from '@/lib/api';

export default function Home() {
  const [analysisResult, setAnalysisResult] = useState<TradeAnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | Error | ApiError | null>(null);
  const [currentTicker, setCurrentTicker] = useState<string>('');
  const [retryCount, setRetryCount] = useState(0);

  const handleAnalyze = useCallback(async (ticker: string, isRetry: boolean = false) => {
    setIsLoading(true);
    setError(null);
    setCurrentTicker(ticker);
    
    if (!isRetry) {
      setRetryCount(0);
    }
    
    try {
      const result = await api.analyzeTrade({ ticker });
      setAnalysisResult(result);
      setRetryCount(0);
    } catch (err) {
      console.error('Analysis failed:', err);
      setError(err instanceof Error ? err : new Error('Failed to analyze the trade. Please try again.'));
      
      // Auto-retry for network errors (up to 2 times)
      if (err instanceof ApiError && err.code === 'NETWORK_ERROR' && retryCount < 2) {
        setTimeout(() => {
          setRetryCount(prev => prev + 1);
          handleAnalyze(ticker, true);
        }, 2000 * (retryCount + 1)); // Exponential backoff
      }
    } finally {
      setIsLoading(false);
    }
  }, [retryCount]);

  const handleRetry = useCallback(() => {
    if (currentTicker) {
      handleAnalyze(currentTicker, true);
    }
  }, [currentTicker, handleAnalyze]);

  const handleReset = useCallback(() => {
    setAnalysisResult(null);
    setError(null);
    setCurrentTicker('');
    setRetryCount(0);
  }, []);

  const handleInputChange = useCallback(() => {
    setError(null);
  }, []);

  return (
    <ErrorBoundary>
      <main className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6">
            <div className="text-center">
              <h1 className="text-2xl sm:text-3xl font-bold text-gray-900">
                Options Trade Evaluator
              </h1>
              <p className="mt-2 text-base sm:text-lg text-gray-600 px-4 sm:px-0">
                AI-powered options trading analysis and recommendations
              </p>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {isLoading ? (
            /* Loading State */
            <div className="flex flex-col items-center justify-center min-h-[60vh]">
              <AnalysisLoadingSpinner />
              {retryCount > 0 && (
                <div className="mt-4 text-center">
                  <p className="text-sm text-gray-600">
                    Retry attempt {retryCount} of 2...
                  </p>
                </div>
              )}
            </div>
          ) : error ? (
            /* Error State */
            <div className="flex flex-col items-center justify-center min-h-[60vh]">
              <AnalysisErrorMessage 
                error={error}
                onRetry={handleRetry}
                onReset={handleReset}
              />
            </div>
          ) : !analysisResult ? (
            /* Search Interface */
            <div className="flex flex-col items-center justify-center min-h-[60vh]">
              <div className="w-full max-w-2xl px-4 sm:px-0">
                <div className="text-center mb-8">
                  <h2 className="text-xl sm:text-2xl font-semibold text-gray-900 mb-4">
                    Get Started
                  </h2>
                  <p className="text-sm sm:text-base text-gray-600 max-w-lg mx-auto px-4 sm:px-0">
                    Enter a stock ticker to receive comprehensive options trading analysis 
                    including technical indicators, fundamental screening, and strategy recommendations.
                  </p>
                </div>
                
                <TickerSearch onAnalyze={handleAnalyze} isLoading={isLoading} onInputChange={handleInputChange} />
              </div>
            </div>
          ) : (
            /* Results Interface */
            <div className="py-4 sm:py-8">
              <AnalysisResult result={analysisResult} onReset={handleReset} />
            </div>
          )}
        </div>

        {/* Footer */}
        <footer className="bg-white border-t border-gray-200 mt-8 sm:mt-16">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
            <div className="text-center text-gray-500 text-xs sm:text-sm">
              <p>
                Options Trade Evaluator - AI-powered trading analysis
              </p>
              <p className="mt-1">
                This tool provides educational analysis only. Not financial advice.
              </p>
            </div>
          </div>
        </footer>
      </main>
    </ErrorBoundary>
  );
}