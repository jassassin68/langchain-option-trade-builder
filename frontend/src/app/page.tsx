'use client';

import { useState } from 'react';
import TickerSearch from '@/components/TickerSearch';
import AnalysisResult from '@/components/AnalysisResult';
import ErrorBoundary from '@/components/ErrorBoundary';
import { TradeAnalysisResult } from '@/types';
import { api } from '@/lib/api';

export default function Home() {
  const [analysisResult, setAnalysisResult] = useState<TradeAnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async (ticker: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await api.analyzeTrade({ ticker });
      setAnalysisResult(result);
    } catch (err) {
      console.error('Analysis failed:', err);
      setError('Failed to analyze the trade. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setAnalysisResult(null);
    setError(null);
  };

  const handleInputChange = () => {
    setError(null);
  };

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
          {!analysisResult ? (
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
                
                {error && (
                  <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-md mx-4 sm:mx-0">
                    <div className="flex">
                      <div className="flex-shrink-0">
                        <svg
                          className="h-5 w-5 text-red-400"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                          aria-hidden="true"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
                          />
                        </svg>
                      </div>
                      <div className="ml-3">
                        <h3 className="text-sm font-medium text-red-800">
                          Analysis Error
                        </h3>
                        <p className="mt-1 text-sm text-red-700">{error}</p>
                      </div>
                    </div>
                  </div>
                )}
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