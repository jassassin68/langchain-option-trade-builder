'use client';

import React from 'react';
import { ApiError } from '@/lib/api';

interface ErrorMessageProps {
  error: string | Error | ApiError;
  onRetry?: () => void;
  onDismiss?: () => void;
  className?: string;
  variant?: 'error' | 'warning' | 'info';
}

export default function ErrorMessage({ 
  error, 
  onRetry, 
  onDismiss, 
  className = '',
  variant = 'error'
}: ErrorMessageProps) {
  const formatErrorMessage = (error: string | Error | ApiError): { title: string; message: string; canRetry: boolean } => {
    if (typeof error === 'string') {
      return { title: 'Error', message: error, canRetry: true };
    }

    if (error && typeof error === 'object' && 'code' in error && 'status' in error) {
      const apiError = error as ApiError;
      switch (apiError.code) {
        case 'NETWORK_ERROR':
          return {
            title: 'Connection Error',
            message: 'Unable to connect to the server. Please check your internet connection and try again.',
            canRetry: true
          };
        case 'RATE_LIMIT':
          return {
            title: 'Rate Limited',
            message: `Too many requests. Please wait ${apiError.retryAfter || 60} seconds before trying again.`,
            canRetry: true
          };
        case 'TICKER_NOT_FOUND':
          return {
            title: 'Ticker Not Found',
            message: 'The ticker symbol was not found. Please check the symbol and try again.',
            canRetry: false
          };
        case 'INSUFFICIENT_DATA':
          return {
            title: 'Insufficient Data',
            message: 'Not enough data available for analysis. Please try a different ticker.',
            canRetry: false
          };
        default:
          return {
            title: 'API Error',
            message: apiError.message || 'An unexpected error occurred.',
            canRetry: apiError.status ? apiError.status >= 500 : true
          };
      }
    }

    if (error instanceof Error) {
      return {
        title: 'Error',
        message: error.message,
        canRetry: true
      };
    }

    return {
      title: 'Unknown Error',
      message: 'An unexpected error occurred.',
      canRetry: true
    };
  };

  const { title, message, canRetry } = formatErrorMessage(error);

  const variantStyles = {
    error: {
      container: 'bg-red-50 border-red-200 text-red-800',
      icon: 'text-red-400',
      button: 'bg-red-600 hover:bg-red-700 text-white'
    },
    warning: {
      container: 'bg-yellow-50 border-yellow-200 text-yellow-800',
      icon: 'text-yellow-400',
      button: 'bg-yellow-600 hover:bg-yellow-700 text-white'
    },
    info: {
      container: 'bg-blue-50 border-blue-200 text-blue-800',
      icon: 'text-blue-400',
      button: 'bg-blue-600 hover:bg-blue-700 text-white'
    }
  };

  const styles = variantStyles[variant];

  const getIcon = () => {
    switch (variant) {
      case 'warning':
        return (
          <svg className={`h-5 w-5 ${styles.icon}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        );
      case 'info':
        return (
          <svg className={`h-5 w-5 ${styles.icon}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      default:
        return (
          <svg className={`h-5 w-5 ${styles.icon}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        );
    }
  };

  return (
    <div className={`border rounded-md p-4 ${styles.container} ${className}`}>
      <div className="flex">
        <div className="flex-shrink-0">
          {getIcon()}
        </div>
        <div className="ml-3 flex-1">
          <h3 className="text-sm font-medium">{title}</h3>
          <p className="mt-1 text-sm">{message}</p>
          
          {(onRetry || onDismiss) && (
            <div className="mt-3 flex space-x-2">
              {onRetry && canRetry && (
                <button
                  onClick={onRetry}
                  className={`inline-flex items-center px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${styles.button}`}
                >
                  <svg className="h-4 w-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Try Again
                </button>
              )}
              {onDismiss && (
                <button
                  onClick={onDismiss}
                  className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-gray-600 hover:text-gray-800 transition-colors"
                >
                  Dismiss
                </button>
              )}
            </div>
          )}
        </div>
        
        {onDismiss && (
          <div className="ml-auto pl-3">
            <button
              onClick={onDismiss}
              className={`inline-flex rounded-md p-1.5 hover:bg-opacity-20 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-red-50 focus:ring-red-600 ${styles.icon}`}
            >
              <span className="sr-only">Dismiss</span>
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// Specialized error component for analysis failures
export function AnalysisErrorMessage({ 
  error, 
  onRetry, 
  onReset,
  className = '' 
}: { 
  error: string | Error | ApiError;
  onRetry?: () => void;
  onReset?: () => void;
  className?: string;
}) {
  return (
    <div className={`text-center py-8 ${className}`}>
      <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-red-100 mb-4">
        <svg className="h-6 w-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
        </svg>
      </div>
      
      <h3 className="text-lg font-medium text-gray-900 mb-2">Analysis Failed</h3>
      
      <div className="max-w-md mx-auto mb-6">
        <ErrorMessage 
          error={error} 
          onRetry={onRetry}
          variant="error"
          className="text-left"
        />
      </div>
      
      <div className="flex justify-center space-x-3">
        {onRetry && (
          <button
            onClick={onRetry}
            className="inline-flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-md transition-colors"
          >
            <svg className="h-4 w-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Retry Analysis
          </button>
        )}
        {onReset && (
          <button
            onClick={onReset}
            className="inline-flex items-center px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white font-medium rounded-md transition-colors"
          >
            Try Different Ticker
          </button>
        )}
      </div>
    </div>
  );
}