'use client';

import React from 'react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
  className?: string;
}

export default function LoadingSpinner({ 
  size = 'md', 
  text = 'Loading...', 
  className = '' 
}: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8'
  };

  const textSizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg'
  };

  return (
    <div className={`flex items-center justify-center ${className}`}>
      <div className="flex items-center space-x-2">
        <div
          className={`animate-spin rounded-full border-2 border-gray-300 border-t-blue-600 ${sizeClasses[size]}`}
          role="status"
          aria-label="Loading"
        />
        {text && (
          <span className={`text-gray-600 ${textSizeClasses[size]}`}>
            {text}
          </span>
        )}
      </div>
    </div>
  );
}

// Specialized component for analysis loading
export function AnalysisLoadingSpinner({ className = '' }: { className?: string }) {
  return (
    <div className={`flex flex-col items-center justify-center py-8 ${className}`}>
      <div className="relative">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-gray-200 border-t-blue-600"></div>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="animate-pulse h-4 w-4 bg-blue-600 rounded-full"></div>
        </div>
      </div>
      <div className="mt-4 text-center">
        <p className="text-lg font-medium text-gray-900">Analyzing...</p>
        <p className="text-sm text-gray-600 mt-1">
          Evaluating technical indicators, fundamentals, and options data
        </p>
      </div>
    </div>
  );
}