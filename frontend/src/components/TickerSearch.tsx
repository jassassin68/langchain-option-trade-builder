'use client';

import React, { useState, useRef, useEffect, KeyboardEvent } from 'react';
import { useDebounce } from '@/hooks/useDebounce';
import { api } from '@/lib/api';
import { TickerResult } from '@/types';

interface TickerSearchProps {
  onAnalyze: (ticker: string) => void;
  isLoading: boolean;
  onInputChange?: () => void;
}

export default function TickerSearch({ onAnalyze, isLoading, onInputChange }: TickerSearchProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<TickerResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [error, setError] = useState<string | null>(null);
  const [selectedTicker, setSelectedTicker] = useState<string>('');

  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const debouncedQuery = useDebounce(query, 300);

  // Search for tickers when debounced query changes
  useEffect(() => {
    const searchTickers = async () => {
      if (debouncedQuery.trim().length === 0) {
        setResults([]);
        setShowDropdown(false);
        setError(null);
        return;
      }

      setIsSearching(true);
      setError(null);

      try {
        const response = await api.searchTickers(debouncedQuery.trim());
        setResults(response.results);
        setShowDropdown(response.results.length > 0);
        setSelectedIndex(-1);
      } catch (err) {
        setError('Failed to search tickers. Please try again.');
        setResults([]);
        setShowDropdown(false);
      } finally {
        setIsSearching(false);
      }
    };

    searchTickers();
  }, [debouncedQuery]);

  // Handle input change
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.toUpperCase();
    setQuery(value);
    setSelectedTicker('');
    onInputChange?.();
  };

  // Handle keyboard navigation
  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (!showDropdown || results.length === 0) {
      if (e.key === 'Enter' && selectedTicker) {
        handleAnalyze();
      }
      return;
    }

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(prev => 
          prev < results.length - 1 ? prev + 1 : 0
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(prev => 
          prev > 0 ? prev - 1 : results.length - 1
        );
        break;
      case 'Enter':
        e.preventDefault();
        if (selectedIndex >= 0 && selectedIndex < results.length) {
          selectTicker(results[selectedIndex]);
        } else if (selectedTicker) {
          handleAnalyze();
        }
        break;
      case 'Escape':
        setShowDropdown(false);
        setSelectedIndex(-1);
        inputRef.current?.blur();
        break;
    }
  };

  // Select a ticker from dropdown
  const selectTicker = (ticker: TickerResult) => {
    setQuery(ticker.ticker);
    setSelectedTicker(ticker.ticker);
    setShowDropdown(false);
    setSelectedIndex(-1);
    inputRef.current?.focus();
  };

  // Handle analyze button click
  const handleAnalyze = () => {
    if (selectedTicker && !isLoading) {
      onAnalyze(selectedTicker);
    }
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(event.target as Node)
      ) {
        setShowDropdown(false);
        setSelectedIndex(-1);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="w-full max-w-md mx-auto">
      <div className="relative">
        {/* Search Input */}
        <div className="relative">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Enter stock ticker (e.g., AAPL)"
            className="w-full px-4 py-3 text-lg border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none transition-colors"
            disabled={isLoading}
          />
          {isSearching && (
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500"></div>
            </div>
          )}
        </div>

        {/* Dropdown */}
        {showDropdown && results.length > 0 && (
          <div
            ref={dropdownRef}
            className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto"
          >
            {results.map((result, index) => (
              <div
                key={result.ticker}
                onClick={() => selectTicker(result)}
                className={`px-4 py-3 cursor-pointer transition-colors ${
                  index === selectedIndex
                    ? 'bg-blue-100 text-blue-900'
                    : 'hover:bg-gray-100'
                }`}
              >
                <div className="font-semibold text-gray-900">
                  {result.ticker}
                </div>
                <div className="text-sm text-gray-600 truncate">
                  {result.company_name}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="mt-2 text-sm text-red-600 bg-red-50 border border-red-200 rounded-md px-3 py-2">
          {error}
        </div>
      )}

      {/* Analyze Button */}
      <button
        onClick={handleAnalyze}
        disabled={!selectedTicker || isLoading}
        className={`w-full mt-4 px-6 py-3 text-lg font-semibold rounded-lg transition-all ${
          selectedTicker && !isLoading
            ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-md hover:shadow-lg'
            : 'bg-gray-300 text-gray-500 cursor-not-allowed'
        }`}
      >
        {isLoading ? (
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
            Analyzing...
          </div>
        ) : (
          'Analyze Trade'
        )}
      </button>

      {/* Input Validation Message */}
      {query && !selectedTicker && !isSearching && results.length === 0 && (
        <div className="mt-2 text-sm text-gray-600">
          Please select a ticker from the dropdown or enter a valid ticker symbol.
        </div>
      )}
    </div>
  );
}