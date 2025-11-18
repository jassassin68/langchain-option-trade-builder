/**
 * Accessibility tests for components
 * Tests keyboard navigation and screen reader support as required by 1.5, 8.1, 8.2, 8.3, 8.4
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import TickerSearch from '../TickerSearch';
import AnalysisResult from '../AnalysisResult';
import ErrorMessage from '../ErrorMessage';
import { TradeAnalysisResult } from '@/types';
import { api } from '@/lib/api';

// Mock the API module
jest.mock('@/lib/api', () => ({
  api: {
    searchTickers: jest.fn(),
  },
}));

const mockApi = api as jest.Mocked<typeof api>;

describe('Accessibility Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('TickerSearch Accessibility', () => {
    const mockOnAnalyze = jest.fn();
    const mockTickerResults = [
      { ticker: 'AAPL', company_name: 'Apple Inc.', exchange: 'NASDAQ' },
      { ticker: 'GOOGL', company_name: 'Alphabet Inc.', exchange: 'NASDAQ' },
      { ticker: 'MSFT', company_name: 'Microsoft Corporation', exchange: 'NASDAQ' },
    ];

    beforeEach(() => {
      mockApi.searchTickers.mockResolvedValue({
        results: mockTickerResults,
        count: mockTickerResults.length,
      });
    });

    it('has proper ARIA labels and roles', () => {
      render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
      
      const input = screen.getByRole('combobox');
      expect(input).toHaveAttribute('aria-label', 'Search for stock ticker');
      expect(input).toHaveAttribute('aria-expanded', 'false');
      expect(input).toHaveAttribute('aria-autocomplete', 'list');
      
      const button = screen.getByRole('button', { name: 'Analyze Trade' });
      expect(button).toBeInTheDocument();
    });

    it('updates ARIA attributes when dropdown is open', async () => {
      const user = userEvent.setup();
      render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
      
      const input = screen.getByRole('combobox');
      
      await user.type(input, 'A');
      
      // Wait for dropdown to appear
      await screen.findByText('AAPL');
      
      expect(input).toHaveAttribute('aria-expanded', 'true');
      
      const listbox = screen.getByRole('listbox');
      expect(listbox).toBeInTheDocument();
      expect(listbox).toHaveAttribute('aria-label', 'Ticker search results');
    });

    it('supports keyboard navigation with arrow keys', async () => {
      const user = userEvent.setup();
      render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
      
      const input = screen.getByRole('combobox');
      
      await user.type(input, 'A');
      await screen.findByText('AAPL');
      
      // Navigate down with arrow key
      fireEvent.keyDown(input, { key: 'ArrowDown', code: 'ArrowDown' });
      
      const firstOption = screen.getByRole('option', { name: /AAPL - Apple Inc./ });
      expect(firstOption).toHaveAttribute('aria-selected', 'true');
      
      // Navigate down again
      fireEvent.keyDown(input, { key: 'ArrowDown', code: 'ArrowDown' });
      
      const secondOption = screen.getByRole('option', { name: /GOOGL - Alphabet Inc./ });
      expect(secondOption).toHaveAttribute('aria-selected', 'true');
      expect(firstOption).toHaveAttribute('aria-selected', 'false');
    });

    it('supports keyboard navigation with up arrow', async () => {
      const user = userEvent.setup();
      render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
      
      const input = screen.getByRole('combobox');
      
      await user.type(input, 'A');
      await screen.findByText('AAPL');
      
      // Navigate to last item first
      fireEvent.keyDown(input, { key: 'ArrowDown', code: 'ArrowDown' });
      fireEvent.keyDown(input, { key: 'ArrowDown', code: 'ArrowDown' });
      fireEvent.keyDown(input, { key: 'ArrowDown', code: 'ArrowDown' });
      
      const lastOption = screen.getByRole('option', { name: /MSFT - Microsoft Corporation/ });
      expect(lastOption).toHaveAttribute('aria-selected', 'true');
      
      // Navigate up
      fireEvent.keyDown(input, { key: 'ArrowUp', code: 'ArrowUp' });
      
      const secondOption = screen.getByRole('option', { name: /GOOGL - Alphabet Inc./ });
      expect(secondOption).toHaveAttribute('aria-selected', 'true');
      expect(lastOption).toHaveAttribute('aria-selected', 'false');
    });

    it('selects option with Enter key', async () => {
      const user = userEvent.setup();
      render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
      
      const input = screen.getByRole('combobox');
      
      await user.type(input, 'A');
      await screen.findByText('AAPL');
      
      // Navigate to first option and select with Enter
      fireEvent.keyDown(input, { key: 'ArrowDown', code: 'ArrowDown' });
      fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
      
      expect(input).toHaveValue('AAPL');
      expect(input).toHaveAttribute('aria-expanded', 'false');
    });

    it('closes dropdown with Escape key', async () => {
      const user = userEvent.setup();
      render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
      
      const input = screen.getByRole('combobox');
      
      await user.type(input, 'A');
      await screen.findByText('AAPL');
      
      expect(input).toHaveAttribute('aria-expanded', 'true');
      
      fireEvent.keyDown(input, { key: 'Escape', code: 'Escape' });
      
      expect(input).toHaveAttribute('aria-expanded', 'false');
      expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
    });

    it('has proper focus management', async () => {
      const user = userEvent.setup();
      render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
      
      const input = screen.getByRole('combobox');
      const button = screen.getByRole('button', { name: 'Analyze Trade' });
      
      // Tab to input
      await user.tab();
      expect(input).toHaveFocus();
      
      // Tab to button
      await user.tab();
      expect(button).toHaveFocus();
    });

    it('announces loading state to screen readers', () => {
      render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={true} />);
      
      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-label', 'Analyzing trade, please wait');
      expect(button).toBeDisabled();
    });

    it('has proper error announcements', async () => {
      mockApi.searchTickers.mockRejectedValue(new Error('Search failed'));
      
      const user = userEvent.setup();
      render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
      
      const input = screen.getByRole('combobox');
      await user.type(input, 'A');
      
      // Wait for error to appear
      await screen.findByRole('alert');
      
      const errorAlert = screen.getByRole('alert');
      expect(errorAlert).toBeInTheDocument();
      expect(errorAlert).toHaveTextContent('Search failed');
    });
  });

  describe('AnalysisResult Accessibility', () => {
    const mockOnReset = jest.fn();
    const mockResult: TradeAnalysisResult = {
      ticker: 'AAPL',
      company_name: 'Apple Inc.',
      recommendation: {
        should_trade: true,
        confidence: 0.85,
        strategy: 'Cash-Secured Put',
        contracts: [
          {
            action: 'SELL',
            type: 'PUT',
            strike: 150,
            expiration: '2024-01-19',
            quantity: 1,
            premium_credit: 2.50,
          },
        ],
        risk_metrics: {
          max_profit: 250,
          max_loss: -14750,
          breakeven: 147.50,
          prob_profit: 0.65,
          return_on_capital: 0.017,
        },
        reasoning_steps: [
          {
            step: 'Technical Analysis',
            passed: true,
            reasoning: 'Stock shows strong technical indicators.',
            confidence: 0.8,
          },
        ],
      },
    };

    it('has proper heading structure', () => {
      render(<AnalysisResult result={mockResult} onReset={mockOnReset} />);
      
      // Main heading
      expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent('AAPL - Apple Inc.');
      
      // Section headings
      expect(screen.getByRole('heading', { level: 3, name: /recommended strategy/i })).toBeInTheDocument();
      expect(screen.getByRole('heading', { level: 3, name: /contract details/i })).toBeInTheDocument();
      expect(screen.getByRole('heading', { level: 3, name: /risk metrics/i })).toBeInTheDocument();
      expect(screen.getByRole('heading', { level: 3, name: /analysis breakdown/i })).toBeInTheDocument();
    });

    it('has proper ARIA labels for recommendation badge', () => {
      render(<AnalysisResult result={mockResult} onReset={mockOnReset} />);
      
      const recommendationBadge = screen.getByText('✓ YES').closest('div');
      expect(recommendationBadge).toHaveAttribute('role', 'status');
      expect(recommendationBadge).toHaveAttribute('aria-label', 'Recommendation: Yes, trade with 85% confidence');
    });

    it('has expandable sections with proper ARIA attributes', async () => {
      const user = userEvent.setup();
      render(<AnalysisResult result={mockResult} onReset={mockOnReset} />);
      
      const expandButton = screen.getByRole('button', { name: /technical analysis/i });
      expect(expandButton).toHaveAttribute('aria-expanded', 'false');
      expect(expandButton).toHaveAttribute('aria-controls');
      
      await user.click(expandButton);
      
      expect(expandButton).toHaveAttribute('aria-expanded', 'true');
    });

    it('has proper table structure for contract details', () => {
      render(<AnalysisResult result={mockResult} onReset={mockOnReset} />);
      
      const table = screen.getByRole('table');
      expect(table).toBeInTheDocument();
      
      // Check for proper table headers
      expect(screen.getByRole('columnheader', { name: 'Action' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: 'Type' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: 'Strike' })).toBeInTheDocument();
    });

    it('has proper button accessibility', () => {
      render(<AnalysisResult result={mockResult} onReset={mockOnReset} />);
      
      const resetButton = screen.getByRole('button', { name: 'Analyze Another Stock' });
      expect(resetButton).toBeInTheDocument();
      expect(resetButton).not.toHaveAttribute('aria-disabled');
    });

    it('announces status changes to screen readers', () => {
      const negativeResult = {
        ...mockResult,
        recommendation: {
          ...mockResult.recommendation,
          should_trade: false,
          confidence: 0.75,
        },
      };

      render(<AnalysisResult result={negativeResult} onReset={mockOnReset} />);
      
      const recommendationBadge = screen.getByText('✗ NO').closest('div');
      expect(recommendationBadge).toHaveAttribute('role', 'status');
      expect(recommendationBadge).toHaveAttribute('aria-label', 'Recommendation: No, do not trade with 75% confidence');
    });
  });

  describe('ErrorMessage Accessibility', () => {
    const mockOnRetry = jest.fn();
    const mockOnDismiss = jest.fn();

    it('has proper alert role', () => {
      render(<ErrorMessage error="Test error" />);
      
      const alert = screen.getByRole('alert');
      expect(alert).toBeInTheDocument();
      expect(alert).toHaveTextContent('Test error');
    });

    it('has proper button labels', () => {
      render(<ErrorMessage error="Test error" onRetry={mockOnRetry} onDismiss={mockOnDismiss} />);
      
      const retryButton = screen.getByRole('button', { name: /try again/i });
      expect(retryButton).toBeInTheDocument();
      
      const dismissButtons = screen.getAllByRole('button', { name: /dismiss/i });
      expect(dismissButtons.length).toBeGreaterThan(0);
    });

    it('has proper focus management for dismiss button', async () => {
      const user = userEvent.setup();
      render(<ErrorMessage error="Test error" onDismiss={mockOnDismiss} />);
      
      const dismissButtons = screen.getAllByRole('button', { name: /dismiss/i });
      const xButton = dismissButtons.find(button => button.textContent === '×');
      
      if (xButton) {
        await user.click(xButton);
        expect(mockOnDismiss).toHaveBeenCalled();
      }
    });

    it('supports keyboard navigation', async () => {
      const user = userEvent.setup();
      render(<ErrorMessage error="Test error" onRetry={mockOnRetry} onDismiss={mockOnDismiss} />);
      
      // Tab through buttons
      await user.tab();
      const retryButton = screen.getByRole('button', { name: /try again/i });
      expect(retryButton).toHaveFocus();
      
      await user.tab();
      const dismissButtons = screen.getAllByRole('button', { name: /dismiss/i });
      const focusedButton = dismissButtons.find(button => button === document.activeElement);
      expect(focusedButton).toBeTruthy();
    });

    it('has proper ARIA live region for dynamic errors', () => {
      const { rerender } = render(<ErrorMessage error="First error" />);
      
      const alert = screen.getByRole('alert');
      expect(alert).toHaveAttribute('aria-live', 'polite');
      
      rerender(<ErrorMessage error="Second error" />);
      expect(alert).toHaveTextContent('Second error');
    });
  });

  describe('Color Contrast and Visual Accessibility', () => {
    it('uses sufficient color contrast for success states', () => {
      const mockResult: TradeAnalysisResult = {
        ticker: 'AAPL',
        company_name: 'Apple Inc.',
        recommendation: {
          should_trade: true,
          confidence: 0.85,
          strategy: 'Cash-Secured Put',
          contracts: [],
          risk_metrics: undefined,
          reasoning_steps: [],
        },
      };

      render(<AnalysisResult result={mockResult} onReset={jest.fn()} />);
      
      const successBadge = screen.getByText('✓ YES').closest('div');
      expect(successBadge).toHaveClass('bg-green-100', 'text-green-800');
    });

    it('uses sufficient color contrast for error states', () => {
      const mockResult: TradeAnalysisResult = {
        ticker: 'AAPL',
        company_name: 'Apple Inc.',
        recommendation: {
          should_trade: false,
          confidence: 0.75,
          strategy: undefined,
          contracts: [],
          risk_metrics: undefined,
          reasoning_steps: [],
        },
      };

      render(<AnalysisResult result={mockResult} onReset={jest.fn()} />);
      
      const errorBadge = screen.getByText('✗ NO').closest('div');
      expect(errorBadge).toHaveClass('bg-red-100', 'text-red-800');
    });

    it('provides text alternatives for visual indicators', () => {
      render(<ErrorMessage error="Test error" />);
      
      // Error icon should have text alternative
      const errorIcon = screen.getByText('Error');
      expect(errorIcon).toBeInTheDocument();
    });
  });

  describe('Screen Reader Support', () => {
    it('provides descriptive text for complex UI elements', () => {
      const mockResult: TradeAnalysisResult = {
        ticker: 'AAPL',
        company_name: 'Apple Inc.',
        recommendation: {
          should_trade: true,
          confidence: 0.85,
          strategy: 'Cash-Secured Put',
          contracts: [
            {
              action: 'SELL',
              type: 'PUT',
              strike: 150,
              expiration: '2024-01-19',
              quantity: 1,
              premium_credit: 2.50,
            },
          ],
          risk_metrics: {
            max_profit: 250,
            max_loss: -14750,
            breakeven: 147.50,
            prob_profit: 0.65,
            return_on_capital: 0.017,
          },
          reasoning_steps: [],
        },
      };

      render(<AnalysisResult result={mockResult} onReset={jest.fn()} />);
      
      // Check for screen reader friendly descriptions
      expect(screen.getByText('Maximum profit: $250.00')).toBeInTheDocument();
      expect(screen.getByText('Maximum loss: $14750.00')).toBeInTheDocument();
      expect(screen.getByText('Breakeven price: $147.50')).toBeInTheDocument();
    });

    it('provides context for form controls', () => {
      render(<TickerSearch onAnalyze={jest.fn()} isLoading={false} />);
      
      const input = screen.getByRole('combobox');
      expect(input).toHaveAttribute('placeholder', 'Enter stock ticker (e.g., AAPL)');
      expect(input).toHaveAttribute('aria-label', 'Search for stock ticker');
    });
  });
});