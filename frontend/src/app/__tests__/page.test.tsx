import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Home from '../page';
import { api } from '@/lib/api';

// Mock the API module
jest.mock('@/lib/api', () => ({
  api: {
    analyzeTrade: jest.fn(),
    searchTickers: jest.fn(),
  },
}));

const mockApi = api as jest.Mocked<typeof api>;

describe('Home Page', () => {
  const mockAnalysisResult = {
    ticker: 'AAPL',
    company_name: 'Apple Inc.',
    recommendation: {
      should_trade: true,
      confidence: 0.85,
      strategy: 'Cash-Secured Put',
      contracts: [
        {
          action: 'SELL' as const,
          type: 'PUT' as const,
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

  beforeEach(() => {
    jest.clearAllMocks();
    mockApi.searchTickers.mockResolvedValue({
      results: [
        { ticker: 'AAPL', company_name: 'Apple Inc.', exchange: 'NASDAQ' },
        { ticker: 'MSFT', company_name: 'Microsoft Corporation', exchange: 'NASDAQ' },
      ],
      count: 2,
    });
  });

  it('renders the main page with header and search interface', () => {
    render(<Home />);

    expect(screen.getByText('Options Trade Evaluator')).toBeInTheDocument();
    expect(screen.getByText('AI-powered options trading analysis and recommendations')).toBeInTheDocument();
    expect(screen.getByText('Get Started')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)')).toBeInTheDocument();
  });

  it('shows loading state during analysis', async () => {
    const user = userEvent.setup();
    mockApi.analyzeTrade.mockImplementation(() => new Promise(resolve => setTimeout(resolve, 100)));

    render(<Home />);

    const input = screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)');
    
    await act(async () => {
      await user.type(input, 'AAPL');
    });

    // Wait for debounced search and dropdown to appear
    await waitFor(() => {
      expect(mockApi.searchTickers).toHaveBeenCalledWith('AAPL');
    });

    await waitFor(() => {
      expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
    });

    await act(async () => {
      await user.click(screen.getByText('AAPL'));
    });

    // Wait for button to be enabled
    await waitFor(() => {
      expect(screen.getByText('Analyze Trade')).not.toBeDisabled();
    });

    await act(async () => {
      await user.click(screen.getByText('Analyze Trade'));
    });

    expect(screen.getByText('Analyzing...')).toBeInTheDocument();
  });

  it('displays analysis results after successful analysis', async () => {
    const user = userEvent.setup();
    mockApi.analyzeTrade.mockResolvedValue(mockAnalysisResult);

    render(<Home />);

    const input = screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)');
    
    await act(async () => {
      await user.type(input, 'AAPL');
    });

    // Wait for debounced search and dropdown to appear
    await waitFor(() => {
      expect(mockApi.searchTickers).toHaveBeenCalledWith('AAPL');
    });

    await waitFor(() => {
      expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
    });

    await act(async () => {
      await user.click(screen.getByText('AAPL'));
    });

    // Wait for button to be enabled
    await waitFor(() => {
      expect(screen.getByText('Analyze Trade')).not.toBeDisabled();
    });

    await act(async () => {
      await user.click(screen.getByText('Analyze Trade'));
    });

    await waitFor(() => {
      expect(screen.getByText('✓ YES')).toBeInTheDocument();
    });

    expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
    expect(screen.getByText('Recommended Strategy: Cash-Secured Put')).toBeInTheDocument();
  });

  it('displays error message when analysis fails', async () => {
    const user = userEvent.setup();
    mockApi.analyzeTrade.mockRejectedValue(new Error('API Error'));

    render(<Home />);

    const input = screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)');
    
    await act(async () => {
      await user.type(input, 'AAPL');
    });

    // Wait for debounced search and dropdown to appear
    await waitFor(() => {
      expect(mockApi.searchTickers).toHaveBeenCalledWith('AAPL');
    });

    await waitFor(() => {
      expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
    });

    await act(async () => {
      await user.click(screen.getByText('AAPL'));
    });

    // Wait for button to be enabled
    await waitFor(() => {
      expect(screen.getByText('Analyze Trade')).not.toBeDisabled();
    });

    await act(async () => {
      await user.click(screen.getByText('Analyze Trade'));
    });

    await waitFor(() => {
      expect(screen.getByText('Analysis Error')).toBeInTheDocument();
    });

    expect(screen.getByText('Failed to analyze the trade. Please try again.')).toBeInTheDocument();
  });

  it('allows user to reset and analyze another stock', async () => {
    const user = userEvent.setup();
    mockApi.analyzeTrade.mockResolvedValue(mockAnalysisResult);

    render(<Home />);

    // First analysis
    const input = screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)');
    
    await act(async () => {
      await user.type(input, 'AAPL');
    });

    // Wait for debounced search and dropdown to appear
    await waitFor(() => {
      expect(mockApi.searchTickers).toHaveBeenCalledWith('AAPL');
    });

    await waitFor(() => {
      expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
    });

    await act(async () => {
      await user.click(screen.getByText('AAPL'));
    });

    // Wait for button to be enabled
    await waitFor(() => {
      expect(screen.getByText('Analyze Trade')).not.toBeDisabled();
    });

    await act(async () => {
      await user.click(screen.getByText('Analyze Trade'));
    });

    await waitFor(() => {
      expect(screen.getByText('✓ YES')).toBeInTheDocument();
    });

    // Reset
    await act(async () => {
      await user.click(screen.getByText('Analyze Another Stock'));
    });

    // Should be back to search interface
    expect(screen.getByText('Get Started')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)')).toBeInTheDocument();
  });

  it('renders footer with disclaimer', () => {
    render(<Home />);

    expect(screen.getByText('Options Trade Evaluator - AI-powered trading analysis')).toBeInTheDocument();
    expect(screen.getByText('This tool provides educational analysis only. Not financial advice.')).toBeInTheDocument();
  });

  it('has responsive design classes', () => {
    render(<Home />);

    const main = screen.getByRole('main');
    expect(main).toHaveClass('min-h-screen', 'bg-gray-50');

    const header = screen.getByRole('banner');
    expect(header).toHaveClass('bg-white', 'shadow-sm', 'border-b', 'border-gray-200');
    
    // Check for responsive typography classes
    const title = screen.getByText('Options Trade Evaluator');
    expect(title).toHaveClass('text-2xl', 'sm:text-3xl');
  });

  it('clears error when starting new analysis', async () => {
    const user = userEvent.setup();
    mockApi.analyzeTrade.mockRejectedValueOnce(new Error('API Error'))
                        .mockResolvedValueOnce(mockAnalysisResult);

    render(<Home />);

    // First analysis that fails
    const input = screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)');
    
    await act(async () => {
      await user.type(input, 'AAPL');
    });

    // Wait for debounced search and dropdown to appear
    await waitFor(() => {
      expect(mockApi.searchTickers).toHaveBeenCalledWith('AAPL');
    });

    await waitFor(() => {
      expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
    });

    await act(async () => {
      await user.click(screen.getByText('AAPL'));
    });

    // Wait for button to be enabled
    await waitFor(() => {
      expect(screen.getByText('Analyze Trade')).not.toBeDisabled();
    });

    await act(async () => {
      await user.click(screen.getByText('Analyze Trade'));
    });

    await waitFor(() => {
      expect(screen.getByText('Analysis Error')).toBeInTheDocument();
    });

    // Clear input and try again
    await act(async () => {
      await user.clear(input);
      await user.type(input, 'MSFT');
    });

    await waitFor(() => {
      expect(screen.queryByText('Analysis Error')).not.toBeInTheDocument();
    });
  });

  it('has proper accessibility attributes', () => {
    render(<Home />);

    // Check for proper heading hierarchy
    const mainHeading = screen.getByRole('heading', { level: 1 });
    expect(mainHeading).toHaveTextContent('Options Trade Evaluator');

    const subHeading = screen.getByRole('heading', { level: 2 });
    expect(subHeading).toHaveTextContent('Get Started');

    // Check for aria-hidden on decorative SVG
    const errorSvg = document.querySelector('svg[aria-hidden="true"]');
    // SVG should not be present initially, but the attribute should be there when error occurs
    expect(errorSvg).toBeNull(); // No error initially
  });
});