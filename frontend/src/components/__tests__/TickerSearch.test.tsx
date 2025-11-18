import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import TickerSearch from '../TickerSearch';
import { api } from '@/lib/api';

// Mock the API module
jest.mock('@/lib/api', () => ({
  api: {
    searchTickers: jest.fn(),
  },
}));

const mockApi = api as jest.Mocked<typeof api>;

describe('TickerSearch', () => {
  const mockOnAnalyze = jest.fn();
  const mockTickerResults = [
    { ticker: 'AAPL', company_name: 'Apple Inc.', exchange: 'NASDAQ' },
    { ticker: 'GOOGL', company_name: 'Alphabet Inc.', exchange: 'NASDAQ' },
    { ticker: 'MSFT', company_name: 'Microsoft Corporation', exchange: 'NASDAQ' },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    mockApi.searchTickers.mockResolvedValue({
      results: mockTickerResults,
      count: mockTickerResults.length,
    });
  });

  it('renders search input and analyze button', () => {
    render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
    
    expect(screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analyze Trade' })).toBeInTheDocument();
  });

  it('disables analyze button when no ticker is selected', () => {
    render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
    
    const analyzeButton = screen.getByRole('button', { name: 'Analyze Trade' });
    expect(analyzeButton).toBeDisabled();
  });

  it('shows loading state when isLoading is true', () => {
    render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={true} />);
    
    expect(screen.getByText('Analyzing...')).toBeInTheDocument();
    expect(screen.getByRole('button')).toBeDisabled();
  });

  it('searches for tickers with debounced input', async () => {
    const user = userEvent.setup();
    render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
    
    const input = screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)');
    await user.type(input, 'AAP');

    // Wait for debounce delay
    await waitFor(() => {
      expect(mockApi.searchTickers).toHaveBeenCalledWith('AAP');
    }, { timeout: 500 });
  });

  it('displays search results in dropdown', async () => {
    const user = userEvent.setup();
    render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
    
    const input = screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)');
    await user.type(input, 'A');

    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument();
      expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
    });
  });

  it('selects ticker from dropdown on click', async () => {
    const user = userEvent.setup();
    render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
    
    const input = screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)');
    await user.type(input, 'A');

    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument();
    });

    await user.click(screen.getByText('AAPL'));
    
    expect(input).toHaveValue('AAPL');
    expect(screen.getByRole('button', { name: 'Analyze Trade' })).not.toBeDisabled();
  });

  it('handles keyboard navigation in dropdown', async () => {
    const user = userEvent.setup();
    render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
    
    const input = screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)');
    await user.type(input, 'A');

    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument();
    });

    // Navigate down with arrow key
    await user.keyboard('{ArrowDown}');
    await user.keyboard('{ArrowDown}');
    await user.keyboard('{Enter}');

    expect(input).toHaveValue('GOOGL');
  });

  it('closes dropdown on escape key', async () => {
    const user = userEvent.setup();
    render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
    
    const input = screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)');
    await user.type(input, 'A');

    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument();
    });

    await user.keyboard('{Escape}');
    
    expect(screen.queryByText('AAPL')).not.toBeInTheDocument();
  });

  it('calls onAnalyze when analyze button is clicked', async () => {
    const user = userEvent.setup();
    render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
    
    const input = screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)');
    await user.type(input, 'A');

    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument();
    });

    await user.click(screen.getByText('AAPL'));
    await user.click(screen.getByRole('button', { name: 'Analyze Trade' }));

    expect(mockOnAnalyze).toHaveBeenCalledWith('AAPL');
  });

  it('calls onAnalyze when enter is pressed with selected ticker', async () => {
    const user = userEvent.setup();
    render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
    
    const input = screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)');
    await user.type(input, 'A');

    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument();
    });

    await user.click(screen.getByText('AAPL'));
    await user.keyboard('{Enter}');

    expect(mockOnAnalyze).toHaveBeenCalledWith('AAPL');
  });

  it('shows error message when search fails', async () => {
    mockApi.searchTickers.mockRejectedValue(new Error('API Error'));
    
    const user = userEvent.setup();
    render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
    
    const input = screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)');
    await user.type(input, 'A');

    await waitFor(() => {
      expect(screen.getByText('Failed to search tickers. Please try again.')).toBeInTheDocument();
    });
  });

  it('converts input to uppercase', async () => {
    const user = userEvent.setup();
    render(<TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />);
    
    const input = screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)');
    await user.type(input, 'aapl');

    expect(input).toHaveValue('AAPL');
  });

  it('closes dropdown when clicking outside', async () => {
    const user = userEvent.setup();
    render(
      <div>
        <TickerSearch onAnalyze={mockOnAnalyze} isLoading={false} />
        <div data-testid="outside">Outside element</div>
      </div>
    );
    
    const input = screen.getByPlaceholderText('Enter stock ticker (e.g., AAPL)');
    await user.type(input, 'A');

    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument();
    });

    await user.click(screen.getByTestId('outside'));
    
    expect(screen.queryByText('AAPL')).not.toBeInTheDocument();
  });
});