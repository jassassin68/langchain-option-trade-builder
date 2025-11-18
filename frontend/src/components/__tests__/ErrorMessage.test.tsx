import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ErrorMessage, { AnalysisErrorMessage } from '../ErrorMessage';
import { ApiError } from '@/lib/api';

describe('ErrorMessage', () => {
  const mockOnRetry = jest.fn();
  const mockOnDismiss = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders with string error message', () => {
    render(<ErrorMessage error="Test error message" />);
    
    expect(screen.getByText('Error')).toBeInTheDocument();
    expect(screen.getByText('Test error message')).toBeInTheDocument();
  });

  it('renders with Error object', () => {
    const error = new Error('Test error object');
    render(<ErrorMessage error={error} />);
    
    expect(screen.getByText('Error')).toBeInTheDocument();
    expect(screen.getByText('Test error object')).toBeInTheDocument();
  });

  it('renders network error with proper formatting', () => {
    const error = new ApiError('Network failed', 0, 'NETWORK_ERROR');
    render(<ErrorMessage error={error} />);
    
    expect(screen.getByText('Connection Error')).toBeInTheDocument();
    expect(screen.getByText(/Unable to connect to the server/)).toBeInTheDocument();
  });

  it('renders rate limit error with retry time', () => {
    const error = new ApiError('Rate limited', 429, 'RATE_LIMIT', 60);
    render(<ErrorMessage error={error} />);
    
    expect(screen.getByText('Rate Limited')).toBeInTheDocument();
    expect(screen.getByText(/wait 60 seconds/)).toBeInTheDocument();
  });

  it('renders ticker not found error', () => {
    const error = new ApiError('Ticker not found', 404, 'TICKER_NOT_FOUND');
    render(<ErrorMessage error={error} />);
    
    expect(screen.getByText('Ticker Not Found')).toBeInTheDocument();
    expect(screen.getByText(/ticker symbol was not found/)).toBeInTheDocument();
  });

  it('renders insufficient data error', () => {
    const error = new ApiError('Insufficient data', 422, 'INSUFFICIENT_DATA');
    render(<ErrorMessage error={error} />);
    
    expect(screen.getByText('Insufficient Data')).toBeInTheDocument();
    expect(screen.getByText(/Not enough data available/)).toBeInTheDocument();
  });

  it('shows retry button for retryable errors', () => {
    const error = new ApiError('Server error', 500);
    render(<ErrorMessage error={error} onRetry={mockOnRetry} />);
    
    const retryButton = screen.getByRole('button', { name: /try again/i });
    expect(retryButton).toBeInTheDocument();
  });

  it('does not show retry button for non-retryable errors', () => {
    const error = new ApiError('Ticker not found', 404, 'TICKER_NOT_FOUND');
    render(<ErrorMessage error={error} onRetry={mockOnRetry} />);
    
    expect(screen.queryByRole('button', { name: /try again/i })).not.toBeInTheDocument();
  });

  it('calls onRetry when retry button is clicked', async () => {
    const user = userEvent.setup();
    const error = new ApiError('Server error', 500);
    render(<ErrorMessage error={error} onRetry={mockOnRetry} />);
    
    const retryButton = screen.getByRole('button', { name: /try again/i });
    await user.click(retryButton);
    
    expect(mockOnRetry).toHaveBeenCalledTimes(1);
  });

  it('shows dismiss button when onDismiss is provided', () => {
    render(<ErrorMessage error="Test error" onDismiss={mockOnDismiss} />);
    
    const dismissButtons = screen.getAllByText('Dismiss');
    expect(dismissButtons.length).toBeGreaterThan(0);
  });

  it('calls onDismiss when dismiss button is clicked', async () => {
    const user = userEvent.setup();
    render(<ErrorMessage error="Test error" onDismiss={mockOnDismiss} />);
    
    // Get the visible dismiss button (not the sr-only one)
    const dismissButtons = screen.getAllByText('Dismiss');
    const visibleDismissButton = dismissButtons.find(button => 
      !button.classList.contains('sr-only') && button.tagName === 'BUTTON'
    );
    
    if (visibleDismissButton) {
      await user.click(visibleDismissButton);
    }
    
    expect(mockOnDismiss).toHaveBeenCalledTimes(1);
  });

  it('calls onDismiss when X button is clicked', async () => {
    const user = userEvent.setup();
    render(<ErrorMessage error="Test error" onDismiss={mockOnDismiss} />);
    
    // Get all dismiss buttons and click the X button (second one)
    const dismissButtons = screen.getAllByRole('button', { name: /dismiss/i });
    await user.click(dismissButtons[1]); // X button is the second one
    
    expect(mockOnDismiss).toHaveBeenCalledTimes(1);
  });

  it('renders warning variant with correct styling', () => {
    render(<ErrorMessage error="Warning message" variant="warning" />);
    
    expect(document.querySelector('.bg-yellow-50')).toBeInTheDocument();
  });

  it('renders info variant with correct styling', () => {
    render(<ErrorMessage error="Info message" variant="info" />);
    
    expect(document.querySelector('.bg-blue-50')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    render(<ErrorMessage error="Test error" className="custom-error" />);
    
    expect(document.querySelector('.custom-error')).toBeInTheDocument();
  });
});

describe('AnalysisErrorMessage', () => {
  const mockOnRetry = jest.fn();
  const mockOnReset = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders analysis error message', () => {
    render(<AnalysisErrorMessage error="Analysis failed" />);
    
    expect(screen.getByText('Analysis Failed')).toBeInTheDocument();
  });

  it('shows retry button when onRetry is provided', () => {
    render(<AnalysisErrorMessage error="Analysis failed" onRetry={mockOnRetry} />);
    
    expect(screen.getByRole('button', { name: /retry analysis/i })).toBeInTheDocument();
  });

  it('shows reset button when onReset is provided', () => {
    render(<AnalysisErrorMessage error="Analysis failed" onReset={mockOnReset} />);
    
    expect(screen.getByRole('button', { name: /try different ticker/i })).toBeInTheDocument();
  });

  it('calls onRetry when retry button is clicked', async () => {
    const user = userEvent.setup();
    render(<AnalysisErrorMessage error="Analysis failed" onRetry={mockOnRetry} />);
    
    const retryButton = screen.getByRole('button', { name: /retry analysis/i });
    await user.click(retryButton);
    
    expect(mockOnRetry).toHaveBeenCalledTimes(1);
  });

  it('calls onReset when reset button is clicked', async () => {
    const user = userEvent.setup();
    render(<AnalysisErrorMessage error="Analysis failed" onReset={mockOnReset} />);
    
    const resetButton = screen.getByRole('button', { name: /try different ticker/i });
    await user.click(resetButton);
    
    expect(mockOnReset).toHaveBeenCalledTimes(1);
  });

  it('applies custom className', () => {
    render(<AnalysisErrorMessage error="Analysis failed" className="analysis-error" />);
    
    expect(document.querySelector('.analysis-error')).toBeInTheDocument();
  });
});