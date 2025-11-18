import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AnalysisResult from '../AnalysisResult';
import { TradeAnalysisResult } from '@/types';

describe('AnalysisResult', () => {
  const mockOnReset = jest.fn();

  const mockPositiveResult: TradeAnalysisResult = {
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
          reasoning: 'Stock shows strong technical indicators with RSI at 45 and good volume.',
          confidence: 0.8,
        },
        {
          step: 'Fundamental Screening',
          passed: true,
          reasoning: 'Company has strong fundamentals with P/E ratio of 25 and low debt.',
          confidence: 0.9,
        },
      ],
    },
  };

  const mockNegativeResult: TradeAnalysisResult = {
    ticker: 'XYZ',
    company_name: 'XYZ Corporation',
    recommendation: {
      should_trade: false,
      confidence: 0.75,
      strategy: undefined,
      contracts: [],
      risk_metrics: undefined,
      reasoning_steps: [
        {
          step: 'Technical Analysis',
          passed: false,
          reasoning: 'Stock shows weak technical indicators with low volume.',
          confidence: 0.7,
        },
        {
          step: 'Options Analysis',
          passed: false,
          reasoning: 'Options chain has poor liquidity with wide bid-ask spreads.',
          confidence: 0.8,
        },
      ],
    },
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders ticker and company name', () => {
    render(<AnalysisResult result={mockPositiveResult} onReset={mockOnReset} />);
    
    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
  });

  it('displays positive recommendation with confidence', () => {
    render(<AnalysisResult result={mockPositiveResult} onReset={mockOnReset} />);
    
    expect(screen.getByText('✓ YES')).toBeInTheDocument();
    expect(screen.getByText('(85% confidence)')).toBeInTheDocument();
  });

  it('displays negative recommendation with confidence', () => {
    render(<AnalysisResult result={mockNegativeResult} onReset={mockOnReset} />);
    
    expect(screen.getByText('✗ NO')).toBeInTheDocument();
    expect(screen.getByText('(75% confidence)')).toBeInTheDocument();
  });

  it('shows strategy card for positive recommendations', () => {
    render(<AnalysisResult result={mockPositiveResult} onReset={mockOnReset} />);
    
    expect(screen.getByText('Recommended Strategy: Cash-Secured Put')).toBeInTheDocument();
  });

  it('does not show strategy card for negative recommendations', () => {
    render(<AnalysisResult result={mockNegativeResult} onReset={mockOnReset} />);
    
    expect(screen.queryByText(/Recommended Strategy:/)).not.toBeInTheDocument();
  });

  it('displays contract details correctly', () => {
    render(<AnalysisResult result={mockPositiveResult} onReset={mockOnReset} />);
    
    expect(screen.getByText('Contract Details')).toBeInTheDocument();
    expect(screen.getByText('SELL')).toBeInTheDocument();
    expect(screen.getByText('PUT')).toBeInTheDocument();
    expect(screen.getByText('$150')).toBeInTheDocument();
    expect(screen.getByText('2024-01-19')).toBeInTheDocument();
    expect(screen.getByText('1')).toBeInTheDocument();
    expect(screen.getByText('$2.5')).toBeInTheDocument();
  });

  it('displays risk metrics correctly', () => {
    render(<AnalysisResult result={mockPositiveResult} onReset={mockOnReset} />);
    
    expect(screen.getByText('Risk Metrics')).toBeInTheDocument();
    expect(screen.getByText('$250.00')).toBeInTheDocument(); // Max Profit
    expect(screen.getByText('$14750.00')).toBeInTheDocument(); // Max Loss (absolute value)
    expect(screen.getByText('$147.50')).toBeInTheDocument(); // Breakeven
    expect(screen.getByText('65.0%')).toBeInTheDocument(); // Prob of Profit
    expect(screen.getByText('1.7%')).toBeInTheDocument(); // Return on Capital
  });

  it('displays reasoning steps with correct status indicators', () => {
    render(<AnalysisResult result={mockPositiveResult} onReset={mockOnReset} />);
    
    expect(screen.getByText('Analysis Breakdown')).toBeInTheDocument();
    expect(screen.getByText('Technical Analysis')).toBeInTheDocument();
    expect(screen.getByText('Fundamental Screening')).toBeInTheDocument();
    
    // Check for passed indicators (✓) - only in reasoning steps
    const passedIndicators = screen.getAllByText('✓');
    expect(passedIndicators).toHaveLength(2); // 2 in reasoning steps
  });

  it('shows failed indicators for negative results', () => {
    render(<AnalysisResult result={mockNegativeResult} onReset={mockOnReset} />);
    
    // Check for failed indicators (✗) - only in reasoning steps
    const failedIndicators = screen.getAllByText('✗');
    expect(failedIndicators).toHaveLength(2); // 2 in reasoning steps
  });

  it('expands and collapses reasoning steps on click', async () => {
    const user = userEvent.setup();
    render(<AnalysisResult result={mockPositiveResult} onReset={mockOnReset} />);
    
    // Initially, detailed reasoning should not be visible
    expect(screen.queryByText('Stock shows strong technical indicators')).not.toBeInTheDocument();
    
    // Click on the first reasoning step
    await user.click(screen.getByText('Technical Analysis'));
    
    // Now the detailed reasoning should be visible
    expect(screen.getByText('Stock shows strong technical indicators with RSI at 45 and good volume.')).toBeInTheDocument();
    
    // Click again to collapse
    await user.click(screen.getByText('Technical Analysis'));
    
    // Detailed reasoning should be hidden again
    expect(screen.queryByText('Stock shows strong technical indicators')).not.toBeInTheDocument();
  });

  it('calls onReset when analyze another stock button is clicked', async () => {
    const user = userEvent.setup();
    render(<AnalysisResult result={mockPositiveResult} onReset={mockOnReset} />);
    
    await user.click(screen.getByText('Analyze Another Stock'));
    
    expect(mockOnReset).toHaveBeenCalledTimes(1);
  });

  it('handles multiple breakeven points correctly', () => {
    const resultWithMultipleBreakevens: TradeAnalysisResult = {
      ...mockPositiveResult,
      recommendation: {
        ...mockPositiveResult.recommendation,
        risk_metrics: {
          ...mockPositiveResult.recommendation.risk_metrics!,
          breakeven: [145.50, 152.50],
        },
      },
    };

    render(<AnalysisResult result={resultWithMultipleBreakevens} onReset={mockOnReset} />);
    
    expect(screen.getByText('$145.50, $152.50')).toBeInTheDocument();
  });

  it('displays confidence percentages for reasoning steps', () => {
    render(<AnalysisResult result={mockPositiveResult} onReset={mockOnReset} />);
    
    expect(screen.getByText('(80% confidence)')).toBeInTheDocument();
    expect(screen.getByText('(90% confidence)')).toBeInTheDocument();
  });

  it('applies correct styling for positive and negative recommendations', () => {
    const { rerender } = render(<AnalysisResult result={mockPositiveResult} onReset={mockOnReset} />);
    
    // Check positive styling
    const positiveElement = screen.getByText('✓ YES').closest('div');
    expect(positiveElement).toHaveClass('bg-green-100', 'text-green-800');
    
    // Check negative styling
    rerender(<AnalysisResult result={mockNegativeResult} onReset={mockOnReset} />);
    const negativeElement = screen.getByText('✗ NO').closest('div');
    expect(negativeElement).toHaveClass('bg-red-100', 'text-red-800');
  });
});