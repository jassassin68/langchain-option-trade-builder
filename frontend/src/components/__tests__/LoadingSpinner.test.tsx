import React from 'react';
import { render, screen } from '@testing-library/react';
import LoadingSpinner, { AnalysisLoadingSpinner } from '../LoadingSpinner';

describe('LoadingSpinner', () => {
  it('renders with default props', () => {
    render(<LoadingSpinner />);
    
    expect(screen.getByRole('status', { name: 'Loading' })).toBeInTheDocument();
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('renders with custom text', () => {
    render(<LoadingSpinner text="Processing..." />);
    
    expect(screen.getByText('Processing...')).toBeInTheDocument();
  });

  it('renders without text when text prop is empty', () => {
    render(<LoadingSpinner text="" />);
    
    expect(screen.getByRole('status', { name: 'Loading' })).toBeInTheDocument();
    expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
  });

  it('applies correct size classes', () => {
    const { rerender } = render(<LoadingSpinner size="sm" />);
    expect(document.querySelector('.h-4')).toBeInTheDocument();

    rerender(<LoadingSpinner size="md" />);
    expect(document.querySelector('.h-6')).toBeInTheDocument();

    rerender(<LoadingSpinner size="lg" />);
    expect(document.querySelector('.h-8')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    render(<LoadingSpinner className="custom-class" />);
    
    expect(document.querySelector('.custom-class')).toBeInTheDocument();
  });
});

describe('AnalysisLoadingSpinner', () => {
  it('renders analysis loading spinner with correct text', () => {
    render(<AnalysisLoadingSpinner />);
    
    expect(screen.getByText('Analyzing...')).toBeInTheDocument();
    expect(screen.getByText('Evaluating technical indicators, fundamentals, and options data')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    render(<AnalysisLoadingSpinner className="analysis-loading" />);
    
    expect(document.querySelector('.analysis-loading')).toBeInTheDocument();
  });

  it('has proper loading animation elements', () => {
    render(<AnalysisLoadingSpinner />);
    
    // Check for spinning elements
    expect(document.querySelector('.animate-spin')).toBeInTheDocument();
    expect(document.querySelector('.animate-pulse')).toBeInTheDocument();
  });
});