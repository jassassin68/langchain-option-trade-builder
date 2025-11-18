'use client';

import React, { useState } from 'react';
import { TradeAnalysisResult } from '@/types';

interface AnalysisResultProps {
  result: TradeAnalysisResult;
  onReset: () => void;
}

export default function AnalysisResult({ result, onReset }: AnalysisResultProps) {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());

  const toggleStep = (index: number) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSteps(newExpanded);
  };

  const { recommendation } = result;
  const confidencePercentage = Math.round(recommendation.confidence * 100);

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6">
      {/* Header with Ticker and Company */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900">{result.ticker}</h2>
        <p className="text-lg text-gray-600">{result.company_name}</p>
      </div>

      {/* Recommendation Badge */}
      <div className="flex justify-center">
        <div
          className={`inline-flex items-center px-6 py-3 rounded-full text-xl font-bold ${
            recommendation.should_trade
              ? 'bg-green-100 text-green-800 border-2 border-green-300'
              : 'bg-red-100 text-red-800 border-2 border-red-300'
          }`}
        >
          <span className="mr-2">
            {recommendation.should_trade ? '✓ YES' : '✗ NO'}
          </span>
          <span className="text-sm font-medium">
            ({confidencePercentage}% confidence)
          </span>
        </div>
      </div>

      {/* Strategy Card */}
      {recommendation.should_trade && recommendation.strategy && (
        <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Recommended Strategy: {recommendation.strategy}
          </h3>
          
          {/* Contract Details */}
          {recommendation.contracts.length > 0 && (
            <div className="mb-4">
              <h4 className="text-md font-medium text-gray-700 mb-2">Contract Details</h4>
              <div className="space-y-2">
                {recommendation.contracts.map((contract, index) => (
                  <div
                    key={index}
                    className="bg-gray-50 rounded-md p-3 text-sm"
                  >
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <div>
                        <span className="font-medium text-gray-600">Action:</span>
                        <span className={`ml-1 ${contract.action === 'BUY' ? 'text-green-600' : 'text-red-600'}`}>
                          {contract.action}
                        </span>
                      </div>
                      <div>
                        <span className="font-medium text-gray-600">Type:</span>
                        <span className="ml-1">{contract.type}</span>
                      </div>
                      <div>
                        <span className="font-medium text-gray-600">Strike:</span>
                        <span className="ml-1">${contract.strike}</span>
                      </div>
                      <div>
                        <span className="font-medium text-gray-600">Expiration:</span>
                        <span className="ml-1">{contract.expiration}</span>
                      </div>
                      <div>
                        <span className="font-medium text-gray-600">Quantity:</span>
                        <span className="ml-1">{contract.quantity}</span>
                      </div>
                      {contract.premium_credit && (
                        <div>
                          <span className="font-medium text-gray-600">Premium:</span>
                          <span className="ml-1">${contract.premium_credit}</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Risk Metrics */}
          {recommendation.risk_metrics && (
            <div>
              <h4 className="text-md font-medium text-gray-700 mb-2">Risk Metrics</h4>
              <div className="bg-gray-50 rounded-md p-4">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
                  <div className="flex justify-between">
                    <span className="font-medium text-gray-600">Max Profit:</span>
                    <span className="text-green-600 font-semibold">
                      ${recommendation.risk_metrics.max_profit.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium text-gray-600">Max Loss:</span>
                    <span className="text-red-600 font-semibold">
                      ${Math.abs(recommendation.risk_metrics.max_loss).toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium text-gray-600">Breakeven:</span>
                    <span className="font-semibold">
                      {Array.isArray(recommendation.risk_metrics.breakeven)
                        ? recommendation.risk_metrics.breakeven.map(be => `$${be.toFixed(2)}`).join(', ')
                        : `$${recommendation.risk_metrics.breakeven.toFixed(2)}`
                      }
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium text-gray-600">Prob. of Profit:</span>
                    <span className="font-semibold">
                      {(recommendation.risk_metrics.prob_profit * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium text-gray-600">Return on Capital:</span>
                    <span className="font-semibold">
                      {(recommendation.risk_metrics.return_on_capital * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Reasoning Steps Accordion */}
      <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
        <div className="p-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Analysis Breakdown</h3>
        </div>
        <div className="divide-y divide-gray-200">
          {recommendation.reasoning_steps.map((step, index) => (
            <div key={index} className="p-4">
              <button
                onClick={() => toggleStep(index)}
                className="w-full flex items-center justify-between text-left hover:bg-gray-50 rounded-md p-2 -m-2 transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <div
                    className={`w-6 h-6 rounded-full flex items-center justify-center text-white text-sm font-bold ${
                      step.passed ? 'bg-green-500' : 'bg-red-500'
                    }`}
                  >
                    {step.passed ? '✓' : '✗'}
                  </div>
                  <span className="font-medium text-gray-900">{step.step}</span>
                  <span className="text-sm text-gray-500">
                    ({Math.round(step.confidence * 100)}% confidence)
                  </span>
                </div>
                <svg
                  className={`w-5 h-5 text-gray-400 transition-transform ${
                    expandedSteps.has(index) ? 'rotate-180' : ''
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </button>
              
              {expandedSteps.has(index) && (
                <div className="mt-3 pl-9">
                  <p className="text-gray-700 text-sm leading-relaxed">
                    {step.reasoning}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-center space-x-4">
        <button
          onClick={onReset}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors shadow-md hover:shadow-lg"
        >
          Analyze Another Stock
        </button>
      </div>
    </div>
  );
}