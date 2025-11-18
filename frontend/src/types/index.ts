// Type definitions for the Options Trade Evaluator

export interface TickerResult {
  ticker: string;
  company_name: string;
  exchange: string;
}

export interface TickerSearchResponse {
  results: TickerResult[];
  count: number;
}

export interface Contract {
  action: "BUY" | "SELL";
  type: "CALL" | "PUT";
  strike: number;
  expiration: string;
  quantity: number;
  premium_credit?: number;
}

export interface RiskMetrics {
  max_profit: number;
  max_loss: number;
  breakeven: number | number[];
  prob_profit: number;
  return_on_capital: number;
}

export interface ReasoningStep {
  step: string;
  passed: boolean;
  reasoning: string;
  confidence: number;
}

export interface TradeRecommendation {
  should_trade: boolean;
  confidence: number;
  strategy?: string;
  contracts: Contract[];
  risk_metrics?: RiskMetrics;
  reasoning_steps: ReasoningStep[];
}

export interface TradeAnalysisResult {
  ticker: string;
  company_name: string;
  recommendation: TradeRecommendation;
}

export interface TradeAnalysisRequest {
  ticker: string;
}