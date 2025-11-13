"""
LangChain chains for options trade evaluation.

This module contains all the analysis chains used in the sequential
evaluation workflow:
- TechnicalAnalysisChain: Evaluates technical indicators
- FundamentalScreeningChain: Screens fundamental health
- OptionsAnalysisChain: Analyzes options contract quality
- StrategySelectionChain: Selects optimal trading strategy
- RiskAssessmentChain: Calculates risk metrics and final recommendation
- OptionsEvaluationAgent: Orchestrates the complete workflow
"""

from app.chains.technical_analysis_chain import (
    TechnicalAnalysisChain,
    TechnicalAnalysisResult
)
from app.chains.fundamental_screening_chain import (
    FundamentalScreeningChain,
    FundamentalScreeningResult
)
from app.chains.options_analysis_chain import (
    OptionsAnalysisChain,
    OptionsAnalysisResult
)
from app.chains.strategy_selection_chain import (
    StrategySelectionChain,
    StrategyRecommendation
)
from app.chains.risk_assessment_chain import (
    RiskAssessmentChain,
    RiskAssessmentResult,
    RiskMetrics,
    ContractDetail
)
from app.chains.options_evaluation_agent import (
    OptionsEvaluationAgent
)

__all__ = [
    'TechnicalAnalysisChain',
    'TechnicalAnalysisResult',
    'FundamentalScreeningChain',
    'FundamentalScreeningResult',
    'OptionsAnalysisChain',
    'OptionsAnalysisResult',
    'StrategySelectionChain',
    'StrategyRecommendation',
    'RiskAssessmentChain',
    'RiskAssessmentResult',
    'RiskMetrics',
    'ContractDetail',
    'OptionsEvaluationAgent'
]
