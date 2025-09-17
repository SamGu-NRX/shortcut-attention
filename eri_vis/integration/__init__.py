"""
ERI Visualization Integration Package

This package provides integration components for connecting the ERI visualization
system with the existing Mammoth continual learning framework.

Components:
- mammoth_integration: Bridge between EinstellungEvaluator and visualization system
- hooks: Experiment lifecycle hooks for automatic visualization generation
"""

from .mammoth_integration import MammothERIIntegration

__all__ = ['MammothERIIntegration']
