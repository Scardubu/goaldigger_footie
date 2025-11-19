"""
Data pipeline package for football betting insights platform.
Handles data collection, integration, transformation, and loading.
"""
from scripts.data_pipeline.db_integrator import DataIntegrator

__all__ = ["DataIntegrator"]
