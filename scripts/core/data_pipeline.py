# Create new file: scripts/core/data_pipeline.py

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataPipeline:
    """Handles data transformation, cleaning, and feature engineering"""
    
    def __init__(self, validator):
        self.validator = validator
        self.transformations = {}
        
    def register_transformation(self, name, transform_func):
        """Register a transformation function"""
        self.transformations[name] = transform_func
        
    def process_data(
        self, 
        data: pd.DataFrame, 
        transformations_to_apply: list = None, 
        column_types: dict = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Process data through validation and transformations.
        Returns: (validated_data, validation_report)
        """
        validated_data, validation_report = self.validator.validate_dataset(data, column_types=column_types)
        
        # Apply requested transformations
        if transformations_to_apply:
            for transform_name in transformations_to_apply:
                if transform_name in self.transformations:
                    try:
                        validated_data = self.transformations[transform_name](validated_data)
                        logger.info(f"Applied transformation: {transform_name}")
                    except Exception as e:
                        logger.error(f"Error applying transformation {transform_name}: {str(e)}")
                        
        return validated_data, validation_report
        
    # Example transformation functions
    @staticmethod
    def create_rolling_averages(df, columns, windows=[3, 5, 10]):
        """Create rolling averages for specified columns"""
        result = df.copy()
        for col in columns:
            if col in df.columns:
                for window in windows:
                    result[f"{col}_rolling_{window}"] = df[col].rolling(window=window).mean()
        return result