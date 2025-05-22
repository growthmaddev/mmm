"""
Seasonality extraction utilities for MMM
This module provides functionality to extract seasonality features from date columns
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys

def extract_seasonality_features(df, date_column='Date'):
    """
    Extract seasonality features from the date column
    
    Args:
        df: DataFrame containing the data
        date_column: Name of the date column
        
    Returns:
        DataFrame with added seasonality features
    """
    try:
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Ensure the date column is in datetime format
        if date_column in result_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(result_df[date_column]):
                try:
                    result_df[date_column] = pd.to_datetime(result_df[date_column], errors='coerce')
                    print(f"Converted {date_column} to datetime format", file=sys.stderr)
                except Exception as e:
                    print(f"Error converting {date_column} to datetime: {str(e)}", file=sys.stderr)
                    return df  # Return original if conversion fails
        else:
            print(f"Date column '{date_column}' not found in dataframe", file=sys.stderr)
            return df
        
        # Extract basic time components
        result_df['month'] = result_df[date_column].dt.month
        result_df['quarter'] = result_df[date_column].dt.quarter
        result_df['year'] = result_df[date_column].dt.year
        result_df['day_of_week'] = result_df[date_column].dt.dayofweek + 1  # 1-7 for Monday-Sunday
        result_df['week_of_year'] = result_df[date_column].dt.isocalendar().week
        
        # Create month and quarter dummies for better modeling
        month_dummies = pd.get_dummies(result_df['month'], prefix='month', drop_first=True)
        quarter_dummies = pd.get_dummies(result_df['quarter'], prefix='quarter', drop_first=True)
        
        # Add dummies to the dataframe
        result_df = pd.concat([result_df, month_dummies, quarter_dummies], axis=1)
        
        # Create holiday indicators (basic implementation)
        # This is a simplified approach - for production use, consider holiday libraries like holidays
        def is_holiday(date):
            # Check if it's a major US holiday (simplified)
            month, day = date.month, date.day
            
            # New Year's Day
            if month == 1 and day == 1:
                return 1
                
            # Independence Day
            if month == 7 and day == 4:
                return 1
                
            # Christmas Day
            if month == 12 and day == 25:
                return 1
                
            # Thanksgiving (simplified - 4th Thursday in November)
            if month == 11 and day >= 22 and day <= 28 and date.dayofweek == 3:
                return 1
                
            # Memorial Day (simplified - last Monday in May)
            if month == 5 and date.dayofweek == 0 and day > 24:
                return 1
                
            # Labor Day (simplified - first Monday in September)
            if month == 9 and date.dayofweek == 0 and day <= 7:
                return 1
                
            return 0
            
        result_df['is_holiday'] = result_df[date_column].apply(is_holiday)
        
        # Create sin/cos features for cyclic patterns
        result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
        result_df['quarter_sin'] = np.sin(2 * np.pi * result_df['quarter'] / 4)
        result_df['quarter_cos'] = np.cos(2 * np.pi * result_df['quarter'] / 4)
        result_df['day_of_week_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['day_of_week_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
        
        print(f"Successfully extracted seasonality features: {[col for col in result_df.columns if col not in df.columns]}", file=sys.stderr)
        return result_df
        
    except Exception as e:
        print(f"Error extracting seasonality features: {str(e)}", file=sys.stderr)
        return df  # Return original if extraction fails