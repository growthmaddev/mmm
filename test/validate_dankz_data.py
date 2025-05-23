#!/usr/bin/env python3
"""
Data Validation Script for Dankz Test Data

This script analyzes the Dankz test dataset to validate its structure and contents 
for compatibility with our MarketMixMaster implementation.
"""

import pandas as pd
import re
from datetime import datetime
import os
import sys

# Find the v2 version of the dataset
def find_data_file():
    attached_file = "./attached_assets/dankztestdata_v2.csv"
    if os.path.exists(attached_file):
        return attached_file
    
    # Look for uploads
    uploads = "./uploads/"
    if os.path.exists(uploads):
        for file in os.listdir(uploads):
            if "dankztestdata_v2.csv" in file:
                return os.path.join(uploads, file)
    
    # Fallback to any available version
    if os.path.exists("./attached_assets/dankztestdata.csv"):
        return "./attached_assets/dankztestdata.csv"
    
    # Look for any version in uploads
    if os.path.exists(uploads):
        for file in os.listdir(uploads):
            if "dankztestdata" in file:
                return os.path.join(uploads, file)
    
    return None

def detect_date_format(date_col):
    """Detect the date format in the column"""
    sample = date_col.dropna().iloc[0] if not date_col.empty else None
    if not sample:
        return "No dates found"
    
    # Try to determine if DD/MM/YYYY or MM/DD/YYYY
    try:
        parts = sample.split('/')
        if len(parts) == 3:
            day_part = int(parts[0])
            month_part = int(parts[1])
            # If day > 12, it must be DD/MM/YYYY
            if day_part > 12:
                return "DD/MM/YYYY"
            # If month > 12, it must be MM/DD/YYYY 
            elif month_part > 12:
                return "MM/DD/YYYY"
            else:
                # Ambiguous, need more analysis
                return "Ambiguous (could be DD/MM/YYYY or MM/DD/YYYY)"
    except:
        pass
    
    # Try other common formats
    for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y"]:
        try:
            datetime.strptime(str(sample), fmt)
            return fmt
        except:
            continue
            
    return "Unknown date format"

def has_comma_separators(col):
    """Check if numeric column contains comma separators"""
    # Convert to string first to handle numeric types
    string_vals = col.astype(str)
    return any(',' in val for val in string_vals if not pd.isna(val))

def identify_column_type(name):
    """Identify the type of column based on name patterns"""
    name_lower = name.lower()
    
    if "date" in name_lower or "week" in name_lower:
        return "Date"
    elif "spend" in name_lower or "cost" in name_lower:
        return "Marketing Channel Spend"
    elif "sales" in name_lower or "revenue" in name_lower or "conversion" in name_lower:
        return "Target Variable"
    elif any(x in name_lower for x in ["temp", "weather", "holiday", "promo", "season"]):
        return "Control Variable"
    else:
        return "Unknown"

def analyze_data(file_path):
    """Analyze the dataset and return a summary report"""
    print(f"Analyzing file: {file_path}")
    
    try:
        # Try first with no specific parsing to see raw values
        df_raw = pd.read_csv(file_path, nrows=5)
        print("\nRAW DATA PREVIEW (First 5 rows, no parsing):")
        print(df_raw.head())
        
        # Now load the full dataset
        df = pd.read_csv(file_path)
        
        # Basic dataset information
        total_rows = len(df)
        total_cols = len(df.columns)
        
        # Detect date columns
        date_cols = []
        for col in df.columns:
            col_type = identify_column_type(col)
            if col_type == "Date":
                date_cols.append(col)
        
        # Analyze column types and details
        column_analysis = []
        marketing_channels = []
        target_variables = []
        control_variables = []
        
        for col in df.columns:
            col_type = df[col].dtype
            sample = df[col].dropna().iloc[0] if not df[col].empty and not df[col].dropna().empty else None
            identified_type = identify_column_type(col)
            
            has_commas = has_comma_separators(df[col]) if col_type != 'object' or pd.api.types.is_numeric_dtype(df[col]) else False
            missing_values = df[col].isna().sum()
            missing_pct = (missing_values / total_rows) * 100
            
            # Store in relevant category
            if identified_type == "Marketing Channel Spend":
                marketing_channels.append(col)
            elif identified_type == "Target Variable":
                target_variables.append(col)
            elif identified_type == "Control Variable":
                control_variables.append(col)
            
            column_analysis.append({
                "name": col,
                "pandas_type": str(col_type),
                "identified_type": identified_type,
                "sample_value": str(sample),
                "has_comma_separators": has_commas,
                "missing_values": missing_values,
                "missing_percentage": f"{missing_pct:.2f}%",
            })
        
        # Date format detection
        date_formats = {}
        for col in date_cols:
            date_formats[col] = detect_date_format(df[col])
        
        # Generate summary report
        summary = {
            "file_path": file_path,
            "total_rows": total_rows,
            "total_columns": total_cols,
            "column_analysis": column_analysis,
            "date_columns": date_cols,
            "date_formats": date_formats,
            "marketing_channels": marketing_channels,
            "target_variables": target_variables,
            "control_variables": control_variables,
        }
        
        return summary
    
    except Exception as e:
        print(f"Error analyzing data: {str(e)}")
        return None

def print_report(summary):
    """Print a formatted summary report"""
    if not summary:
        print("No analysis available.")
        return
    
    print("\n" + "="*80)
    print(f"DATASET VALIDATION REPORT")
    print("="*80)
    
    print(f"\nFile: {summary['file_path']}")
    print(f"Total Rows (Weeks of Data): {summary['total_rows']}")
    print(f"Total Columns: {summary['total_columns']}")
    
    print("\n" + "-"*80)
    print("COLUMN ANALYSIS:")
    print("-"*80)
    
    for col in summary['column_analysis']:
        print(f"\n{col['name']} ({col['identified_type']})")
        print(f"  Data Type: {col['pandas_type']}")
        print(f"  Sample Value: {col['sample_value']}")
        print(f"  Has Comma Separators: {col['has_comma_separators']}")
        print(f"  Missing Values: {col['missing_values']} ({col['missing_percentage']})")
    
    print("\n" + "-"*80)
    print("DATE FORMAT DETECTION:")
    print("-"*80)
    
    for col, fmt in summary['date_formats'].items():
        print(f"{col}: {fmt}")
    
    print("\n" + "-"*80)
    print("IDENTIFIED COLUMNS:")
    print("-"*80)
    
    print("\nMarketing Channels:")
    for channel in summary['marketing_channels']:
        print(f"  - {channel}")
    
    print("\nTarget Variables:")
    for target in summary['target_variables']:
        print(f"  - {target}")
    
    print("\nControl Variables:")
    for control in summary['control_variables']:
        print(f"  - {control}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    # Evaluate data for MMM readiness
    issues = []
    
    # Check if we have enough data
    if summary['total_rows'] < 8:
        issues.append("CRITICAL: Less than 8 weeks of data available. MMM requires at least 8 weeks for reliable results.")
    
    # Check if we have date column(s)
    if not summary['date_columns']:
        issues.append("CRITICAL: No date column detected. MMM requires a date column.")
    
    # Check if we have marketing channels
    if len(summary['marketing_channels']) < 2:
        issues.append("CRITICAL: Less than 2 marketing channels detected. MMM works best with multiple channels.")
    
    # Check if we have a target variable
    if not summary['target_variables']:
        issues.append("CRITICAL: No target variable (sales/revenue) detected. MMM requires a target variable.")
    
    # Check for missing values
    missing_data_cols = [col['name'] for col in summary['column_analysis'] 
                        if col['missing_values'] > 0 and col['identified_type'] in 
                        ["Marketing Channel Spend", "Target Variable", "Date"]]
    if missing_data_cols:
        issues.append(f"WARNING: Missing values in important columns: {', '.join(missing_data_cols)}")
    
    # Check date format
    date_format_issues = []
    for col, fmt in summary['date_formats'].items():
        if fmt != "DD/MM/YYYY" and "Ambiguous" in fmt:
            date_format_issues.append(f"{col} ({fmt})")
    
    if date_format_issues:
        issues.append(f"WARNING: Date format might not be DD/MM/YYYY in columns: {', '.join(date_format_issues)}")
    
    # Check for comma separators in numeric columns
    comma_cols = [col['name'] for col in summary['column_analysis'] 
                 if col['has_comma_separators'] and col['identified_type'] == "Marketing Channel Spend"]
    
    if comma_cols:
        issues.append(f"NOTE: Comma separators found in numeric columns: {', '.join(comma_cols)}. Our system handles this correctly.")
    
    # Print issues or all-clear
    if issues:
        print("\nData issues found:")
        for issue in issues:
            print(f"  - {issue}")
        
        if any("CRITICAL" in issue for issue in issues):
            print("\nRECOMMENDATION: Data preprocessing required before MMM workflow can proceed.")
        else:
            print("\nRECOMMENDATION: Data is usable but should be checked for the warnings above.")
    else:
        print("\nNo issues found! Data is ready for MMM workflow testing.")

def main():
    # Find the data file
    data_file = find_data_file()
    if not data_file:
        print("Error: Could not find dankztestdata_v2.csv or any version of it.")
        sys.exit(1)
    
    # Analyze the data
    summary = analyze_data(data_file)
    
    # Print the report
    print_report(summary)

if __name__ == "__main__":
    main()