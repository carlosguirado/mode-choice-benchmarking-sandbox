"""
Script to analyze and compare Swissmetro and ModeCanada datasets
"""

from mcbs.datasets import DatasetLoader
import pandas as pd
import numpy as np

def analyze_dataset(data, name):
    """Analyze a single dataset's structure and content."""
    print(f"\n{'='*20} {name} Dataset Analysis {'='*20}")
    
    # Basic info
    print("\nBasic Information:")
    print("-" * 50)
    print(f"Shape: {data.shape}")
    print("\nColumns:", ', '.join(data.columns))
    
    # Data types
    print("\nData Types:")
    print("-" * 50)
    for col in data.columns:
        print(f"{col:<20} {data[col].dtype}")
    
    # Check for string data
    string_cols = data.select_dtypes(include=['object']).columns
    if len(string_cols) > 0:
        print("\nColumns with string data:")
        print("-" * 50)
        for col in string_cols:
            unique_vals = data[col].unique()
            print(f"{col:<20} Unique values: {unique_vals}")
    
    # Value distributions
    print("\nValue Distributions:")
    print("-" * 50)
    
    # Special handling for choice variable
    if 'CHOICE' in data.columns:
        choice_col = 'CHOICE'
    elif 'choice' in data.columns:
        choice_col = 'choice'
    else:
        choice_col = None
        
    if choice_col:
        print(f"\n{choice_col} distribution:")
        print(data[choice_col].value_counts(dropna=False).sort_index())
    
    # Check for availability variables
    av_cols = [col for col in data.columns if '_AV' in col]
    if av_cols:
        print("\nAvailability variables:")
        for col in av_cols:
            print(f"\n{col} distribution:")
            print(data[col].value_counts(dropna=False).sort_index())
    
    # Basic statistics for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    print("\nNumeric columns statistics:")
    print("-" * 50)
    print(data[numeric_cols].describe())
    
    # Check for missing values
    missing = data.isnull().sum()
    if missing.any():
        print("\nMissing values:")
        print("-" * 50)
        print(missing[missing > 0])

def analyze_format(data, name):
    """Analyze if dataset is in wide or long format."""
    print(f"\n{name} Format Analysis:")
    print("-" * 50)
    
    # Check for indicators of long format
    has_alt_column = 'alt' in data.columns or 'ALT' in data.columns
    has_choice_column = 'choice' in data.columns or 'CHOICE' in data.columns
    
    # Check for mode-specific columns (indicators of wide format)
    columns = set(data.columns)
    mode_specific_cols = [col for col in columns if any(mode in col.upper() for mode in ['TRAIN', 'CAR', 'BUS', 'AIR', 'SM'])]
    
    if has_alt_column:
        print("Dataset appears to be in LONG format:")
        print("- Has alternative column")
        if 'alt' in data.columns:
            print(f"- Alternatives: {sorted(data['alt'].unique())}")
        elif 'ALT' in data.columns:
            print(f"- Alternatives: {sorted(data['ALT'].unique())}")
    elif mode_specific_cols:
        print("Dataset appears to be in WIDE format:")
        print("- Has mode-specific columns:", sorted(mode_specific_cols))
    else:
        print("Format unclear - needs further investigation")
    
    print("\nRelevant columns for format determination:")
    format_relevant_cols = []
    if has_alt_column:
        format_relevant_cols.append('alt' if 'alt' in columns else 'ALT')
    if has_choice_column:
        format_relevant_cols.append('choice' if 'choice' in columns else 'CHOICE')
    format_relevant_cols.extend(mode_specific_cols)
    
    for col in sorted(format_relevant_cols):
        if col in data.columns:
            print(f"\n{col} values:")
            print(data[col].value_counts(dropna=False).sort_index())

def main():
    """Load and analyze both datasets."""
    loader = DatasetLoader()
    
    # Load datasets
    swissmetro_data = loader.load_dataset("swissmetro_dataset")
    modecanada_data = loader.load_dataset("modecanada_dataset")
    
    # Analyze format
    analyze_format(swissmetro_data, "Swissmetro")
    analyze_format(modecanada_data, "ModeCanada")
    
    # Analyze content
    analyze_dataset(swissmetro_data, "Swissmetro")
    analyze_dataset(modecanada_data, "ModeCanada")
    
    # Compare datasets
    print("\n" + "="*20 + " Dataset Comparison " + "="*20)
    print("\nCommon columns:")
    common_cols = set(swissmetro_data.columns) & set(modecanada_data.columns)
    print(sorted(common_cols))
    
    print("\nSwissmetro-specific columns:")
    swiss_specific = set(swissmetro_data.columns) - set(modecanada_data.columns)
    print(sorted(swiss_specific))
    
    print("\nModeCanada-specific columns:")
    canada_specific = set(modecanada_data.columns) - set(swissmetro_data.columns)
    print(sorted(canada_specific))

if __name__ == "__main__":
    main()
