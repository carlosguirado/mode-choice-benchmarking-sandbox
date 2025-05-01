#!/usr/bin/env python3
"""
Example demonstrating the fetch_data functionality for downloading and using datasets
from the Mode Choice Benchmarking Sandbox (MCBS).
"""

import pandas as pd
import matplotlib.pyplot as plt
from mcbs.datasets import fetch_data

def main():
    # List available datasets
    print("Fetching Swissmetro dataset...")
    
    # Download the dataset using fetch_data
    # This will automatically cache it in ~/.mcbs/datasets
    df = fetch_data("swissmetro_dataset")
    
    # Display basic information about the dataset
    print(f"\nDataset shape: {df.shape}")
    print(f"Available columns: {', '.join(df.columns)}")
    
    # Basic exploratory analysis
    print(f"\nChoice distribution:")
    choice_counts = df['CHOICE'].value_counts()
    print(choice_counts)
    
    # You can also get features and target separately
    X, y = fetch_data("swissmetro_dataset", return_X_y=True)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Example with a different dataset
    print("\nFetching LTDS dataset...")
    ltds_df = fetch_data("ltds_dataset")
    print(f"LTDS dataset shape: {ltds_df.shape}")
    
    # Basic visualization example
    plt.figure(figsize=(10, 6))
    choice_counts.plot(kind='bar')
    plt.title('Distribution of Transportation Mode Choices')
    plt.xlabel('Mode')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('swissmetro_choices.png')
    print("Created visualization 'swissmetro_choices.png'")
    
    # Example of filtering and analyzing data
    print("\nAnalyzing travel patterns by purpose:")
    purpose_groups = df.groupby('PURPOSE')
    
    for purpose, group in purpose_groups:
        choice_by_purpose = group['CHOICE'].value_counts(normalize=True) * 100
        print(f"\nPurpose {purpose} mode share (%):")
        print(choice_by_purpose)

if __name__ == "__main__":
    main()