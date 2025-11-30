"""
Author: Joshua Ashkinaze

Date: 2025-11-09

Description: Compares the Anderson index computed by Stata's swindex command to the icw_index function from icw.py.

Inputs:
- test_datasets.csv: A CSV file containing some synthetic datasets
- swindex_results.csv: A CSV file containing the Anderson index results computed by Stata's swindex command, which I treat as ground truth.
- swindex_normby_results.csv: A CSV file containing the Anderson index results with normby option

Outputs:
- Prints comparison statistics between the Anderson index computed by Stata's swindex command and the icw_index function from icw.py.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from icw.icw import icw_index
import pandas as pd

FLOAT_TOL = 1e-6


def compare_results(df, stata_results, my_results, test_name):
    """
    Compare Stata and Python implementations and print statistics.

    Args:
        df: Full dataset
        stata_results: DataFrame with Stata results
        my_results: List of dicts with Python results
        test_name: Name of the test for printing
    """
    mydf = pd.DataFrame(my_results)

    merged = stata_results.merge(
        mydf,
        on=['dataset_id', 'obs_id'],
        suffixes=('_stata', '_mine')
    )

    correlation = merged['index_value_stata'].corr(merged['index_value_mine'])
    diff = merged['index_value_stata'] - merged['index_value_mine']

    print(f"\n{'='*70}")
    print(f"{test_name}")
    print(f"{'='*70}")
    print(f"Datasets: {len(merged['dataset_id'].unique())}")
    print(f"Observations: {len(merged)}")
    print(f"\nCorrelation: {correlation:.15f}")
    print(f"Differences > {FLOAT_TOL}: {(diff.abs() > FLOAT_TOL).sum()}")
    print(f"Max absolute difference: {diff.abs().max():.2e}")
    print(f"Median absolute difference: {diff.abs().median():.2e}")
    print(f"Mean absolute difference: {diff.abs().mean():.2e}")

    if (diff.abs() > FLOAT_TOL).sum() > 0:
        print(f"\nDistribution of differences > {FLOAT_TOL}:")
        print(f"  25th percentile: {diff.abs().quantile(0.25):.2e}")
        print(f"  75th percentile: {diff.abs().quantile(0.75):.2e}")
        print(f"  95th percentile: {diff.abs().quantile(0.95):.2e}")
        print(f"  99th percentile: {diff.abs().quantile(0.99):.2e}")


df = pd.read_csv('test_datasets.csv')
print(f"Columns in test_datasets.csv: {df.columns.tolist()}")
print("\nComparing Stata swindex to Python icw_index")
print("="*70)

# Test 1: Full sample normalization
stata_results = pd.read_csv('swindex_results.csv')

my_results = []
for dataset_id in df['dataset_id'].unique():
    dataset = df[df['dataset_id'] == dataset_id].reset_index(drop=True)
    var_cols = ['var1', 'var2', 'var3', 'var4', 'var5']
    arrays = [dataset[col].values for col in var_cols]
    my_index = icw_index(arrays)
    for obs_id, myval in enumerate(my_index):
        my_results.append({
            'dataset_id': dataset_id,
            'obs_id': obs_id,
            'index_value': myval
        })

compare_results(df, stata_results, my_results, "Test 1: Full Sample Normalization (default)")

# Test 2: Control group normalization
stata_results_normby = pd.read_csv('swindex_normby_results.csv')

my_results_normby = []
for dataset_id in df['dataset_id'].unique():
    dataset = df[df['dataset_id'] == dataset_id].reset_index(drop=True)
    var_cols = ['var1', 'var2', 'var3', 'var4', 'var5']
    arrays = [dataset[col].values for col in var_cols]
    ref_mask = (dataset['treat_status'] == 0).values
    my_index = icw_index(arrays, reference_mask=ref_mask)
    for obs_id, myval in enumerate(my_index):
        my_results_normby.append({
            'dataset_id': dataset_id,
            'obs_id': obs_id,
            'index_value': myval
        })

compare_results(df, stata_results_normby, my_results_normby, "Test 2: Control Group Normalization (normby)")