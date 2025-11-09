"""
Author: Joshua Ashkinaze

Date: 2025-11-09

Description: Compares the Anderson index computed by Stata's swindex command to the icw_index function from icw.py.

Inputs:
- test_datasets.csv: A CSV file containing some synthetic datasets
- swindex_results.csv: A CSV file containing the Anderson index results computed by Stata's swindex command, which I treat as ground truth.

Outputs:
- Prints comparison statistics between the Anderson index computed by Stata's swindex command and the icw_index function from icw.py.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from icw import icw_index
import pandas as pd

FLOAT_TOL = 1e-6 # tolerance for floating point comparisons or just language differences



df = pd.read_csv('test_datasets.csv')

# Load Stata swindex results
##############################################
stata_results = pd.read_csv('swindex_results.csv')
stata_results.columns = ['dataset_id', 'index_value']
stata_results['obs_id'] = stata_results.groupby('dataset_id').cumcount()


# Compute indices my way
##############################################
my_results = []
for dataset_id in df['dataset_id'].unique():
    dataset = df[df['dataset_id'] == dataset_id]
    var_cols = ['var1', 'var2', 'var3', 'var4', 'var5']
    arrays = [dataset[col].values for col in var_cols]
    my_index = icw_index(arrays)
    for obs_id, myval in enumerate(my_index):
        my_results.append({
            'dataset_id': dataset_id,
            'obs_id': obs_id,
            'index_value': myval
        })
mydf = pd.DataFrame(my_results)

# Merge
#######################
merged = stata_results.merge(
    mydf,
    on=['dataset_id', 'obs_id'],
    suffixes=('_stata', '_yours')
)


# Calculate statistics
print("Comparing swindex frm stata to icw_index from icw.py")
print("Across {} datasets and {} observations:".format(len(merged['dataset_id'].unique()), len(merged)))
correlation = merged['index_value_stata'].corr(merged['index_value_yours'])
diff = merged['index_value_stata'] - merged['index_value_yours']
print(f"Correlation: {correlation:.15f}")
print(f"Number of differences > {FLOAT_TOL}: {(diff.abs() > FLOAT_TOL).sum()}")
print(f"Max absolute difference: {diff.abs().max():.2e}")
print(f"Median absolute difference: {diff.abs().median():.2e}")
print(f"Mean absolute difference: {diff.abs().mean():.2e}")

