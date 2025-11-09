"""
Author: Joshua Ashkinaze

Date: 2025-11-09

Description: Creates synthetic data to run tests on. Specifically, we have 100 datasets, each with 5 variables and between 80 and 200 observations.

Inputs: None

Outputs:
- test_datasets.csv: A CSV file containing the synthetic datasets
- run_swindex.do: A Stata do file to run swindex on all datasets and save results to swindex_results.csv. I'll use the latter to compare
against my implementation
"""


import numpy as np
import pandas as pd

np.random.seed(42)

all_data = []

for dataset_id in range(100):
    n_vars = 5
    n_obs = np.random.randint(100, 1000)
    data = np.random.randn(n_obs, n_vars)
    df = pd.DataFrame(data, columns=[f'var{i + 1}' for i in range(n_vars)])
    df['dataset_id'] = dataset_id
    df['obs_id'] = range(n_obs)
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
combined_df.to_csv('test_datasets.csv', index=False)

do_file_content = """* Stata do file to run swindex on all test datasets
ssc install swindex
clear all
set more off

* Load the data
import delimited "test_datasets.csv", clear

* Create a file to store results
file open results using "swindex_results.csv", write replace
file write results "dataset_id,index_value" _n

* foreach dataset, do the swindex...
quietly levelsof dataset_id, local(datasets)
foreach ds in `datasets' {

    preserve

    * Keep only this dataset
    keep if dataset_id == `ds'

    * Get list of variable columns (exclude dataset_id and obs_id)
    ds var*
    local varlist `r(varlist)'

    * Run swindex
    capture swindex `varlist', generate(anderson_index)

    if _rc == 0 {
        * Save the index values to results file
        forvalues i = 1/`=_N' {
            local idx_val = anderson_index[`i']
            file write results "`ds',`idx_val'" _n
        }
    }
    else {
        display "Error running swindex for dataset `ds'"
    }

    restore
}

file close results

display "results saved to swindex_results.csv"
"""

with open('run_swindex.do', 'w') as f:
    f.write(do_file_content)

print("\nCreated run_swindex.do file")
print("To run in Stata: do run_swindex.do")