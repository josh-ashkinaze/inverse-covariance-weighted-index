[![Tests](https://github.com/josh-ashkinaze/icw-index/actions/workflows/tests.yml/badge.svg)](https://github.com/josh-ashkinaze/icw-index/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/icw-index.svg)](https://pypi.org/project/icw-index/)
[![Last updated](https://img.shields.io/github/last-commit/josh-ashkinaze/icw-index.svg)](https://github.com/josh-ashkinaze/icw-index/commits/main)


# Inverse-Covariance Weighted Index for Python

A Python implementation of the Inverse-Covariance Weighted (ICW) Index introduced by Anderson (2008) and implemented in Stata's `swindex` by Schwab et al. (2020). I validated this against Stata's `swindex` and produces effectively identical results.

## What is the ICW Index?
Tl;DR: The ICW index is a weighted average of variables where the weights are determined by the inverse of the covariance matrix of the variables.

Anderson (2008) proposed an index to combine multiple outcomes into a single measure using the inverse of the covariance matrix as weights. Why would you do this? Well first, people use indices all the time to avoid a multiple comparison problem. But usually, you would average the index variables so each counts equally. This can be sub-optimal if a bunch of variables all correlate with each other. You may want to up-weight the ones that are providing unique information. So the ICW index down-weights correlated outcomes and up-weights less correlated ones. Also, using the inverse covariance matrix as weights minimizes the variance of the resulting index.


## Install 

```python
pip install icw-index
```

## Usage
This library is a function, `icw_index`, that has two arguments:
- `index_vars`: A list of length `N` numpy arrays, each containing one variable to include in the index. All arrays must be the same length.
- `reference_mask`: (optional) A boolean numpy array of the same length as the index variables, indicating which observations to use as the reference group for normalization. If not provided, the full sample is used (which is the default in STATA `swindex`)

And `icw_index` returns a numpy array of length `N` ICW index values. ICW index values are normalized to have mean 0 and standard deviation 1 either
across the full sample (default) or within the reference group (if `reference_mask` is provided).

### Basic example with numpy arrays


Let's generate some synthetic data first. 
```python
import numpy as np
import pandas as pd
from icw import icw_index
np.random.seed(42)
n = 200
x1 = np.random.normal(loc=2, scale=10, size=n)
x2 = np.random.normal(loc=0.5, scale=4, size=n)
x3 = np.random.normal(loc=-0.2, scale=1, size=n)
```

Here, `index` will be a numpy array of length `n` containing the ICW index values, which have mean 0 and SD of 1. ICW values
can be interpreted as z-scores. 

```python
index = icw_index([x1, x2, x3])
print(f"shape={index.shape}, mean={index.mean():.1e}, std={index.std():.1e}")
# shape=(200,), mean=-1.3e-17, std=1.0e+00
```

### Basic example with Pandas DataFrame
The column `icw` now contains the ICW index values, normalized to have mean 0 and standard deviation 1 across the full sample.

``` python
# Example using Pandas dataframes 
df = pd.DataFrame({'var1': np.random.rand(100),
                   'var2': np.random.rand(100),
                   'var3': np.random.rand(100),
                   'treat': np.random.randint(0, 2, size=100)})

# Full sample normalization, no reference group. Entire index is distributed M=0, SD=1
df['icw'] = icw_index([df['var1'].values, df['var2'].values], df['var3'].values)
```

### Using a reference group for normalization
Now let's say we want to use the control group (where `treat == 0`) as the reference group for normalization instead of a full sample. We can do that by passing a boolean mask to the `reference_mask` argument.
This will mean that the ICW index is mean 0 and SD 1 for the control group, and treatment ICW values can be interpreted as effect sizes relative to the control group.
```python
import numpy as np
import pandas as pd
from icw import icw_index
np.random.seed(42)
df = pd.DataFrame({'var1': np.random.rand(100),
                   'var2': np.random.rand(100),
                   'var3': np.random.rand(100),
                   'treat': np.random.randint(0, 2, size=100)})

# Add some treatment effect to var1 and var2
for vars in ['var1', 'var2']:
    df[vars] = df[vars] + 0.5 * df['treat']

index_vars = ['var1', 'var2', 'var3']

# Normal ICW---full sample normalization
df['icw'] = icw_index([df[x].values for x in index_vars])

# Let's normalize to control group only
ref_mask = (df['treat'] == 0).values
df['icw_control_reference'] = icw_index(
    [df[x].values for x in index_vars],
    reference_mask=ref_mask,
)
```
Now let's take the data generated above and compare `icw` (full sample normalization) to `icw_control_reference` (control group normalization). 
- If we look at `icw`, we see that the treatment has a mean index of about 0.54, while the control group has a mean of about -0.64. So the delta is about 1.18 standard deviations. 
- Now if we look at `icw_control_reference`, we see that the control group is distributed mean 0 and SD 1 (by construction), while the treatment group has mean of about 1.18---which is the same delta as before. 
- So one of the neat things about using a reference group is that the treatment index values can be interpreted directly as effect sizes relative to the reference group, 
since they're in terms of reference group standard deviations.


|   treat |   icw_mean |   icw_sd |   icw_control_ref_mean |   icw_control_ref_sd |
|--------:|-----------:|---------:|-----------------------:|---------------------:|
|       0 |     -0.637 |    0.841 |                  0     |                1     |
|       1 |      0.543 |    0.784 |                  1.188 |                0.941 |

## Implementation Details

The implementation follows the procedure explained by Schwab et al. (2020). I'll quote their steps here for clarity...


We can calculate the standardized weighted index $\tilde{s}$ for each observation $i$ as follows:

1. Select $k$ indicators relevant for outcome $j$.

2. Adjust sign: For all $k$ indicators, ensure the positive direction always indicates a "better outcome".

3. Normalize indicators: Demean all $k$ indicators by subtracting the mean of the indicator in the reference group (the full sample is the default reference group). Then, convert them to effect sizes, $\tilde{y}_k$, by dividing each indicator by its reference group standard deviation.

4. Construct weights: Create weights using $\Sigma^{-1}$, the inverse of the covariance matrix of the normalized indicators. Specifically, set the weight $\tilde{w}_k$ on each indicator equal to the sum of its row entries in $\Sigma^{-1}$. With this rule, highly correlated indicators are assigned small or offsetting weights, while less correlated outcomes receive larger weights.

5. Construct index: Calculate the weighted average of $\tilde{y}_k$ for observation $i$. Formally, the weighted average $\overline{s}_i$ is calculated using $\tilde{s}_i = (1'\Sigma^{-1}1)^{-1}(1'\Sigma^{-1}\tilde{y}_i)$, where $\mathbf{1}$ is a column vector of 1s and $\tilde{y}_i$ is a column vector of all outcomes for observation $i$. This is an efficient GLS estimator.

6. Normalize index: Demean index $\overline{s}_i$ by subtracting the mean of the index in the reference group, and convert it to effect sizes by dividing it by its reference group standard deviation. This normalization results in an index distributed with mean zero and standard deviation one in the reference group.

## Validation

I validated this implementation against Stata's `swindex` (version 14) using 100 synthetic datasets:

- **Datasets**: 100 datasets with 5 variables each
- **Sample sizes**: 500-2000 observations per dataset  
- **Total observations**: 122,444
- **Variables**: Standard normal distribution, no missing data

### Results

Results are identical (within a floating point tolerance) to Stata's `swindex` implementation. Here are the two
options I tested. 

1. **Default settings** (full sample as reference group)
- Correlation: 0.999999999999996
- Differences > 1e-06: 0
- Max absolute difference: 3.08e-07
- Median absolute difference: 3.01e-08
- Mean absolute difference: 3.88e-08

2. **User-specified reference group** (using the control group as reference)
- Correlation: 1.000000000000000
- Differences > 1e-06: 0
- Max absolute difference: 3.31e-07
- Median absolute difference: 2.94e-08
- Mean absolute difference: 3.77e-08

## Limitations and Notes

- **No missing data**: For now, input arrays must not contain NaN values. You can impute or drop missing data before using this function.
- **User handles sign orientation**: Assumes input data is already oriented so higher values indicate better outcomes.
- **Report bugs**: I imagine I missed some edge cases. Feel free to report bugs. 

## System I Ran Tests On
I was using Python 3.13, `dev_requirements.txt` packages, MacOS, and Stata 19.5 for testing.

## References

- Schwab, B., Janzen, S., Magnan, N. P., & Thompson, W. M. (2020). Constructing a summary index using the standardized inverse-covariance weighted average of indicators. *The Stata Journal*, 20(4), 952-964.
- Anderson, M. L. (2008). Multiple Inference and Gender Differences in the Effects of Early Intervention: A Reevaluation of the Abecedarian, Perry Preschool, and Early Training Projects. *Journal of the American Statistical Association*, 103(484), 1481–1495.

## Citation

If you use this implementation in your work, please cite:

```bibtex
@misc{ashkinaze_icw_2025,
  author       = {Ashkinaze, Joshua},
  title        = {icw: Inverse-Covariance Weighted Index for Python},
  year         = {2025},
  url =        = {https://github.com/josh-ashkinaze/icw-index},
}
```

## Issues

Please open an issue if you find any bugs or edge cases.