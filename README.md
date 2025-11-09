# Inverse-Covariance Weighted Index for Python

A Python implementation of the Inverse-Covariance Weighted (ICW) Index introduced by Anderson (2008) and implemented in Stata's `swindex` by Schwab et al. (2020). I validated this against Stata's `swindex` and produces effectively identical results.

## Quick Start

```python
import numpy as np
import pandas as pd
from icw import icw_index # or just copy-paste the icw_index function directly from icw.py

# Example 1: From Numpy arrays
x1 = np.random.rand(100)
x2 = np.random.rand(100)
index = icw_index([x1, x2])

# Example 2: From Pandas DataFrame columns
df = pd.DataFrame({
    'var1': np.random.rand(100), 
    'var2': np.random.rand(100)
})
df['icw'] = icw_index([df['var1'].values, df['var2'].values])
```

## What is the ICW Index?
Tl;DR: The ICW index is a weighted average of variables where the weights are determined by the inverse of the covariance matrix of the variables.

Anderson (2008) proposed an index to combine multiple outcomes into a single measure using the inverse of the covariance matrix as weights. Why would you do this? Well first, people use indices all the time to avoid a multiple comparison problem. But usually, you would average the index variables so each counts equally. This can be sub-optimal if a bunch of variables all correlate with each other. You may want to up-weight the ones that are providing unique information. So the ICW index down-weights correlated outcomes and up-weights less correlated ones. Also, using the inverse covariance matrix as weights minimizes the variance of the resulting index.

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
- **Sample sizes**: 100-1000 observations per dataset  
- **Total observations**: 13,861
- **Variables**: Standard normal distribution, no missing data

### Results

- **Correlation**: 0.999999999999999
- **Max absolute difference**: 2.42e-07
- **Median absolute difference**: 2.88e-08
- **Mean absolute difference**: 3.71e-08
- **Differences > 1e-06**: 0

The results are effectively identical to Stata's implementation.

## Limitations

This implementation is simpler than `swindex` and has the following restrictions:

- **No missing data**: Input arrays must not contain NaN values
- **Full sample reference**: Always uses the full dataset as the reference group
- **User handles sign orientation**: Assumes input data is already oriented so higher values indicate better outcomes

If you need more flexibility, feel free to submit a pull request (with tests against `swindex`).

## System I Ran Tests On
I was using Python 3.13, `requirements.txt` packages, MacOS, and Stata 19.5 for testing.

## References

- Schwab, B., Janzen, S., Magnan, N. P., & Thompson, W. M. (2020). Constructing a summary index using the standardized inverse-covariance weighted average of indicators. *The Stata Journal*, 20(4), 952-964.
- Anderson, M. L. (2008). Multiple Inference and Gender Differences in the Effects of Early Intervention: A Reevaluation of the Abecedarian, Perry Preschool, and Early Training Projects. *Journal of the American Statistical Association*, 103(484), 1481–1495.

## Citation

If you use this implementation in your work, please cite:

```bibtex
@misc{icw_index,
  author = {Joshua Ashkinaze},
  title = {Inverse-Covariance Weighted Index for Python},
  year = {2025},
  url = {https://github.com/josh-ashkinaze/inverse-covariance-weighted-index}
}
```

## Issues

Please open an issue if you find any bugs or edge cases.