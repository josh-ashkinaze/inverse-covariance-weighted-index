def icw_index(arrays):
    import numpy as np
    """
    Create ICW index from multiple indicator arrays. This function follows the methodology
    outlined in Schwab (2020) and Stata's `swindex` with restrictions that:
    - No missing values are allowed in the input arrays. You can preprocess your data to handle missingness before using this function.
    - I always use the full sample covariance matrix for weights (no sub-sampling).

    Args:
        arrays: List of 1D Numpy arrays, each representing an indicator variable. All arrays must be of the same length and contain no NaNs.

    Returns:
        A 1D Numpy array representing the Anderson index, normalized to have mean 0 and standard deviation 1, of same length as input arrays.

    Usage:
        # Example 1---straight from Numpy arrays
        >>> x1 = np.random.rand(100)
        >>> x2 = np.random.rand(100)
        >>> index = icw_index([x1, x2]) # returns icw index as Numpy array

        # Example 2---Pandas DataFrame columns
        >>> df = pd.DataFrame({'var1': np.random.rand(100), 'var2': np.random.rand(100)})
        >>> df['icw'] = icw_index(arrays=[df['var1'].values, df['var2'].values])

    """

    # Validations
    ###########
    # 1. Ensure no nans in input arrays...if so, throw error.
    for arr in arrays:
        if np.isnan(arr).any():
            raise ValueError("Input arrays must not contain NaN values.")

    # 2. Ensure all arrays have same length
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) != 1:
        raise ValueError("All input arrays must have the same length.")

    indicators = np.column_stack(arrays)

    # Normalize indicators
    means = np.mean(indicators, axis=0)
    demeaned = indicators - means
    sds = np.std(indicators, axis=0, ddof=1)
    normalized = demeaned / sds

    # Construct weights
    cov_matrix = np.cov(normalized, rowvar=False)
    inv_cov = np.linalg.inv(cov_matrix)

    # Weight on each indicator equals sum of its row entries in inverse covariance
    weights = np.sum(inv_cov, axis=1)

    # Construct index as weighted average of normalized indicators
    index = np.dot(normalized, weights)

    # Normalize index
    index_mean = np.mean(index)
    index_demeaned = index - index_mean
    index_sd = np.std(index, ddof=1)
    final_index = index_demeaned / index_sd

    return final_index


