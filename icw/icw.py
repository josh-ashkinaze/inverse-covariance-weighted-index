def icw_index(arrays, reference_mask=None):
    """
    Create ICW index from multiple indicator arrays.

    Follows Schwab (2020) and Stata's swindex with restrictions:
    - No missing values allowed in input arrays

    Args:
    - arrays: List of 1D arrays (indicators, same length, no NaNs)
    - reference_mask: Boolean array indicating reference group for covariance calculation. This is equivalent to STATA's swindex
    normby() option.If None, uses all observations as the reference group, which is the most straightforward.

    Returns:
    - 1D array (ICW index) normalized to mean=0, sd=1 for reference group.
    - If reference is full sample, entire index has mean=0, sd=1.
    - If reference is (eg) control group, then control group has mean=0, sd=1 and treated group ICWs are relative to that.

    Usage:
        x1 = np.random.rand(100)
        x2 = np.random.rand(100)
        index = icw_index([x1, x2])

        # Example using Pandas dataframes and reference group normalization
        df = pd.DataFrame({'var1': np.random.rand(100),
                            'var2': np.random.rand(100),
                            'treat': np.random.randint(0, 2, size=100)})
        ref_mask = (df['treat'] == 0).values

        # Full sample normalization, no reference group. Entire index is distributed M=0, SD=1
        df['icw'] = icw_index([df['var1'].values, df['var2'].values])

        # User-specified reference group normalization. Control group is distributed M=0, SD=1 and treatment group is relative to that.
        df['icw_control_reference'] = icw_index([df['var1'].values, df['var2'].values],
                                         reference_mask=ref_mask)
    """
    import numpy as np

    # Validate arrays
    for arr in arrays:
        if np.isnan(arr).any():
            raise ValueError("Input arrays must not contain NaN values.")

    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) != 1:
        raise ValueError("All input arrays must have the same length.")

    # Validate and set reference mask
    if reference_mask is not None:
        if len(reference_mask) != lengths[0]:
            raise ValueError("Reference mask must be same length as input arrays.")
        if reference_mask.dtype != bool:
            raise ValueError("Reference mask must be a boolean array.")
        if np.isnan(reference_mask).any():
            raise ValueError("Reference mask must not contain NaN values.")
    else:
        reference_mask = np.ones(lengths[0], dtype=bool)

    # Stack indicators and extract reference subset
    indicators = np.column_stack(arrays)
    ref_indicators = indicators[reference_mask, :]

    # Normalize indicators using reference group stats
    means = np.mean(ref_indicators, axis=0)
    sds = np.std(ref_indicators, axis=0, ddof=1)
    normalized_all = (indicators - means) / sds

    # Compute weights from inverse covariance of reference group
    cov_matrix = np.cov(normalized_all, rowvar=False)
    inv_cov = np.linalg.inv(cov_matrix)
    weights = np.sum(inv_cov, axis=1)

    # Construct index for all observations
    index_all = np.dot(normalized_all, weights)

    # Normalize using reference group statistics
    index_ref = index_all[reference_mask]
    index_mean = np.mean(index_ref)
    index_sd = np.std(index_ref, ddof=1)
    final_index = (index_all - index_mean) / index_sd

    return final_index
