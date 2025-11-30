"""
Unit tests for ICW index implementation.

Compares the Anderson index computed by Stata's swindex command to the icw_index function from icw.py.
"""

import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from icw.icw import icw_index
import pandas as pd
import numpy as np


FLOAT_TOL = 1e-6


class TestICWIndex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv('test_datasets.csv')
        cls.stata_results = pd.read_csv('swindex_results.csv')
        cls.stata_results_normby = pd.read_csv('swindex_normby_results.csv')

    def compute_python_results(self, use_reference_mask=False):
        """
        Compute ICW index for all datasets.

        Args:
            use_reference_mask: Whether to use reference mask (normby option)

        Returns:
            List of dicts with results
        """
        results = []
        for dataset_id in self.df['dataset_id'].unique():
            dataset = self.df[self.df['dataset_id'] == dataset_id].reset_index(drop=True)
            var_cols = ['var1', 'var2', 'var3', 'var4', 'var5']
            arrays = [dataset[col].values for col in var_cols]

            if use_reference_mask:
                ref_mask = (dataset['treat_status'] == 0).values
                index_values = icw_index(arrays, reference_mask=ref_mask)
            else:
                index_values = icw_index(arrays)

            for obs_id, val in enumerate(index_values):
                results.append({
                    'dataset_id': dataset_id,
                    'obs_id': obs_id,
                    'index_value': val
                })
        return results

    def compare_results(self, stata_results, python_results):
        """
        Compare Stata and Python results.

        Returns:
            Dict with comparison statistics
        """
        python_df = pd.DataFrame(python_results)
        merged = stata_results.merge(
            python_df,
            on=['dataset_id', 'obs_id'],
            suffixes=('_stata', '_python')
        )

        correlation = merged['index_value_stata'].corr(merged['index_value_python'])

        return {
            'n_datasets': len(merged['dataset_id'].unique()),
            'n_obs': len(merged),
            'correlation': correlation,
            'stata_values': merged['index_value_stata'].values,
            'python_values': merged['index_value_python'].values
        }

    def test_full_sample_normalization(self):
        """Test full sample normalization matches Stata swindex"""
        python_results = self.compute_python_results(use_reference_mask=False)
        stats = self.compare_results(self.stata_results, python_results)

        all_close = np.allclose(stats['stata_values'], stats['python_values'],
                               atol=FLOAT_TOL, rtol=0)

        self.assertTrue(all_close,
                       "Python and Stata results differ by more than tolerance")
        self.assertGreater(stats['correlation'], 0.9999,
                          f"Correlation {stats['correlation']:.15f} is too low")
        print("Full sample stats")
        print(f"Datasets: {stats['n_datasets']}, Observations: {stats['n_obs']}, Correlation: {stats['correlation']:.15f},"
              f"n Differences > {FLOAT_TOL}: {np.sum(np.abs(stats['stata_values'] - stats['python_values']) > FLOAT_TOL)}")

    def test_control_group_normalization(self):
        """Test control group normalization matches Stata swindex normby"""
        python_results = self.compute_python_results(use_reference_mask=True)
        stats = self.compare_results(self.stata_results_normby, python_results)

        all_close = np.allclose(stats['stata_values'], stats['python_values'],
                               atol=FLOAT_TOL, rtol=0)

        self.assertTrue(all_close,
                       "Python and Stata results differ by more than tolerance")
        self.assertGreater(stats['correlation'], 0.9999,
                          f"Correlation {stats['correlation']:.15f} is too low")
        print("Full sample stats")
        print(f"Datasets: {stats['n_datasets']}, Observations: {stats['n_obs']}, Correlation: {stats['correlation']:.15f},"
              f"n Differences > {FLOAT_TOL}: {np.sum(np.abs(stats['stata_values'] - stats['python_values']) > FLOAT_TOL)}")

    def test_input_validation(self):
        """Test input validation"""
        x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        # Test NaN values
        x1_nan = x1.copy()
        x1_nan[0] = np.nan
        with self.assertRaises(ValueError):
            icw_index([x1_nan, x2])

        # Test different lengths
        x3 = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            icw_index([x1, x3])

        # Test reference mask length mismatch
        ref_mask = np.array([True, False, True])
        with self.assertRaises(ValueError):
            icw_index([x1, x2], reference_mask=ref_mask)

        # Test non-boolean reference mask
        ref_mask = np.array([1, 0, 1, 0, 1])
        with self.assertRaises(ValueError):
            icw_index([x1, x2], reference_mask=ref_mask)


if __name__ == '__main__':
    unittest.main(verbosity=2)