"""
Unit tests for Monte Carlo aggregation behavior.
"""

import numpy as np

import ringdownanalysis.monte_carlo as monte_carlo
from ringdownanalysis.monte_carlo import MonteCarloAnalyzer


def test_run_filters_failed_trials_before_computing_stats(monkeypatch):
    """Monte Carlo stats should use only successful numeric errors."""
    trial_results = {
        0: (0.1, 0.2, 1.0, None),
        1: (None, -0.1, None, -2.0),
        2: (0.3, None, 3.0, 4.0),
    }

    def fake_process_single_trial(args):
        trial_idx = args[0]
        err_nls, err_dft, err_q_nls, err_q_dft = trial_results[trial_idx]
        success = {"nls": err_nls is not None, "dft": err_dft is not None}
        return trial_idx, err_nls, err_dft, err_q_nls, err_q_dft, success

    monkeypatch.setattr(monte_carlo, "_process_single_trial", fake_process_single_trial)

    result = MonteCarloAnalyzer().run(
        f0=5.0,
        fs=100.0,
        N=100,
        A0=1.0,
        snr_db=40.0,
        Q=100.0,
        n_mc=3,
        seed=123,
    )

    np.testing.assert_allclose(result["errors_nls"], np.array([0.1, 0.3]))
    np.testing.assert_allclose(result["errors_dft"], np.array([0.2, -0.1]))
    np.testing.assert_allclose(result["errors_q_nls"], np.array([1.0, 3.0]))
    np.testing.assert_allclose(result["errors_q_dft"], np.array([-2.0, 4.0]))

    assert result["errors_nls"].dtype.kind == "f"
    assert np.isclose(result["stats"]["nls"]["mean"], 0.2)
    assert np.isclose(result["stats"]["q_dft"]["rmse"], np.sqrt(10.0))
