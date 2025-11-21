import pytest
import numpy as np
from asymcorr import CorrelationUncertainty


def test_validate_inputs():
    # Test matching lengths
    x = [1, 2, 3]
    y = [4, 5, 6]
    cu = CorrelationUncertainty(x, y)

    # Test mismatched lengths
    with pytest.raises(ValueError):
        CorrelationUncertainty([1, 2], [3, 4, 5])

    # Test negative errors
    with pytest.raises(ValueError):
        CorrelationUncertainty(x, y, xerr=[0.1, -0.2, 0.1])

    # Test incorrect error shape
    with pytest.raises(ValueError):
        CorrelationUncertainty(x, y, xerr=[[0.1, 0.2], [0.1, 0.2]])

    # Test nan_policy raise with NaNs
    with pytest.raises(ValueError):
        CorrelationUncertainty([1, 2, np.nan], [4, 5, 6], nan_policy="raise")

    # Test invalid nan_policy
    with pytest.raises(ValueError):
        CorrelationUncertainty(x, y, nan_policy="invalid")


def test_perturbation_sampling():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 6.0, 7.0, 8.0, 7.0])
    xerr = np.array([0.1, 0.2, 0.1, 0.2, 0.1])
    yerr = np.array([0.2, 0.1, 0.2, 0.1, 0.2])

    cu = CorrelationUncertainty(x, y, xerr=xerr, yerr=yerr)
    rho, pvals = cu.perturbation(n=1000)
    median_rho = np.median(rho)
    assert median_rho > 0.5


def test_bootstrap_resampling():
    x = np.arange(1, 1001)
    y = np.arange(10, 1010)
    cu = CorrelationUncertainty(x, y)
    rho, pvals = cu.bootstrap(n=1000)
    median_rho = np.median(rho)
    print(median_rho)
    assert median_rho > 0.5


def test_composite_sampling():
    x = np.arange(1, 501)
    y = np.arange(20, 520)
    xerr = np.full(500, 0.5)
    yerr = np.full(500, 0.5)
    cu = CorrelationUncertainty(x, y, xerr=xerr, yerr=yerr)
    rho, pvals = cu.composite(n=1000)
    median_rho = np.median(rho)
    assert median_rho > 0.5


def test_nan_handling():
    x = np.arange(1, 100).astype(float)
    y = np.arange(10, 109).astype(float)
    for i in np.random.default_rng(0).choice(len(x), size=10, replace=False):
        if i % 2 == 0:
            y[i] = np.nan
        else:
            x[i] = np.nan

    # Test propagate
    cu = CorrelationUncertainty(x, y, nan_policy="propagate")
    rho, pvals = cu.bootstrap(n=100)
    assert np.isnan(rho).any()

    # Test omit
    cu = CorrelationUncertainty(x, y, nan_policy="omit")
    rho, pvals = cu.bootstrap(n=100)
    median_rho = np.median(rho)
    assert median_rho > 0.5
