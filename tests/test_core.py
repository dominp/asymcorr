import pytest
import numpy as np
from asymcorr import CorrelationUncertainty



def test_initialization():
    """Test basic initialization"""
    x = np.arange(10)
    y = np.arange(10, 20)
    xerr = np.full(10, 0.1)
    yerr = np.full(10, 0.2)

    cu = CorrelationUncertainty(x, y, xerr=xerr, yerr=yerr)
    assert np.array_equal(cu.x, x)
    assert np.array_equal(cu.y, y)
    assert np.array_equal(cu.xerr, np.vstack((xerr, xerr)))
    assert np.array_equal(cu.yerr, np.vstack((yerr, yerr)))

    cu2 = CorrelationUncertainty(x, y)  
    assert np.array_equal(cu2.xerr, np.zeros((2, 10)))
    assert np.array_equal(cu2.yerr, np.zeros((2, 10)))  

    cu3 = CorrelationUncertainty(x, y, xerr)
    assert np.array_equal(cu3.xerr, np.vstack((xerr, xerr)))
    assert np.array_equal(cu3.yerr, np.zeros((2, 10)))

def test_negative_errors():
    x = [1, 2, 3]
    y = [4, 5, 6]
    xerr = [0.1, -0.2, 0.1]
    yerr = [0.2, 0.1, 0.2]

    with pytest.raises(ValueError):
        CorrelationUncertainty(x, y, xerr=xerr, yerr=yerr)

def test_incorrect_error_shape():
    x = [1, 2, 3]
    y = [4, 5, 6]
    xerr = [[0.1, 0.2], [0.1, 0.2]]
    yerr = [0.2, 0.1, 0.2]

    with pytest.raises(ValueError):
        CorrelationUncertainty(x, y, xerr=xerr, yerr=yerr)


def test_mismatched_lengths():
    x = [1, 2, 3]
    y = [4, 5]
    with pytest.raises(ValueError):
        CorrelationUncertainty(x, y)

def test_invalid_nan_policy():
    x = [1, 2, 3]
    y = [4, 5, 6]
    with pytest.raises(ValueError):
        cu = CorrelationUncertainty(x, y, nan_policy="error_please")


def test_nan_policy():    
    x = [1, 2, 3]
    x_nan = [1, np.nan, 3]
    y = [4, 5, 6]
    y_nan = [4, np.nan, 6]
    xerr = [0.1, 0.2, 0.1]
    xerr_nan = [0.1, np.nan, 0.1]
    yerr = [0.2, 0.1, 0.2]
    yerr_nan = [0.2, np.nan, 0.2]

    with pytest.raises(ValueError):
        CorrelationUncertainty(x_nan, y, nan_policy="raise")
    with pytest.raises(ValueError):
        CorrelationUncertainty(x, y_nan, nan_policy="raise")
    with pytest.raises(ValueError):
        CorrelationUncertainty(x, y, xerr=xerr_nan, nan_policy="raise")
    with pytest.raises(ValueError):
        CorrelationUncertainty(x, y, yerr=yerr_nan, nan_policy="raise")
    with pytest.raises(ValueError):
        CorrelationUncertainty(x_nan, y_nan, nan_policy="raise")
    with pytest.raises(ValueError):
        CorrelationUncertainty(x, y, xerr=xerr_nan, yerr=yerr_nan, nan_policy="raise")  
    with pytest.raises(ValueError):
        CorrelationUncertainty(x_nan, y_nan, xerr=xerr_nan, yerr=yerr_nan, nan_policy="raise")

    cu = CorrelationUncertainty(x_nan, y, nan_policy="omit")
    assert len(cu.x) == 2
    cu = CorrelationUncertainty(x, y_nan, nan_policy="omit")
    assert len(cu.x) == 2
    cu = CorrelationUncertainty(x, y, xerr=xerr_nan, nan_policy="omit")
    assert len(cu.x) == 2
    cu = CorrelationUncertainty(x, y, yerr=yerr_nan, nan_policy="omit")
    assert len(cu.x) == 2



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


def test_error_correlation():
    x = np.random.default_rng(0).normal(loc=0, scale=1, size=200)
    y = 2 * x + np.random.default_rng(0).normal(loc=0, scale=0.5, size=200)
    xerr = np.full(200, 0.2)
    yerr = np.full(200, 0.2)
    rhos = []
    z_scores = []
    for err_scale in np.arange(0, 10, 2):
        cu = CorrelationUncertainty(x, y, xerr=xerr * err_scale, yerr=yerr * err_scale)
        rho, z_score = cu.composite(n=500)
        median_rho = np.median(rho)
        median_z = np.median(z_score)
        rhos.append(median_rho)
        z_scores.append(median_z)
    print(rhos, z_scores)
    assert all(rhos[i] >= rhos[i + 1] for i in range(len(rhos) - 1))
    assert all(z_scores[i] >= z_scores[i + 1] for i in range(len(z_scores) - 1))


def test_nan_handling():
    x = np.arange(1, 100).astype(float)
    y = np.arange(10, 109).astype(float)
    xerr = np.full(99, 0.1)
    yerr = np.full(99, 0.1)
    x_nan = x.copy()
    y_nan = y.copy()
    for i in np.random.default_rng(0).choice(len(x), size=20, replace=False):
        if i % 2 == 0:
            y_nan[i] = np.nan
            yerr[i] = np.nan

        else:
            x_nan[i] = np.nan
            xerr[i] = np.nan

    # Test raise
    with pytest.raises(ValueError):
        cu = CorrelationUncertainty(x_nan, y_nan, nan_policy="raise")

    # Test raise if nan in errors
    with pytest.raises(ValueError):
        cu = CorrelationUncertainty(x, y, xerr=xerr, yerr=yerr, nan_policy="raise")

    # Test omit
    cu = CorrelationUncertainty(x_nan, y_nan, nan_policy="omit")
    rho, pvals = cu.bootstrap(n=100)
    median_rho = np.median(rho)
    assert median_rho > 0.5

    # Test omit with errors
    cu = CorrelationUncertainty(x_nan, y_nan, xerr=xerr, yerr=yerr, nan_policy="omit")
    rho, pvals = cu.composite(n=100)
    median_rho = np.median(rho)
    assert median_rho > 0.5
