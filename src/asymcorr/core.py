import numpy as np
from scipy.stats import spearmanr, norm
from typing import Optional, Union, Tuple


class CorrelationUncertainty:
    """
    Compute Spearman or Pearson correlation under measurement uncertainty using:

    - Monte Carlo perturbation sampling
    - Bootstrap resampling
    - Composite (MC + bootstrap) sampling

    The class supports asymmetric measurement uncertainties.
    """

    def __init__(
        self,
        x: Union[list, np.ndarray],
        y: Union[list, np.ndarray],
        xerr: Optional[Union[list, np.ndarray]] = None,
        yerr: Optional[Union[list, np.ndarray]] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        nan_policy: str = "raise",
    ):
        """
        Initialize CorrelationUncertainty instance.

        Parameters
        ----------
        x : Union[list, np.ndarray]
            X data
        y : Union[list, np.ndarray]
            Y data
        xerr : Optional[Union[list, np.ndarray]], optional
            X measurement uncertainties. Can be:
            - 1D array (shape (n,)): symmetric errors for each point
            - 2D array (shape (2, n)): asymmetric errors, where [0, :] is lower (left), [1, :] is upper (right)
            By default None.
        yerr : Optional[Union[list, np.ndarray]], optional
            Y measurement uncertainties, same format as xerr.
        random_state : Optional[Union[int, np.random.Generator]], optional
            Random state for reproducibility, by default None
        nan_policy : str, optional
            How to handle NaNs: "raise" or "omit", by default "raise"
        """

        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.xerr = xerr
        self.yerr = yerr
        self.nan_policy = nan_policy
        self.rng = np.random.default_rng(random_state)
        self._validate_inputs()

    def _validate_inputs(self):
        """
        Validate input data and errors.

        Raises
        ------
        ValueError
            If input data lengths do not match or if errors are invalid
        ValueError
            If nan_policy is invalid
        """
        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)

        if len(self.x) != len(self.y) or len(self.x) == 0:
            raise ValueError("x and y must have the same non-zero length")

        self.xerr = self._validate_error(self.xerr, len(self.x))
        self.yerr = self._validate_error(self.yerr, len(self.y))

        self.nan_policy = self.nan_policy.strip().lower()
        if self.nan_policy not in ["omit", "raise"]:
            raise ValueError("nan_policy must be one of 'raise' or 'omit'")
        self._nan_policy_filter()

    def _validate_error(self, err: Optional[Union[list, np.ndarray]], n: int) -> np.ndarray:
        """
        Validate error arrays.

        Parameters
        ----------
        err : Optional[Union[list, np.ndarray]]
            Error array, either symmetric (1D) or asymmetric (2D)
        n : int
            Length of the data array

        Returns
        -------
        np.ndarray
            Validated error array of shape (2, n)
        """
        if err is None:
            return np.zeros((2, n))

        err = np.asarray(err)
        if err.ndim == 1:
            if len(err) != n:
                raise ValueError("Error array length must match data length")
            if np.any(err < 0):
                raise ValueError("Errors must be non-negative")
            return np.vstack([err, err])

        elif err.ndim == 2:
            if err.shape != (2, n):
                raise ValueError("Asymmetric error array must have shape (2, len(data))")
            if np.any(err < 0):
                raise ValueError("Errors must be non-negative")
            return err
        else:
            raise ValueError("Error array must be 1D or 2D")

    def _compute_pearson(self, x_samples: np.ndarray, y_samples: np.ndarray) -> np.ndarray:
        """
        Compute Pearson correlation for each sample pair.

        Parameters
        ----------
        x_samples : np.ndarray
            X data samples
        y_samples : np.ndarray
            Y data samples

        Returns
        -------
        np.ndarray
            Array of Pearson correlation coefficients
        """

        x_centred = x_samples - np.mean(x_samples, axis=1, keepdims=True)
        y_centred = y_samples - np.mean(y_samples, axis=1, keepdims=True)

        numerator = np.sum(x_centred * y_centred, axis=1)
        denominator = np.sqrt(np.sum(x_centred**2, axis=1) * np.sum(y_centred**2, axis=1))
        return numerator / denominator

    def _compute_spearman(self, x_samples: np.ndarray, y_samples: np.ndarray) -> np.ndarray:
        """
        Compute Spearman correlation for each sample pair.

        Parameters
        ----------
        x_samples : np.ndarray
            X data samples
        y_samples : np.ndarray
            Y data samples

        Returns
        -------
        np.ndarray
            Array of Spearman correlation coefficients
        """

        x_ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 1, x_samples)
        y_ranks = np.apply_along_axis(lambda y: np.argsort(np.argsort(y)), 1, y_samples)
        return self._compute_pearson(x_ranks, y_ranks)

    def _nan_policy_filter(self):
        """
        Apply nan_policy to filter out NaN values.
        """
        mask = np.isnan(self.x) | np.isnan(self.y)
        mask |= np.isnan(self.xerr).any(axis=0) | np.isnan(self.yerr).any(axis=0)

        if np.any(mask) and self.nan_policy == "raise":
            raise ValueError("Input data contains NaNs, but nan_policy is set to 'raise'")
        elif np.all(mask):
            raise ValueError("All data points are NaNs")

        if self.nan_policy == "omit":
            self.x = self.x[~mask]
            self.y = self.y[~mask]
            self.xerr = self.xerr[:, ~mask]
            self.yerr = self.yerr[:, ~mask]

    def compute_correlation(
        self, x_samples: np.ndarray, y_samples: np.ndarray, method: str = "spearman"
    ) -> np.ndarray:
        """Compute correlation between X and Y using selected method.

        Parameters
        ----------
        x_samples : np.ndarray
            X data samples
        y_samples : np.ndarray
            Y data samples
        method : str, optional
            Correlation method to use, by default "spearman", can be "pearson"

        Returns
        -------
        np.ndarray
            Array of correlation coefficients

        Raises
        ------
        ValueError
            If method is not recognized
        """

        method = method.strip().lower()
        if method == "spearman":
            return self._compute_spearman(x_samples, y_samples)
        elif method == "pearson":
            return self._compute_pearson(x_samples, y_samples)
        else:
            raise ValueError("Method must be 'spearman' or 'pearson'")

    def split_normal(
        self,
        mu: Union[float, list, np.ndarray],
        sigma_left: Union[float, list, np.ndarray],
        sigma_right: Union[float, list, np.ndarray],
        size: int = 1,
    ) -> np.ndarray:
        """Generate random samples from a split normal distribution.

        Parameters
        ----------
        mu : float, list, np.ndarray
            Mean value(s)
        sigma_left : float, list, np.ndarray
            Left-side (lower) standard deviation(s) (corresponds to lower measurement uncertainty)
        sigma_right : float, list, np.ndarray
            Right-side (upper) standard deviation(s) (corresponds to upper measurement uncertainty)
        size : int, optional
            Number of samples to generate, by default 1

        Returns
        -------
        np.ndarray
            Random samples from the split normal distribution
        """
        mu = np.asarray(mu)
        sigma_left = np.asarray(sigma_left)
        sigma_right = np.asarray(sigma_right)

        # Safe elementwise division
        denom = sigma_left + sigma_right
        p_left = np.divide(sigma_left, denom, out=np.full_like(denom, 0.5, dtype=float), where=denom > 0)

        u = self.rng.uniform(0, 1, size=size)

        return np.where(
            u < p_left,
            self.rng.normal(loc=mu, scale=sigma_left, size=size),
            self.rng.normal(loc=mu, scale=sigma_right, size=size),
        )

    def prepare_samples_mc(self, n: int, indices: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Monte Carlo samples for x and y considering measurement uncertainties.

        Parameters
        ----------
        n : int
            Number of samples to generate
        indices : np.ndarray, optional
            Indices to select specific data points, by default None

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Monte Carlo samples for x and y
        """

        if indices is not None:
            x = self.x[indices]
            y = self.y[indices]
            xerr = self.xerr[:, indices]
            yerr = self.yerr[:, indices]
            dims = x.shape
        else:
            x = self.x
            y = self.y
            xerr = self.xerr
            yerr = self.yerr
            dims = (n, len(x))

        x_samples = self.split_normal(x, xerr[0], xerr[1], size=dims)
        y_samples = self.split_normal(y, yerr[0], yerr[1], size=dims)
        return x_samples, y_samples

    def perturbation(
        self, n: int = 10000, method: str = "spearman", return_z_score: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate correlation using Monte Carlo perturbation sampling.

        Parameters
        ----------
        n : int, optional
            Number of simulations, by default 10000
        method : str, optional
            Correlation method to use ("spearman" or "pearson"), by default "spearman"
        return_z_score : bool, optional
            Whether to return z-scores, by default True

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Array of correlation coefficients, and optionally z-scores, depending on return_z_score
        """

        x_samples, y_samples = self.prepare_samples_mc(n)
        rhos = self.compute_correlation(x_samples, y_samples, method=method)
        return self._return_z_scores(return_z_score, rhos, len(self.x))

    def bootstrap(
        self, n: int = 10000, method: str = "spearman", return_z_score: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate correlation using bootstrap method. This method ignores uncertainties.

        Parameters
        ----------
        n : int, optional
            Number of simulations, by default 10000
        method : str, optional
            Correlation method to use ("spearman" or "pearson"), by default "spearman"
        return_z_score : bool, optional
            Whether to return z-scores, by default True

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Array of correlation coefficients, and optionally z-scores, depending on return_z_score
        """

        indices = self.rng.integers(0, len(self.x), size=(n, len(self.x)))
        x_samples = self.x[indices]
        y_samples = self.y[indices]
        rhos = self.compute_correlation(x_samples, y_samples, method=method)
        return self._return_z_scores(return_z_score, rhos, len(self.x))

    def composite(
        self, n: int = 10000, method: str = "spearman", return_z_score: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate correlation using composite method combining bootstrap resampling and Monte Carlo perturbation.

        Parameters
        ----------
        n : int, optional
            Number of simulations, by default 10000
        method : str, optional
            Correlation method to use ("spearman" or "pearson"), by default "spearman"
        return_z_score : bool, optional
            Whether to return z-scores, by default True
        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Array of correlation coefficients, and optionally z-scores, depending on return_z_score
        """

        indices = self.rng.integers(0, len(self.x), size=(n, len(self.x)))
        x_samples, y_samples = self.prepare_samples_mc(n, indices=indices)
        rhos = self.compute_correlation(x_samples, y_samples, method=method)
        return self._return_z_scores(return_z_score, rhos, len(self.x))

    def _fisher_transformation(self, rho: np.ndarray) -> np.ndarray:
        """
        Apply Fisher transformation to Spearman's rho values.
        Parameters
        ----------
        rho : np.ndarray
            Array of Spearman's rho values

        Returns
        -------
        np.ndarray
            Array of Fisher-transformed values
        """
        rho = np.clip(rho, -0.9999, 0.9999)
        return np.arctanh(rho)

    def z_score(self, rho: np.ndarray, N: int) -> np.ndarray:
        """
        Compute z-score for Spearman's rho using Fisher transformation.
        Parameters
        ----------
        rho : np.ndarray
            Array of Spearman's rho values
        N : int
            Sample size

        Returns
        -------
        np.ndarray
            Array of z-scores
        """
        return self._fisher_transformation(rho) * np.sqrt((N - 3) / 1.06)

    def _return_z_scores(
        self, return_z_score: bool, rhos: np.ndarray, N: int
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Helper function to return z-scores if requested.

        Parameters
        ----------
        return_z_score : bool
            Whether to return z-scores
        rhos : np.ndarray
            Array of correlation coefficients
        N : int
            Sample size

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Array of correlation coefficients, and optionally z-scores, depending on return_z_score
        """

        if return_z_score:
            z_scores = self.z_score(rhos, N)
            return rhos, z_scores
        return rhos

    def compare_methods(
        self, n: int = 10000, method: str = "spearman", print_summary: bool = True, return_z_score: bool = True
    ) -> dict:
        """
        Compare all three methods + a standard calculation without uncertainty.

        Parameters
        ----------
        n : int, optional
            Number of simulations, by default 10000
        method : str, optional
            Correlation method to use ("spearman" or "pearson"), by default "spearman"
        print_summary : bool, optional
            Whether to print the summary, by default True
        return_z_score : bool, optional
            Whether to return z-scores, by default True
        Returns
        -------
        dict
            Dictionary of results
        """
        results = {}

        rho, pval = spearmanr(self.x, self.y, nan_policy=self.nan_policy)
        results["standard"] = {rho, pval}
        rhos, z_score = self.perturbation(n, method=method, return_z_score=return_z_score)
        results["perturbation"] = self.summarise(rhos, z_score=z_score)
        rhos, z_score = self.bootstrap(n, method=method, return_z_score=return_z_score)
        results["bootstrap"] = self.summarise(rhos, z_score=z_score)
        rhos, z_score = self.composite(n, method=method, return_z_score=return_z_score)
        results["composite"] = self.summarise(rhos, z_score=z_score)
        if print_summary:
            rho, pval = results["standard"]
            pval = f"{pval:.2e}" if pval < 0.001 else f"{pval:.3f}"
            print(f"Standard method: {rho:.2f} (p={pval})")
            print(f"---" * 5)
            for method, summary in results.items():
                if method == "standard":
                    continue
                print(method.capitalize())
                self.print_summary(summary)
                print(f"---" * 5)
        return results

    @staticmethod
    def summarise(rhos: np.ndarray, sigma: int = 1, z_score: np.ndarray = None) -> dict:
        """
        Summarise correlation results.

        Parameters
        ----------
        rhos : np.ndarray
            Array of correlation coefficients
        sigma : int, optional
            Number of standard deviations for confidence interval, by default 1
        z_score : np.ndarray, optional
            Array of z-scores, by default None

        Returns
        -------
        dict
            Summary dictionary
        """
        sigma = norm.sf(sigma)
        if z_score is not None:
            z_mean = np.mean(z_score)
            z_std = np.std(z_score)
        else:
            z_mean = None
            z_std = None

        output = {
            "rho_mean": np.mean(rhos),
            "rho_std": np.std(rhos),
            "rho_ci": (
                np.percentile(rhos, sigma * 100),  # 15.9th percentile
                np.percentile(rhos, (1 - sigma) * 100),
            ),
            "z_mean": z_mean,
            "z_std": z_std,
        }
        return output

    @staticmethod
    def print_summary(summary: dict):
        """
        Print summary dictionary in a readable format.

        Parameters
        ----------
        summary : dict
            Summary dictionary
        """

        rho_median = f'Rho mean: {summary["rho_mean"]:.2f} ± {summary["rho_std"]:.2f}'
        cis = f'CI: ({summary["rho_ci"][0]:.2f}, {summary["rho_ci"][1]:.2f})'
        z_score = (
            f'Z mean: {summary["z_mean"]:.2f} ± {summary["z_std"]:.2f}'
            if summary["z_mean"] is not None
            else "Z mean: N/A"
        )
        print(rho_median, cis, z_score, sep="\n")
