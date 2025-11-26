import numpy as np
from scipy.stats import spearmanr, norm


class CorrelationUncertainty:
    """
    Compute Spearman correlation under measurement uncertainty using:

    - Monte Carlo perturbation sampling
    - Bootstrap resampling
    - Composite (MC + bootstrap) sampling
    """

    def __init__(self, x, y, xerr=None, yerr=None, random_state=None, nan_policy="propagate"):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.xerr = xerr
        self.yerr = yerr
        self.nan_policy = nan_policy
        self.rng = np.random.default_rng(random_state)
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate shapes and convert errors to 2xN arrays."""
        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)

        if len(self.x) != len(self.y) or len(self.x) == 0:
            raise ValueError("x and y must have the same non-zero length")

        self.xerr = self._validate_error(self.xerr, len(self.x))
        self.yerr = self._validate_error(self.yerr, len(self.y))

        if self.nan_policy not in ["propagate", "raise", "omit"]:
            raise ValueError("nan_policy must be one of 'propagate', 'raise', or 'omit'")
        if self.nan_policy == "raise" and (np.isnan(self.x).any() or np.isnan(self.y).any()):
            raise ValueError("Input data contains NaNs, but nan_policy is set to 'raise'")

    def _validate_error(self, err, n):
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


    def _compute_pearson(self,x_samples,y_samples):
        """Compute Pearson correlation for each sample pair."""
        
        x_centred = x_samples - np.mean(x_samples, axis=1, keepdims=True)
        y_centred = y_samples - np.mean(y_samples, axis=1, keepdims=True)

        numerator = np.sum(x_centred * y_centred, axis=1)
        denominator = np.sqrt(np.sum(x_centred**2, axis=1) * np.sum(y_centred**2, axis=1))
        return numerator / denominator
    
    def _compute_spearman(self,x_samples,y_samples):
        """Compute Spearman correlation for each sample pair."""

        x_ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 1, x_samples)
        y_ranks = np.apply_along_axis(lambda y: np.argsort(np.argsort(y)), 1, y_samples)
        return self._compute_pearson(x_ranks, y_ranks)




    def split_normal(self, mu, sigma_left, sigma_right, size=1):
        """
        Sample from a split (asymmetric) normal distribution.
        Left and right std devs determine which side is used.
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

    def prepare_samples_mc(self, n, indices=None):
        """Prepare Monte Carlo perturbed samples for x and y."""

        if indices is not None:
            x = self.x[indices]
            y = self.y[indices]
            xerr = self.xerr[:, indices]
            yerr = self.yerr[:, indices]
        else:
            x = self.x
            y = self.y
            xerr = self.xerr
            yerr = self.yerr

        x_samples = self.split_normal(x, xerr[0], xerr[1], size=(n, len(x)))
        y_samples = self.split_normal(y, yerr[0], yerr[1], size=(n, len(y)))
        return x_samples, y_samples

    # ----------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------
    def perturbation(self, n=10000):
        """
        Monte Carlo perturbation sampling.
        Returns arrays of rho and p values.
        """
        x_samples, y_samples = self.prepare_samples_mc(n)
        rhos = np.empty(n)

        for i in range(n):
            rhos[i], _ = spearmanr(x_samples[i], y_samples[i], nan_policy=self.nan_policy)

        return rhos

    def bootstrap(self, n=10000):
        """
        Standard bootstrap sampling of (x, y) pairs.
        """
        indices = self.rng.integers(0, len(self.x), size=(n, len(self.x)))

        rhos = np.empty(n)

        for i in range(n):
            rhos[i], _ = spearmanr(
                self.x[indices[i]],
                self.y[indices[i]],
                nan_policy=self.nan_policy,
            )

        return rhos

    def composite(self, n=10000):
        """
        Composite method:
        bootstrap indices + Monte Carlo perturbation for each bootstrap sample.
        """

        indices = self.rng.integers(0, len(self.x), size=(n, len(self.x)))

        rhos = np.empty(n)
        for i, idx in enumerate(indices):
            x_s, y_s = self.prepare_samples_mc(1, indices=idx)
            x_s = x_s.flatten()
            y_s = y_s.flatten()
            rhos[i], _ = spearmanr(x_s, y_s, nan_policy=self.nan_policy)

        return rhos

    def _fisher_transformation(self, rho):
        rho = np.clip(rho, -0.9999, 0.9999)
        return np.arctanh(rho)

    def z_score(self, rho, N):
        """Compute z-score for Spearman's rho using Fisher transformation."""
        return self._fisher_transformation(rho) * np.sqrt((N - 3) / 1.06)

    def compare_methods(self, n=10000, print_summary=True):
        """
        Compare all three methods + a standard calculation without uncertainty.
        Returns a dictionary of results or/and prints the summary.
        """
        results = {}

        rho, pval = spearmanr(self.x, self.y, nan_policy=self.nan_policy)
        results["standard"] = {rho, pval}
        rhos = self.perturbation(n)
        results["perturbation"] = self.summarise(rhos)
        rhos = self.bootstrap(n)
        results["bootstrap"] = self.summarise(rhos)
        rhos = self.composite(n)
        results["composite"] = self.summarise(rhos)

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

    @staticmethod
    def summarise(rhos, sigma=1, z_score=None):
        """
        Summarise correlation results with median, std of rho and C.I. of p-values and significance fraction of p<0.05.
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
    def print_summary(summary):
        """
        Print summary dictionary in a readable format.
        """
        rho_median = f'Rho mean: {summary["rho_mean"]:.2f} ± {summary["rho_std"]:.2f}'
        cis = f'CI: ({summary["rho_ci"][0]:.2f}, {summary["rho_ci"][1]:.2f})'
        z_score = (
            f'Z mean: {summary["z_mean"]:.2f} ± {summary["z_std"]:.2f}'
            if summary["z_mean"] is not None
            else "Z mean: N/A"
        )
        print(rho_median, cis, z_score, sep="\n")
