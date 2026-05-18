# Spearman Correlation with Uncertainty

Implements Spearman rank correlation with measurement uncertainties, based on Curran (2014): https://arxiv.org/abs/1411.3816v2  
Includes support for asymmetric errors via a split-normal distribution.

## Quickstart

```python
from asymcorr import CorrelationUncertainty
import numpy as np

x = np.arange(1, 6)
y = 2 * x + np.random.default_rng(0).normal(scale=0.5, size=5)
cu = CorrelationUncertainty(x, y, xerr=np.full(5, 0.1), yerr=np.full(5, 0.2))
rhos, z = cu.composite(n=1000)
print("median rho:", np.median(rhos))
```