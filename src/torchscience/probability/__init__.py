"""Probability distributions with differentiable CDF, PDF, PPF, and SF.

This module provides functional operators for probability distributions,
complementing torch.distributions with:
- CDFs for all distributions (not just 6)
- PPF (quantile/inverse CDF) functions
- Second-order gradient support for all operators

Example
-------
>>> import torch
>>> from torchscience.probability import normal_cumulative_distribution, normal_quantile
>>>
>>> # Compute CDF
>>> x = torch.tensor([0.0, 1.0, 2.0])
>>> p = normal_cumulative_distribution(x)  # tensor([0.5, 0.8413, 0.9772])
>>>
>>> # Compute quantiles
>>> probs = torch.tensor([0.025, 0.5, 0.975])
>>> quantiles = normal_quantile(probs)  # tensor([-1.96, 0.0, 1.96])
"""

from ._beta import (
    beta_cumulative_distribution,
    beta_log_probability_density,
    beta_probability_density,
    beta_quantile,
    beta_survival,
)
from ._binomial import (
    binomial_cumulative_distribution,
    binomial_log_probability_mass,
    binomial_probability_mass,
    binomial_survival,
)
from ._chi2 import (
    chi2_cumulative_distribution,
    chi2_log_probability_density,
    chi2_probability_density,
    chi2_quantile,
    chi2_survival,
)
from ._exceptions import DomainError, ProbabilityError
from ._f import (
    f_cumulative_distribution,
    f_log_probability_density,
    f_probability_density,
    f_quantile,
    f_survival,
)
from ._gamma import (
    gamma_cumulative_distribution,
    gamma_log_probability_density,
    gamma_probability_density,
    gamma_quantile,
    gamma_survival,
)
from ._normal import (
    normal_cumulative_distribution,
    normal_log_probability_density,
    normal_probability_density,
    normal_quantile,
    normal_survival,
)
from ._poisson import (
    poisson_cumulative_distribution,
    poisson_log_probability_mass,
    poisson_probability_mass,
    poisson_survival,
)

__all__ = [
    "DomainError",
    "ProbabilityError",
    # Beta distribution
    "beta_cumulative_distribution",
    "beta_log_probability_density",
    "beta_probability_density",
    "beta_quantile",
    "beta_survival",
    # Binomial distribution
    "binomial_cumulative_distribution",
    "binomial_log_probability_mass",
    "binomial_probability_mass",
    "binomial_survival",
    # Chi-squared distribution
    "chi2_cumulative_distribution",
    "chi2_log_probability_density",
    "chi2_probability_density",
    "chi2_quantile",
    "chi2_survival",
    # F distribution
    "f_cumulative_distribution",
    "f_log_probability_density",
    "f_probability_density",
    "f_quantile",
    "f_survival",
    # Gamma distribution
    "gamma_cumulative_distribution",
    "gamma_log_probability_density",
    "gamma_probability_density",
    "gamma_quantile",
    "gamma_survival",
    # Normal distribution
    "normal_cumulative_distribution",
    "normal_log_probability_density",
    "normal_probability_density",
    "normal_quantile",
    "normal_survival",
    # Poisson distribution
    "poisson_cumulative_distribution",
    "poisson_log_probability_mass",
    "poisson_probability_mass",
    "poisson_survival",
]
