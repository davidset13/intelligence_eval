from scipy.stats import norm
import numpy as np

def min_sample_size_safe_mle_wald(distribution: str, N: int, eps: float = 0.01, alpha: float = 0.05) -> int:
    if distribution not in ["bernoulli"]:
        raise ValueError(f"Distribution {distribution} not supported")

    # X1, X2, ..., Xn ~ Ber(p)
    # f(Xi) = p^Xi * (1-p)^(1-Xi)
    # Derivation leads to: p_hat = sum(Xi) / n
    # Variance of p_hat = p_hat * (1-p_hat) / n
    # Maximized at p = 0.5, or p(1-p) = 0.25

    # Our confidence interval must be 2-tailed
    # P( -eps/sig < Z < eps/sig) = 1 - alpha -> P(Z < eps/sig) = 1 - alpha/2
    confidence_interval: float = 1 - alpha/2
    z_score: float = norm.ppf(confidence_interval)

    for n in range(1, N):
        fpc = (N - n) / (N - 1)
        se = np.sqrt(0.25 / n * fpc)
        if se < eps / z_score:
            return n
    raise ValueError("No sufficient sample size found")

def Wald_CI(distribution: str, N: int, n: int, p_hat: float, alpha: float = 0.05) -> tuple[float, float]:
    if distribution not in ["bernoulli"]:
        raise ValueError(f"Distribution {distribution} not supported")

    standard_error: float = np.sqrt((p_hat * (1 - p_hat) / n) * ((N - n) / (N - 1)))
    confidence_interval: float = 1 - alpha/2
    z_score: float = norm.ppf(confidence_interval)
    margin_of_error: float = z_score * standard_error
    return (float(round(p_hat - margin_of_error, 4)), float(round(p_hat + margin_of_error, 4)))


