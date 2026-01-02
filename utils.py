import numpy as np
from scipy.optimize import newton

def estimate_beta(time, hiv_data):
    beta_est = [np.log(hiv_data[0] / hiv_data[t]) / t for t in time[1:]]
    return sum(beta_est) / len(beta_est)

def function_gamma(g, t, y_real, beta):
    # Ecuación para encontrar gamma numéricamente [cite: 64]
    term1 = -5.2371 * np.exp(0.0529 * t) / (g + 0.0529)
    term2 = (5 * g + 4.9726) * np.exp(-g * t) / (g - 0.0529)
    return term1 + term2 - y_real[t]

def estimate_gamma(time, y_real, beta_val):
    gamma_est = []
    for t in range(1, 15):
        try:
            sol = newton(function_gamma, x0=0.11, args=(t, y_real, beta_val))
            gamma_est.append(sol)
        except: continue
    return sum(gamma_est) / len(gamma_est)