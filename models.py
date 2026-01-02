import numpy as np

# Modelo XYZ (Analítico) [cite: 64]
def formula_X(t, beta, X0=99):
    return X0 * np.exp(-beta * t)

def formula_Y(t, beta, gamma, X0=99, Y0=5):
    C1 = (X0 * beta) / (gamma - beta)
    return C1 * np.exp(-beta * t) + (Y0 - C1) * np.exp(-gamma * t)

# Modelo SXYZ (Numérico con dinámica vital) [cite: 54, 55, 56]
def modelo_SXYZ(y, t, alpha, beta, gamma, r, k):
    S, X, Y, N = y
    dN_dt = r * N * (1 - N / k)  # Crecimiento logístico [cite: 51]
    dS_dt = dN_dt - alpha * S * (X + Y)
    dX_dt = alpha * S * (X + Y) - beta * X
    dY_dt = beta * X - gamma * Y
    return [dS_dt, dX_dt, dY_dt, dN_dt]

# Crecimiento poblacional explícito
def N_t(t, r, k, N0=10244247):
    term = ((k - N0) / N0) * np.exp(-r * t)
    return k / (1 + term)

