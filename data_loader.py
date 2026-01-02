import numpy as np
import pandas as pd

# Datos históricos (1986-2000)
years = np.arange(1986, 2001)
hiv_cases = np.array([99, 75, 93, 121, 140, 183, 175, 102, 122, 124, 234, 364, 362, 493, 545])
aids_cases = np.array([5, 11, 14, 13, 28, 37, 71, 82, 102, 116, 99, 121, 150, 176, 251])
death_aids = np.array([2, 4, 6, 5, 23, 17, 32, 59, 62, 80, 92, 99, 122, 142, 142])

# Datos poblacionales N
data_N = np.array([10244247, 10335342, 10432585, 10533243, 10631799, 
                   10717640, 10785801, 10841031, 10886021, 10925362, 
                   10962010, 10997934, 11034712, 11072230, 11109109])

def get_initial_conditions():
    # S = N - (X + Y + Z)
    X0 = hiv_cases[0]
    Y0 = aids_cases[0]
    Z0 = death_aids[0]
    N0 = data_N[0]
    S0 = N0 - (X0 + Y0 + Z0)
    return [S0, X0, Y0, N0]
    