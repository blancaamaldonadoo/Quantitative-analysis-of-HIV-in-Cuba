import matplotlib.pyplot as plt
from scipy.integrate import odeint
import data_loader as dl
import models as mdl
import utils as utl
import numpy as np

# 1. Preparación de datos
time = np.arange(0, 15)
y0 = dl.get_initial_conditions()

# 2. Estimación de parámetros XYZ [cite: 66]
beta_xyz = utl.estimate_beta(time, dl.hiv_cases)
gamma_xyz = utl.estimate_gamma(time, dl.aids_cases, beta_xyz)

# 3. Simulación SXYZ [cite: 80]
k = 12000000         
r = 0.073            
alpha = 2.3e-10 
t_sim = np.linspace(0, 14, 100)
sol_sxyz = odeint(mdl.modelo_SXYZ, y0, t_sim, args=(alpha, 0.18, 0.011, r, k))

# 4. Gráfica Final (SXYZ) [cite: 82]
plt.figure(figsize=(10,6))
plt.plot(dl.years, dl.hiv_cases, 'bo', label="HIV Real")
plt.plot(dl.years, dl.aids_cases, 'ro', label="AIDS Real")
plt.plot(t_sim + 1986, sol_sxyz[:, 1], 'b-', label="HIV Model (X)")
plt.plot(t_sim + 1986, sol_sxyz[:, 2], 'r-', label="AIDS Model (Y)")
plt.title("SXYZ Model Performance")
plt.legend()
plt.grid()
plt.savefig('SXYZ_Final_Result.png')
print("Proceso completado. Gráfica guardada.")
