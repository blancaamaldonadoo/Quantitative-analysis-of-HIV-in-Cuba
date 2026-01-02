import numpy as np
import skimage
import skimage.io
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d

#datos por cada 100.000 habitantes

year= [1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000]


hiv_cases= [99,75,93,121,140,183,175,102,122,124,234,364,362,493,545]
aids_cases=[5,11,14,13,28,37,71,82,102,116,99,121,150,176,251]
death_aids= [2,4,6,5,23,17,32,59,62,80,92,99,98,122,142]

tabla = pd.DataFrame({
    "Year": year,
    "HIV Cases": hiv_cases,
    "AIDS Cases": aids_cases,
    "Deaths by AIDS": death_aids
})

print(tabla)

plt.figure(figsize=(10,6))
plt.plot(tabla["Year"], tabla["HIV Cases"], marker='o', label= "HIV Cases")
plt.plot(tabla["Year"], tabla["AIDS Cases"], marker='s', label="AIDS Cases")
plt.plot(tabla["Year"], tabla["Deaths by AIDS"], marker='^', label="Deaths by AIDS")

plt.title("Evolution of HIV and AIDS cases in Cuba (1986-2000)")
plt.xlabel("Year")
plt.ylabel("Number of cases")
plt.title("Evolution of HIV and AIDS Cases in Cuba (1986–2000)")
plt.grid(True, linestyle="--")
plt.legend()
plt.show()
plt.savefig('CUBA_evolution.png')


 #TOTAL POPULATION:
data_N=[
    10244247,  # 1986
    10335342,  # 1987
    10432585,  # 1988
    10533243,  # 1989
    10631799,  # 1990
    10717640,  # 1991
    10785801,  # 1992
    10841031,  # 1993
    10886021,  # 1994
    10925362,  # 1995
    10962010,  # 1996
    10997934,  # 1997
    11034712,  # 1998
    11072230,  # 1999
    11109109   # 2000
]


data_X = np.array([99, 75, 93, 121, 140, 183, 175, 182, 122, 124, 234, 364, 362, 393, 545])
data_Y = np.array([5, 11, 14, 13, 28, 37, 71, 82, 102, 116, 99, 131, 150, 176, 251])
data_Z = np.array([2, 4, 6, 5, 23, 17, 32, 59, 62, 80, 93, 99, 122, 142, 142])
data_S= data_N-(data_X + data_Y + data_Z)

"""
mean_X_to_Y = 8.5 #years
beta = 1 / mean_X_to_Y
mean_Y_to_Z = 1.34 #years
gamma = 1 / mean_Y_to_Z
"""

data_years=np.arange(1986,2001)
time= data_years - 1986
#print(time)


#---------- MODELO XYZ--------------
import jax.numpy as jnp
from jax import grad, jit
from scipy.optimize import minimize


def function_B(t):
    return (np.log(99/data_X[t]))/t

beta_estimation=[]
for t in time[1:]:
    beta_estimation.append(function_B(t))

sum_betas= sum(beta_estimation)
beta= sum_betas/len(beta_estimation)


print(beta_estimation)
print("\n",beta)


from scipy.optimize import fsolve
from scipy.optimize import newton

def function_gamma(g, time):
    term1 = -5.2371 * np.exp(0.0529 * time) / (g + 0.0529)
    term2 = (5 * g + 4.9726) * np.exp(-g * time) / (g - 0.0529)
    y_t = data_Y[time]
    return term1 + term2 - y_t

#hacer avg 
gamma_estimation=[]

for t in range(1,15):
    try:
        sol = newton(function_gamma, x0= 0.11448453235792273, args=(t,))
        gamma_estimation.append(sol)
        
    except RuntimeError:
            print(f"No convergió en el año {t}")
    

gamma_sum = sum(gamma_estimation)
gamma = (gamma_sum)/len(gamma_estimation)
print("\nGamma: ", gamma)

#print(f"La solución encontrada para gamma es: {gamma}")


# 1. Valores iniciales (Solo se usan los datos reales para el primer punto)
X_actual = data_X[0]
Y0 = data_Y[0]

"""
# 2. Función XYZ: Ahora usa variables genéricas, NO data_X[t]
def function_XYZ(beta, gamma, x_temp, y_temp):
    # El cambio depende de los valores actuales que lleva el modelo
    dX_dt = -beta * x_temp
    dY_dt = beta * x_temp - gamma * y_temp
    return [dX_dt, dY_dt]
"""
    
def formula_Y(t):
    C1 = (99.0 * beta) / (gamma - beta)
    term2 = np.exp(-beta * t)
    term3 = (Y0 - C1) * np.exp(-gamma * t)
    return C1 * term2 + term3


def formula_X(t):
    C1 = 99 * (np.exp (-beta * t))
    return C1


time= np.arange(0,15)
solution_X = []
solution_Y = []


# 4. Gráfica
plt.figure(figsize=(10,6))
plt.plot(data_years, data_X, 'o', label="VIH Real Cases", color='blue')
plt.plot(data_years, data_Y, 'o', label="AIDS Real Cases", color='red')

# Graficamos las soluciones (deben tener el mismo largo que data_years)
plt.plot(data_years, formula_X(time), linestyle='-', label="HIV Model", color='blue')
plt.plot(data_years, formula_Y(time) , linestyle='-', label="AIDS Model", color='red')
plt.xlabel("Year")
plt.ylabel("Number of cases ")

plt.grid()
plt.legend()
plt.show()
plt.savefig('Model_XYZ.png')



# MODELO SXYZ

def N_t(t):
    term = ((k - N0)/N0) * np.exp(-r*t)
    return k / (1 + term)


def modelo_SXYZ(y, t, alpha, beta, gamma, r, k):
    
    S, X, Y, N = y
    dN_dt = r * N * (1 - N / k)
    dS_dt = dN_dt - alpha * S * (X + Y)
    dX_dt = alpha * S * (X + Y) - beta * X
    dY_dt = beta * X - gamma * Y
    
    return [dS_dt, dX_dt, dY_dt, dN_dt]
    

# Parámetros sugeridos para empezar a probar:
"""k = 11109109         # Capacidad de carga (un poco más que la población del 2000)
r = 0.01             # Crecimiento poblacional del 1% aprox.
alpha = 2.3e-10      # Valor muy pequeño porque S es muy grande (10 millones)
beta_SXYZ = 0.18     # Tasa de progresión HIV -> AIDS
gamma_SXYZ = 0.011   # Tasa de salida (recuperación o fallecimiento)
"""
k = 12000000         
r = 0.073            
alpha = 2.3e-10      
beta_SXYZ = 0.18    
gamma_SXYZ = 0.011 

# Condiciones iniciales (t=0, año 1986)
S0 = data_S[0]
X0 = data_X[0]
Y0 = data_Y[0]
N0 = data_N[0]
y0 = [S0, X0, Y0, N0]

# Vector de tiempo
t = np.linspace(0, 14, 100) # De 1986 a 2000

"""
N0 = data_N[0]
alpha_SXYZ = 
beta_SXYZ = beta
gamma_SXYZ = 
k = 
r = """

# Resolver el sistema
solucion = odeint(modelo_SXYZ, y0, t, args=(alpha, beta_SXYZ, gamma_SXYZ, r, k))

# Extraer resultados
S_model = solucion[:, 0]
X_model = solucion[:, 1]
Y_model = solucion[:, 2]

# Graficar
plt.figure(figsize=(10,6))
plt.plot(data_years, data_X, 'bo', label="HIV Real")

plt.plot(data_years, data_Y, 'ro', label="AIDS Real")
plt.plot(t + 1986, X_model, 'b-', label="HIV Model (X)")
plt.plot(t + 1986, Y_model, 'r-', label="AIDS Model (Y)")
plt.title("SXYZ Model vs Real Cases")
plt.xlabel("Year")
plt.ylabel("Number of cases")
plt.legend()
plt.grid()
plt.show()
plt.savefig('Model_SXYZ.png')


def function_r(t): 
    term1 = (k/data_N[t]) -1
    term2= (k/N0) -1
    return  -np.log(term1/term2)

r_estimation=[]

for t in range (14):
    r_estimation.append(function_r(t))

sum_r= sum(r_estimation)
r= sum_r/len(r_estimation)


print(r_estimation)
print("\n",r)


k= 11109109
r= 0.18

k=12000000
r= 0.073


def N_t(t):
    term = ((k - N0)/N0) * np.exp(-r*t)
    return k / (1 + term)


def modelo_SXYZ(y, t, alpha, beta, gamma, r, k):
    
    S, X, Y, N = y
    dN_dt = r * N * (1 - N / k)
    dS_dt = dN_dt - alpha * S * (X + Y)
    dX_dt = alpha * S * (X + Y) - beta * X
    dY_dt = beta * X - gamma * Y
    
    return [dS_dt, dX_dt, dY_dt, dN_dt]
    

# Parámetros sugeridos para empezar a probar:
"""k = 11109109         # Capacidad de carga (un poco más que la población del 2000)
r = 0.01             # Crecimiento poblacional del 1% aprox.
alpha = 2.3e-10      # Valor muy pequeño porque S es muy grande (10 millones)
beta_SXYZ = 0.18     # Tasa de progresión HIV -> AIDS
gamma_SXYZ = 0.011   # Tasa de salida (recuperación o fallecimiento)
"""
k = 12000000         
r = 0.073            
alpha = 2.3e-10 
beta_SXYZ = 0.18    
gamma_SXYZ = 0.011 

# Condiciones iniciales (t=0, año 1986)
S0 = data_S[0]
X0 = data_X[0]
Y0 = data_Y[0]
N0 = data_N[0]
y0 = [S0, X0, Y0, N0]

# Vector de tiempo
t = np.linspace(0, 14, 100) # De 1986 a 2000

"""
N0 = data_N[0]
alpha_SXYZ = 
beta_SXYZ = beta
gamma_SXYZ = 
k = 
r = """

# Resolver el sistema
solucion = odeint(modelo_SXYZ, y0, t, args=(alpha, beta_SXYZ, gamma_SXYZ, r, k))

# Extraer resultados
S_model = solucion[:, 0]
X_model = solucion[:, 1]
Y_model = solucion[:, 2]

# Graficar
plt.figure(figsize=(10,6))
plt.plot(data_years, data_N, 'go', label="Real Population")
plt.plot(data_years, N_t(time), 'g-', label="Population Model adjusted")
r=0.456
plt.plot(data_years, N_t(time), 'y-', label="Population Model for average r=0.456")

r= 0.282
plt.plot(data_years, N_t(time), 'y--', label="Population Model for r=0.32")

r= 0.15
plt.plot(data_years, N_t(time), 'y:', label="Population Model for r=0.2")

plt.title("Population Model (r=0.073 | K=12.000.000)")
plt.xlabel("Year")
plt.ylabel("Number of people (10 millions)")
plt.legend()
plt.show()
plt.savefig('Population_model.png')
