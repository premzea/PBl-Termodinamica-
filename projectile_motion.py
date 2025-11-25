import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import bisect

h = 60 #m
# """Conditions"""
R = 8.3145 #Pam**3/molK
# m = 0.01940        # Masa del proyectil (kg) [cite: 160]
m = (67 + 52)/1000 #kg
g = 9.8 #m/s**2
P_atm = 101325*0.89  # Presión atmosférica (Pa)


"""Van der Waals"""
Tc = 132.63 #K
Pc = 37.858 #bar
a = (27*R**2*Tc**2)/(64*Pc*10**5)
b = (1*R*Tc)/(8*Pc*10**5)

def VvdW (T: float, P, b, R, a) -> np.complex128:
  """Van der Waals Molar Volume Solver
  Args:      
      T (float): Temperature [K]
      P (float): Pressure [Pa]
      R (float): Ideal Gas Constant [Pa m**3 / mol K]
      b (float): Van der Waals constant b [m**3/mol]
      a (float): Van der Waals constant a [Pa m**6 / mol**2]
  Returns:      
      v (np.complex128): Molar Volume roots [m**3/mol]
  """
  #P [Pa], T [K]
  coef = [P, -P*b - R*T, a, -a*b]
  v = np.roots(coef)
  return  v[np.isclose(v.imag, 0)][0] #[m**3/mol]


def PVdW(T: float, N: float, V: float, R: float, b: float, a: float) -> float:
  """Van der Waals Equation of State
  
  Args:      
      T (float): Temperature [K]
      v (float): Molar Volume [m**3/mol]
      R (float): Ideal Gas Constant [Pa m**3 / mol K]
      b (float): Van der Waals constant b [m**3/mol]
      a (float): Van der Waals constant a [Pa m**6 / mol**2]
  Returns:      
      P (float): Pressure [Pa]
  """
  return R*T/((V/N)-b) - a/((V/N)**2)

"""Kinematic Viscocity"""
T = 298.15 #K
#asumiremos T cst y ambiente o la medimos ese dia
Mw= 29/1000 # kg/mol
#L: longitud caracteristica = D
#kinematic viscosity
# To = 518.7 #R
# mo = 3.62 * 10**-7 #lbs/ft**2
# vs = VvdW(T, P_atm, b, R, a)
# kmu = (mo * (((T*9/5)/To)**(1.5)*(To + 198.72)/((T*9/5) + 198.72))) * (6894.76) * (vs / Mw) #m**2/s

# D = 7/100 #m
# dragCoefficients: list[float] = []



# def dragCoefficient(V, D, kmu, dragCoefficients: list[float]):
#   """Drag Coefficient Calculator
#   Args:      
#       V (float): Velocity [m/s]
#       D (float): Diameter [m]
#       kmu (float): Kinematic Viscosity [m**2/s]
#       dragCoefficients (list): List to append drag coefficients
#   Returns:
#       dragCoefficients (list): Updated list of drag coefficients      
#     """
#   Re = D*V/kmu #adimensional
#   if Re < 10**3:
#     dragCoefficients.append((12*Re**-5))  #en que unidades?? (Edwards et.al 2000)
#   elif Re < 2* 10**5:
#     dragCoefficients.append(((24/Re)*(1+0.15*Re**(0.687)) + (0.42/(1+42500/Re**1.16))))
#   elif Re < 10**6:
#     dragCoefficients.append(((24/Re)*(2.6*(Re/5))/(1+(Re/5)**1.52) + (0.411*(Re/263000)**-7.94)/(1+(Re/263000)**-8) + (Re**0.8/461000)))
#   return dragCoefficients


# def terminalVelocity(dragCoefficients: list[float], m, g, vs, D):
#   '''  Terminal Velocity Calculator
#   Args:
#       dragCoefficients (list): List of drag coefficients
#       m (float): Mass of the projectile [kg]
#       g (float): Gravitational acceleration [m/s**2]
#       vs (float): Specific volume [m**3/mol]
#       D (float): Diameter [m]
    
#     Returns:
#       vt (float): Terminal Velocity [m/s]'''
# #   Cp = sum(dragCoefficients)/len(dragCoefficients) #con este no, no se porque, pero bueno, lueo lo resuelvo. 
#   Cp = 0.5 #funciona con este, da resultados logicos
#   return (2*m*g*vs/(Cp*Mw*(D/2)**2*np.pi))**0.5

# # The function
# def velocity(h, D, kmu, dragCoefficients: list[float], g, vs, m):
#     vo = 20 #m/s
#     dragCoefficient(vo, D, kmu, dragCoefficients)
#     vt = terminalVelocity(dragCoefficients, m, g, vs, D)
#     while abs((h - (-1*(vt**2)/g)*np.log(np.cos(np.arctan(vo/vt))))) > 5:
#         vo += 1
#         dragCoefficient(vo, D, kmu, dragCoefficients)
#         vt = terminalVelocity(dragCoefficients, m, g, vs, D)
#     while abs((h - (-1*(vt**2)/g)*np.log(np.cos(np.arctan(vo/vt))))) > 1:
#         vo += 0.1
#         dragCoefficient(vo, D, kmu, dragCoefficients)
#         vt = terminalVelocity(dragCoefficients, m, g, vs, D)
#     print(str(round(vo,2)) + " m/s")

# velocity(h, D, kmu, dragCoefficients, g, vs, m)


# ############################################################################################################################################################################################################################################################################
# #Area del Paracaidas

# vc = 4 # velocidad en la caida[m/s]
# Ap = (2*m*g*vs)/(Mw*0.5*vc**2) #m^2
# r = ((Ap/np.pi)**0.5) #m
# print('radius = ' + str(r) + ' m')

# #meta: poder usar nuestro propio coeficiente de arrastre

##############################################################################################################################################################################################################################


# --- 1. Definición de Constantes y Parámetros del Cañón ---

# Parámetros Universales y del Gas (Sistema Internacional: SI)
kB = 1.380649e-23  # Constante de Boltzmann (J/K)
P_atm = 101325*0.89     # Presión atmosférica (Pa)
T = 298.15            # Temperatura del depósito (K), asumida constante [cite: 100] **
Gg = 1             # Gravedad específica del aire [cite: 100]
Z = 1              # Factor de compresibilidad [cite: 100]
B = 3.11e19        # Constante de ingeniería B (unidades ajustadas en SI) [cite: 102]

# Parámetros Físicos y Empíricos del Cañón de Rohrbach et al. (en SI)
m = (67+52)/1000        # Masa del proyectil (kg) [cite: 160] **
D = 7.62/100        # Diámetro del cañón (m) [cite: 109] **
A = np.pi*(D/2)**2 # Área transversal del cañón (m^2)
L = 46/100         # Longitud del cañón para la aceleración (m) [cite: 164] **
V0 = 0.0067      # Volumen del depósito (m^3) [cite: 108] **
f = 0              # Fricción (N), asumida despreciable [cite: 172] 
d = 0.1           # Distancia inicial del proyectil a la válvula (m), asumido. **

# Parámetros Empíricos de la Válvula Calibrada
r_max = 0.80       # Relación de presión crítica (adimensional) [cite: 170]
Cv = 480          # Coeficiente de flujo de la válvula (adimensional) [cite: 170] **

# --- 2. Funciones de Flujo (Q) y Presión ---

def calcular_flujo_Q(P, Pb, r_max, Cv, T, Gg, Z, B):
    """Calcula la tasa de flujo molecular Q (dN/dt) a través de la válvula."""
    
    # 1. Calcular la relación de presión instantánea (Ec. 1)
    if P <= Pb:
        r = 0.0  # El flujo se detiene o es negativo (que no se modela aquí)
    else:
        r = (P - Pb) / P 

    # 2. Determinar el régimen de flujo
    
    # Régimen Choked (Ahogado): r >= r_max (Ec. 8) [cite: 97]
    if r >= r_max:
        # P(t) >= P_b(t) / (1 - r_max)
        Q = (2/3) * B * P * Cv * np.sqrt(r_max / (Gg * T * Z))
    
    # Régimen Nonchoked (No Ahogado): r < r_max (Ec. 7) [cite: 94]
    else:
        # P(t) < P_b(t) / (1 - r_max)
        if r < 0: r = 0 # Asegura que la raíz cuadrada no sea negativa
        Q = B * P * Cv * (1 - r / (3 * r_max)) * np.sqrt(r / (Gg * T * Z))
        
    return Q

# --- 3. El Sistema de Ecuaciones Diferenciales (Forward Model) ---

def sistema_rohrbach(t, y):
    """
    Define el sistema de Ecuaciones Diferenciales Ordinarias (ODEs).
    y = [x, v, N, Nb]
    """
    x, v, N, Nb = y
    
    # 1. Calcular Presiones (Ec. 9 y 10)
    
    # Presión en el Tanque (Depósito)
    P = PVdW(T, N, V0, R, b, a) #[Pa] **

    # Volumen en el Cañón (Barril)
    Vb = A * (d + x)
    
    # Presión en el Cañón (Barril)
    Pb = PVdW(T, Nb, Vb, R, b, a) #[Pa] **
    
    # 2. Calcular Flujo Q (dN/dt)
    Q = calcular_flujo_Q(P, Pb, r_max, Cv, T, Gg, Z, B)
    
    # 3. Definir las Derivadas (dx/dt, dv/dt, dN/dt, dNb/dt)
    
    # Tasa de cambio de posición (dx/dt = v)
    dxdt = v
    
    # Tasa de cambio de velocidad (dv/dt = a) - Ecuación de Movimiento (Ec. 1)
    # F = A*Pb - A*Patm - f. Ya que f=0, F = A*(Pb - Patm)
    # Importante: el proyectil solo se acelera si Pb > Patm
    if Pb > P_atm:
        dvdt = (A * (Pb - P_atm) - f) / m
    else:
        dvdt = 0
        
    # Tasa de cambio de moléculas en el Tanque (Ec. 11)
    dndt = -Q
    
    # Tasa de cambio de moléculas en el Cañón (Ec. 12)
    dnbdt = Q

    return [dxdt, dvdt, dndt, dnbdt]

# --- 4. La Función de Simulación (SIMULACION(P0)) ---

def simulacion_rohrbach(P0):
    """
    Ejecuta el modelo de Rohrbach para una presión inicial P0 y retorna la velocidad de salida.
    """
    
    
    #N(0): Moléculas iniciales en el tanque (Ec. 9) **
    v0 = VvdW(T, P0, b, R, a) #[m**3/mol] **
    N0 = V0/(v0) 



# # Nb(0): Moléculas iniciales en el cañón (asumido a P_atm y volumen inicial A*d) **
    vb0 = VvdW(T, P_atm, b, R, a) #[m**3/mol] **
    Nb0 = (A * d) / vb0

    
    # Vector de estado inicial: y0 = [x0, v0, N0, Nb0]
    y0 = [0.0, 0.0, N0, Nb0]
    
    # Condición de finalización (cuando el proyectil alcanza L)
    def evento_salida(t, y):
        return y[0] - L # Se detiene cuando x - L = 0
    evento_salida.terminal = True
    evento_salida.direction = 1 # Evento solo se activa cuando x está creciendo

    # Resolver el sistema de ODEs (Método Runge-Kutta)
    # t_span es el intervalo de tiempo, solo necesita ser suficientemente grande
    sol = solve_ivp(
        sistema_rohrbach, 
        [0, 0.1], # Intervalo de tiempo: [t_inicio, t_final] (0.1s es conservador)
        y0, 
        method='RK45', 
        events=evento_salida, 
        dense_output=True,
        rtol=1e-6, # Tolerancia relativa
        atol=1e-9  # Tolerancia absoluta
    )

    # Si el evento de salida se activó, la velocidad es la última calculada
    if sol.y_events[0].size > 0:
        # El último valor de velocidad (y[1]) en el tiempo de salida
        v_salida = sol.y[1, -1]
        return v_salida
    else:
        # Si la simulación terminó sin salir del cañón (ej: baja P0), retorna 0
        return 0.0

# --- 5. El Problema Inverso (Búsqueda de P0) ---

def busqueda_presion_inversa(v_deseada, P_min, P_max, tolerancia_v=0.1):
    """
    Encuentra la presión inicial P0 (en kPa) necesaria para alcanzar v_deseada (m/s)
    utilizando el método de la Bisección.
    """
    
    # Definir la función de error (el objetivo es que sea cero)
    def funcion_error(P0):
        v_predicha = simulacion_rohrbach(P0)
        return v_predicha - v_deseada

    # Se verifica que el intervalo [P_min, P_max] encierra la raíz
    if funcion_error(P_min) * funcion_error(P_max) > 0:
        # Se necesita un intervalo que contenga la solución
        print(f"Error: La velocidad deseada {v_deseada:.1f} m/s no está contenida")
        print(f"en el rango de presiones iniciales [{P_min}, {P_max}] Pa.")
        print("Ajuste P_min o P_max para continuar.")
        return None
        
    # Usar el método de bisección para encontrar la raíz (P0)
    P0_requerida = bisect(funcion_error, P_min, P_max, xtol=0.01)

    return P0_requerida

# --- 6. Ejemplo de Uso ---

# EJEMPLO: Quiero que el proyectil alcance 85 m/s.
v_objetivo = 100 # m/s (aproximadamente 300 pies/s)

# Definir un rango de búsqueda (Presiones en kPa)
# Basado en la gráfica Fig. 3(a), la solución debería estar entre 400 y 600 kPa
P_min = 150 *10**3 #[Pa]
P_max = 800 *10**3 #[Pa]

print(f"Objetivo: Encontrar P0 para alcanzar {v_objetivo} m/s")

P0_calculada = busqueda_presion_inversa(v_objetivo, P_min, P_max)

if P0_calculada is not None:
    # Verificación final de la simulación
    v_verificacion = simulacion_rohrbach(P0_calculada)
    
    print("-" * 50)
    print("RESULTADOS DEL MODELO INVERSO DE ROHRBACH:")
    print(f"Velocidad Deseada: {v_objetivo:.2f} m/s")
    print(f"Presión Inicial Requerida (P0): {P0_calculada:.2f} Pa")
    print(f"Verificación de Velocidad (con P0 calculada): {v_verificacion:.2f} m/s")
    print("-" * 50)
