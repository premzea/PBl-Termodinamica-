import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import bisect

h = 60
# """Conditions"""
R = 8.3145 #Pam**3/molK
# m = 0.01940        # Masa del proyectil (kg) [cite: 160]
m = (67 + 52)/1000
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

"""Kinematic Viscocity"""
T = 298.15 #K
#asumiremos T cst y ambiente o la medimos ese dia
Mw= 29/1000 # kg/mol
#L: longitud caracteristica = D
#kinematic viscosity
To = 518.7 #R
mo = 3.62 * 10**-7 #lbs/ft**2
vs = VvdW(T, P_atm, b, R, a)
kmu = ((mo * (((T*9/5)/To)**(1.5)*(To + 198.72)/((T*9/5) + 198.72))) * (6894.76) * (vs / Mw)) #m**2/s

D = 7/100 #m
dragCoefficients: list[float] = []



def dragCoefficient(V, D, kmu, dragCoefficients: list[float]):
  """Drag Coefficient Calculator
  Args:      
      V (float): Velocity [m/s]
      D (float): Diameter [m]
      kmu (float): Kinematic Viscosity [m**2/s]
      dragCoefficients (list): List to append drag coefficients
  Returns:
      dragCoefficients (list): Updated list of drag coefficients      
    """
  Re = D*V/kmu #adimensional
  if Re < 10**3:
    dragCoefficients.append((12*Re**-5))  #en que unidades?? (Edwards et.al 2000)
  elif Re < 2* 10**5:
    dragCoefficients.append(((24/Re)*(1+0.15*Re**(0.687)) + (0.42/(1+42500/Re**1.16))))
  elif Re < 10**6:
    dragCoefficients.append(((24/Re)*(2.6*(Re/5))/(1+(Re/5)**1.52) + (0.411*(Re/263000)**-7.94)/(1+(Re/263000)**-8) + (Re**0.8/461000)))
  return dragCoefficients

def terminalVelocity(dragCoefficients: list[float], m, g, vs, D):
  Cp = sum(dragCoefficients)/len(dragCoefficients)
  return (2*m*g*vs/Cp*((D/2)**2*np.pi))**0.5

# The function
def velocity(h, D, kmu, dragCoefficients: list[float], g, vs, m):
  vo = 20 #m/s
  dragCoefficient(vo, D, kmu, dragCoefficients)
  vt = terminalVelocity(dragCoefficients, m, g, vs, D)
  while abs((h - ((-vt**2)/g)*np.log(np.cos(np.arctan(vo/vt))))) > 1:
    vo += 1
    dragCoefficient(vo, D, kmu, dragCoefficients)
    vt = terminalVelocity(dragCoefficients, m, g, vs, D)
  print(str(vo) + " m/s")

velocity(h, D, kmu, dragCoefficients, g, vs, m)
