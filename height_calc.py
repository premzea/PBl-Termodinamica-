import numpy as np
a=np.pi*(0.07)**2 #m^2
g = 9.8 #m/s^2
Cp= 0.5 #cosas
m= 0.141 #kg
vt = (2*m*g/(Cp*a))**0.5 #m/s
t = 160 #s
tao = vt/g
def h(t):
    print("height: " + str(np.log((np.arccosh(t/tao)))*vt*tao) + " m")

h(t)