import numpy as np
def h(t, tao, vt):
    print("height: " + str(np.log((np.cosh(t/tao)))*vt*tao) + " m")


def main():    
    a=4*np.pi*(0.07/2)**2 #m^2 (area de un cilidro)
    g = 9.8 #m/s^2
    Cp= 0.5 #cosas
    m= 0.141 #kg
    vt = (2*m*g/(Cp*a))**0.5 #m/s
    tao = vt/g
    




    t = 4.25 #s
    h(t, tao, vt)


main()

    

