import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def solve_velocity_profile(
    m,             # projectile mass
    A,             # barrel cross-sectional area
    P0,            # initial reservoir pressure (absolute)
    Patm,          # atmospheric pressure (absolute)
    V0,            # initial reservoir volume (connected to barrel)
    gamma=1.4,     # adiabatic index for air
    f=0.0,         # sliding friction
    L=1.0,         # barrel length
    Nx=2000        # number of x-points
):
    """
    Numerically solve for v(x) using the ODE:
        dv/dx = (A*(P(x)-Patm) - f) / (m*v)
    with adiabatic P(x).
    """

    # x grid
    x = np.linspace(0, L, Nx)

    # Small initial velocity (cannot start at exactly v=0 because of dv/dx ~ 1/v)
    v0 = 1e-6

    def adiabatic_pressure(x):
        return P0 * (V0 / (V0 + A*x))**gamma
  
    def dv_dx(v, x):
        # pressure behind projectile
        P = adiabatic_pressure(x)

        # net force
        F = A*(P - Patm) - f

        # ODE from the paper
        return F / (m * v)

    # solve ODE using odeint
    v = odeint(dv_dx, v0, x).flatten()

    return x, v


# ----------------------------------------------------------------------
# Example usage with representative parameters (change to your setup)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    m    = 0.05          # 50 g projectile
    A    = 3.14e-4       # barrel area for ~2 cm diameter
    P0   = 600000.0      # 600 kPa absolute (≈ 85 psi above atm)
    Patm = 101325.0      # atmospheric pressure
    V0   = 0.001         # 1 liter reservoir = 0.001 m^3
    gamma = 1.4
    f = 0.0
    L = 1.0

    x, v = solve_velocity_profile(m, A, P0, Patm, V0, gamma, f, L)

    plt.plot(x, v)
    plt.xlabel("x (m)")
    plt.ylabel("v(x) (m/s)")
    plt.title("Projectile velocity vs position in barrel")
    plt.grid(True)
    plt.show()
