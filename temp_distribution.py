import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0          # Length of the rod
T = 0.5          # Total time
alpha = 1.0      # Thermal diffusivity

# Discretization
nx = 50                      # Number of spatial points
dx = L / (nx - 1)            # Spatial step size
dt = 0.4 * dx**2 / alpha     # Time step size (satisfies stability condition)
nt = int(T / dt)             # Number of time steps

# Grid
x = np.linspace(0, L, nx)

# Initial Condition
u_initial = np.sin(np.pi * x / L)
u = u_initial.copy()
u_new = np.zeros(nx)

# Time-stepping loop
for n in range(nt):
    for i in range(1, nx - 1):
        u_new[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
    # Boundary conditions (Dirichlet)
    u_new[0] = 0
    u_new[-1] = 0
    # Update for next step
    u, u_new = u_new, u

# Analytical solution for comparison
u_exact = np.exp(- (np.pi**2) * alpha * T / L**2) * np.sin(np.pi * x / L)

# Plotting
plt.plot(x, u_initial, label='Initial Condition')
plt.plot(x, u, label='Numerical Solution at T={}'.format(T))
plt.plot(x, u_exact, '--', label='Analytical Solution')
plt.xlabel('Position x')
plt.ylabel('Temperature u')
plt.legend()
plt.title('1D Heat Equation Solution')
plt.show()
