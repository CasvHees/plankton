import numpy as np 
import functions 
import constants 
import exc1 
import akashiwo_dist

def calculate_dp_dt(p, omega_star, k, Psi=2):
    dp_dt = 1/(2 * Psi) * (k - np.dot(k, p) * p) + 0.5 * np.cross(omega_star, p) 
    return dp_dt 

def calculate_dX_dt(X, p, u, Phi=np.power(70,-3)):
    dX_dt = Phi * p + u 
    return dX_dt 

# Define initial positions and velocities for all particles
x0, vc, ini_velocities, p0 = akashiwo_dist.ini_swimspeed_cells(width=constants.width, length=constants.length, num_cells=30)

# Define the unit vector in the y-direction
k = np.array([0, 1])

# Initialize arrays to store fluid velocity magnitudes at point (64, 64) and the velocities at all grid points
u_at_point = np.zeros(constants.max_iter)
u_allpoints_steady = np.zeros((100, constants.width(), constants.length()))

# Loop over time steps
for t in range(constants.max_iter):
    # Streaming step
    f = functions.streaming(f, constants.width(), constants.length(), constants.boundary, constants.lid_vel)     
    # Collision step
    f, rho, u = functions.collision(f, constants.omega)
    
    # Calculate fluid velocity magnitude at point (64, 64)
    u_at_point[t] = np.linalg.norm(u[64, 64, :])

    # Check for steady-state every 500 iterations after the initial transient
    if t >= constants.steady_state_window:
        if functions.check_steady_state(u_at_point[:t+1], threshold=constants.steady_state_threshold, window=constants.steady_state_window, consecutive=constants.steady_state_cons):
            print("Steady state reached at time step:", t)
            # Measure velocities for all grid points for the next 100 iterations
            u_allpoints_steady = np.array([u_t for u_t in u[t:t+100]])
            break
print(u_allpoints_steady)
