import matplotlib.pyplot as plt
import numpy as np
import functions
import matplotlib.colors
import time
import sys
import constants
import akashiwo_dist

# Set the backend
matplotlib.use('TkAgg')

# ------- Other constants -----------

width = 64
length = 64
max_iter = 5000
omega = 1
lid_vel = 2

# ----- Steady state check parameters ------

window = 100
threshold = 1e-5
consecutive = 5

boundary = (True, True, False, True) 

# Initialize rho and u
rho = np.ones((width, length))
u = np.zeros((width, length, 2))

# Initialize density function w.r.t. rho and u 
f = functions.equilibrium(rho, u)

# calculate corresponding Reynolds number
Re = functions.calculate_re(omega=omega, length=length, lid_vel=lid_vel)

# create a directory to save the figures
import os
if not os.path.exists('LB_plankton_figs'):
    os.makedirs('LB_plankton_figs')

steady_state_iteration = None
u_at_point = np.zeros(max_iter)

for t in range(max_iter):
    # Streaming step
    f = functions.streaming(f, width, length, boundary, lid_vel) 

    # Collision step
    f, rho, u = functions.collision(f, omega)
    
    # Check steady state at point (32, 32)
    velocity_at_32_32 = np.linalg.norm(u[32, 32, :])
    u_at_point[t] = velocity_at_32_32
    
    if t >= window and steady_state_iteration is None:
        if functions.check_steady_state(u_at_point[:t+1], threshold=threshold, window=window, consecutive=consecutive):
            steady_state_iteration = t
            print(f"Steady state reached at iteration {t}. Velocity at (32, 32): {velocity_at_32_32}.")
            steady_state_velocity = u.copy()

            break

    # Save figure every 250 iterations
    if (t + 1) % 250 == 0:
        fig = plt.figure(figsize=(12, 10))
        plt.streamplot(np.arange(width), np.arange(length), u[:,:, 0].T, u[:,:, 1].T, color='r')

        plt.contourf(np.arange(width), np.arange(length), np.linalg.norm(u, axis=2).T, cmap='turbo',levels=100)
        plt.colorbar(label='Mag. of U (mm/s)')

        plt.title('Velocity Field, iter = %d' % (t+1))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(f"LB_plankton_figs/figat_{t+1}.png")
        plt.close()  # Add this line to close the figure after saving

# Plot fluid speed at point (64, 64) over time
plt.figure(figsize=(10, 6))
plt.plot(np.arange(max_iter), u_at_point, color='black')
plt.title('Fluid Speed @(32,32)')
plt.xlabel('Iteration')
plt.ylabel('Fluid parcel velocity (mm/s)')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"LB_plankton_figs/U_Parcel.png")
plt.show()

# Plot velocity field at steady state
if steady_state_iteration is not None:
    fig = plt.figure(figsize=(12, 10))
    plt.streamplot(np.arange(width), np.arange(length), u[:,:, 0].T, u[:,:, 1].T, color='r')

    plt.contourf(np.arange(width), np.arange(length), np.linalg.norm(u, axis=2).T, cmap='turbo',levels=100)
    x0, vc, ini_velocities, p0 = akashiwo_dist.ini_swimspeed_cells(width=width, length=length, num_cells=1000)
    plt.quiver(x0[:, 0], x0[:, 1], ini_velocities[:,0]/1000, ini_velocities[:,1]/1000, color='lime', scale=10) # convert to mm/s    plt.colorbar(label='|u|')

    plt.title(f'Initialization of H.Akashiwo at steady state (iter: {t}), Re: {Re}') # convert to mm/s   
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig("LB_plankton_figs/Velocity_Field_and_Particle_Positions_Steady_State.png")    
    plt.show()
    print(f"Steady state reached at iteration {t}. Phytoplankton cells initialize")
else:
    print("Steady state not reached. Phytoplankton cells do not initialize. Adjust threshold or simulate for longer times")

def calculate_vorticity(u):

    du_dy, du_dx = np.gradient(steady_state_velocity[:,:,0])
    dv_dy, dv_dx = np.gradient(steady_state_velocity[:,:,1])
    fluid_vorticity = dv_dx - du_dy
    return fluid_vorticity

fluid_vorticity = calculate_vorticity(steady_state_velocity)

fig = plt.figure(figsize=(10,10))
plt.contourf(np.arange(width), np.arange(length), fluid_vorticity.T, cmap='turbo',levels=100)
plt.colorbar(label='Vorticity $\omega$')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Vorticity at steady state, Re: {Re}')
plt.savefig("LB_plankton_figs/Vorticity_Steady_State.png")
plt.show()

# ===== ONTO SOLVING THE ORDINARY DIFFERENTIAL EQUATIONS USING RK4 =======
# Function to calculate dp/dt
def calculate_dp_dt(p, omega_star, k, Psi):
    dp_dt = 1 / (2 * Psi) * (k - (k * p)) * p  + 0.5 * np.cross(omega_star, p)
    return dp_dt

# Function to calculate dX/dt
def calculate_dX_dt(X, p, u, Phi):
    dX_dt = Phi * p + u
    return dX_dt

# RK4 integration
def RK4_step(X, p, u, omega_star, k, Phi, Psi, dt):
    k1_p = dt * calculate_dp_dt(p, omega_star, k, Psi)
    k1_X = dt * calculate_dX_dt(X, p, u, Phi)
    
    k2_p = dt * calculate_dp_dt(p + 0.5 * k1_p, omega_star, k, Psi)
    k2_X = dt * calculate_dX_dt(X + 0.5 * k1_X, p + 0.5 * k1_p, u, Phi)
    
    k3_p = dt * calculate_dp_dt(p + 0.5 * k2_p, omega_star, k, Psi)
    k3_X = dt * calculate_dX_dt(X + 0.5 * k2_X, p + 0.5 * k2_p, u, Phi)
    
    k4_p = dt * calculate_dp_dt(p + k3_p, omega_star, k, Psi)
    k4_X = dt * calculate_dX_dt(X + k3_X, p + k3_p, u, Phi)
    
    dp_dt = (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6
    dX_dt = (k1_X + 2 * k2_X + 2 * k3_X + k4_X) / 6
    
    return dp_dt, dX_dt

# params
Psi = 2
Phi = np.power(70.0, -3)

# simulation params
width = 64
length = 64
num_cells = 1000
t_max = 10.0  # tmax
dt = 0.1  # step size
num_steps = int(t_max / dt)  # time steps

# initial conditions
x0, vc, ini_velocities, p0 = akashiwo_dist.ini_swimspeed_cells(width, length, num_cells)


# y-direction unit vector with shape (1000, 2)
k = np.zeros((num_cells, 2))
k[:, 1] = 1  

# arrays for cell positions and orientations
p_values = np.zeros((num_steps, num_cells, 2))
X_values = np.zeros((num_steps, num_cells, 2))
p_values[0] = p0
X_values[0] = x0

# Integration using RK4
for i in range(num_steps):
    # calculate fluid vorticity at current cell positions
    omega_star = np.array([fluid_vorticity[int(x[1]), int(x[0])] for x in X_values[i]])
    omega_star = omega_star.reshape(num_cells,1, 2)  # Reshape to have a 2nd dimension with 1 item (I DONT UNDERSTAND THIS PART)

    print("Shape of omega_star after reshaping:", omega_star.shape)

    # here we calculate derivatives using RK4 with current vorticity
    dp_dt, dX_dt = RK4_step(X_values[i], p_values[i], np.zeros((num_cells, 2)), omega_star, k, Phi, Psi, dt)
    
    # update p and X
    p_values[i + 1] = p_values[i] + dp_dt
    X_values[i + 1] = X_values[i] + dX_dt

