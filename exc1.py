import matplotlib.pyplot as plt
import numpy as np
import functions
import akashiwo_dist
import matplotlib.colors
import sys
import time
from matplotlib import rc
import constants



rc('text', usetex=True)
# Set the backend
matplotlib.use('TkAgg')

# Initialize rho and u
rho = np.ones((constants.width, constants.length))
u = np.zeros((constants.width, constants.length, 2))

# Initialize density function w.r.t. rho and u 
f = functions.equilibrium(rho, u)

# Variables for checking if there is a steady state 
steady_state_threshold = 1e-5
steady_state_window = 100
steady_state_cons = 5 
steady_state_reached = False 

# calculate corresponding Reynolds number
Re = functions.calculate_re(omega=constants.omega, length=constants.length, lid_vel=constants.lid_vel)
# create a directory to save the figures
import os
if not os.path.exists('LB_plankton_figs'):
    os.makedirs('LB_plankton_figs')

t_ini = 0 
start_time = time.time()

u_at_point = np.zeros(constants.max_iter)

for t in range(constants.max_iter):
    # Streaming step
    f = functions.streaming(f, constants.width, constants.length, constants.boundary, constants.lid_vel)     
    # Collision step
    f, rho, u = functions.collision(f, constants.omega)
    
    u_at_point[t] = np.linalg.norm(u[32,32,:])

    # Check for steady-state every 500 iters after the initial transient
    if t >= steady_state_window:
        if functions.check_steady_state(u_at_point[:t+1],threshold=steady_state_threshold,window=steady_state_window,consecutive=steady_state_cons):
            steady_state_reached = True
            break

    # Save figure every 500 iterations
#    if (t + 1) % 500 == 0:
        #        print(f"Iteration {t+1}:")
        #      print(f"Time taken: {time.time() - start_time:.2f} seconds")
        #fig = plt.figure(figsize=(12, 10))
        #plt.streamplot(np.arange(width), np.arange(length), u[:,:, 0].T, u[:,:, 1].T, color='r')

        
        # plt.contourf(np.arange(width), np.arange(length), np.linalg.norm(u, axis=2).T, cmap='turbo',levels=100)
        #plt.colorbar(label='|u| (mm/s)')
        
        #plt.title('Velocity Field')
#plt.xlabel('X'):wq
        #plt.ylabel('Y')
        #plt.savefig(f"LB_plankton_figs/figat_{t+1}.png")
#plt.close()




# Plot velocity field at steady state
if steady_state_reached:
    plt.figure(figsize=(12, 10))

    # Plot velocity field
    plt.streamplot(np.arange(constants.width), np.arange(constants.length), u[:,:, 0].T, u[:,:, 1].T, color='r')
    plt.contourf(np.arange(constants.width), np.arange(constants.length), np.linalg.norm(u, axis=2).T, cmap='turbo', levels=100)
    plt.colorbar(label='Mag. of U (mm/s)')

    # Plot particle positions
    x0, vc, ini_velocities, p0 = akashiwo_dist.ini_swimspeed_cells(constants.width, constants.length)
    plt.quiver(x0[:, 0], x0[:, 1], ini_velocities[:,0]/1000, ini_velocities[:,1]/1000, color='lime', scale=10) # convert to mm/s
    plt.title(f'Initialization of H.Akashiwo at steady state (iter: {t}), Re: {Re}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.savefig("LB_plankton_figs/Velocity_Field_and_Particle_Positions_Steady_State.png")
    plt.show()

    print(f"Steady state reached at iteration {t}. Phytoplankton cells initialize")
else:
    print("Steady state not reached. Phytoplankton cells do not initialize. Adjust threshold or simulate for longer times")




plt.figure(figsize=(10, 6))
plt.plot(u_at_point, color='black')
plt.title('Fluid Speed @(64,64)')
plt.xlabel('time')
plt.ylabel('Fluid parcel velocity (mm/s)')
plt.grid(True)
plt.xlim([0,max_iter])
plt.tight_layout()
plt.savefig(f"LB_plankton_figs/U_Parcel.png")




