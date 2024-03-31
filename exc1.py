import matplotlib.pyplot as plt
import numpy as np
import functions
import matplotlib.colors
from matplotlib.colors
import time
import sys

# Set the backend
matplotlib.use('TkAgg')

width = 128
length = 128
max_iter = 5000
omega = 1
lid_vel = 0.15

boundary = (True, True, False, True) 

# Initialize rho and u
rho = np.ones((width, length))
u = np.zeros((width, length, 2))

# Initialize density function w.r.t. rho and u 
f = functions.equilibrium(rho, u)

# calculate corresponding Reynolds number
Re = functions.calculate_re(omega=omega, length=length, lid_vel=lid_vel)
pos, orient = functions.initialize_cells(num_cells, domain_size)
# create a directory to save the figures
import os
if not os.path.exists('LB_plankton_figs'):
    os.makedirs('LB_plankton_figs')

for t in range(max_iter):
    # Streaming step
    f = functions.streaming(f, width, length, boundary, lid_vel) 
    
    # Collision step
    f, rho, u = functions.collision(f, omega)
    cell_velocities = functions.interpolate_velocity(u, positions)
    positions += cell_velocities
    orientations = functions.update_orientations(orientations, cell_velocities)
    # Save figure every 500 iterations
    if (t + 1) % 500 == 0:
        fig = plt.figure(figsize=(12, 10))
        plt.streamplot(np.arange(width), np.arange(length), u[:,:, 0].T, u[:,:, 1].T, color='r')

 
        plt.contourf(np.arange(width), np.arange(length), np.linalg.norm(u, axis=2).T, cmap='turbo',levels=100)
        plt.colorbar(label='|u|')
        
        plt.title('Velocity Field')
        plt.tight_layout()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(f"LB_plankton_figs/figat_{t+1}.png")
        plt.close()
