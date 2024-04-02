import matplotlib.pyplot as plt
import numpy as np
import functions
import matplotlib.colors
import sys
import time
from matplotlib import rc

rc('text', usetex=True)
# Set the backend
matplotlib.use('TkAgg')

width = 128
length = 128
max_iter = 10000
omega = 1
lid_vel = 2

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

t_ini = 0 
start_time = time.time()

u_at_point = np.zeros(max_iter)

for t in range(max_iter):
    # Streaming step
    f = functions.streaming(f, width, length, boundary, lid_vel)     
    # Collision step
    f, rho, u = functions.collision(f, omega)
    
    u_at_point[t] = np.linalg.norm(u[64,64,:])

    # Save figure every 500 iterations
    if (t + 1) % 500 == 0:
        print(f"Iteration {t+1}:")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        fig = plt.figure(figsize=(12, 10))
        plt.streamplot(np.arange(width), np.arange(length), u[:,:, 0].T, u[:,:, 1].T, color='r')

 
        plt.contourf(np.arange(width), np.arange(length), np.linalg.norm(u, axis=2).T, cmap='turbo',levels=100)
        plt.colorbar(label='|u| (mm/s)')
        
        plt.title('Velocity Field')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(f"LB_plankton_figs/figat_{t+1}.png")
        plt.close()

plt.figure(figsize=(10, 6))
plt.plot(u_at_point, color='black')
plt.title('Fluid Speed @(64,64)')
plt.xlabel('time')
plt.ylabel('Fluid parcel velocity (mm/s)')
plt.grid(True)
plt.xlim([0,max_iter])
plt.tight_layout()
plt.savefig(f"LB_plankton_figs/U_Parcel.png")
plt.show()
