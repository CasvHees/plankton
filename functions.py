import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 13})
import constants
import akashiwo_dist 



# -------------functions-------------
# calculate density at x, t point on pdfGrid d
def f_rho(f):
    # should calculate whole grid
    return np.sum(f, axis=2)


def f_vel(f):
    # first calculate rho for the given f
    rho = f_rho(f)
    
    return (np.dot(f, constants.c).T / rho.T).T

def streaming(f, width= None, length = None, boundry=None, lid_vel=None):
    f_old = np.copy(f)

    # stream
    for i in range(constants.q):
        f[:, :, i] = np.roll(np.roll(f[:, :, i], constants.c[i, 0], axis=0), constants.c[i, 1], axis=1)
    
    # bounce_back at existing not moving walls
    if boundry:
        for i in range(constants.q):
            b = set_boundry(boundry, width, length)
            # bounce back where the wall exists
            f[:, :, i] = np.where(b, f_old[:, :, constants.opposite_direction[i]], f[:, :, i])

    # if upper lid velocity exists
    if lid_vel:
        # moving lid in upper boundry
        # update pdf according to lid velocity
        rho_wall = 2 * (f_old[:, -1, 6] + f_old[:, -1, 2] + f_old[:, -1, 5]) + \
                   f_old[:, -1, 3] + f_old[:, -1, 0] + f_old[:, -1, 1]

        f[:, -1, 4] = f_old[:, -1, constants.opposite_direction[4]] - 6 * constants.w_i[4] * rho_wall * np.dot(constants.c[4], [lid_vel, 0])
        f[:, -1, 8] = f_old[:, -1, constants.opposite_direction[8]] - 6 * constants.w_i[8] * rho_wall * np.dot(constants.c[8], [lid_vel, 0])
        f[:, -1, 7] = f_old[:, -1, constants.opposite_direction[7]] - 6 * constants.w_i[7] * rho_wall * np.dot(constants.c[7], [lid_vel, 0])
    
    return f

# Eq function.
def equilibrium(rho, u):

    cu = np.dot(u,constants.c.T)
    cu2 = cu ** 2
    u2 = u[:,:,0] ** 2 + u[:,:,1] ** 2

    return ((1 + 3*(cu.T) + 9/2*(cu2.T) - 3/2*(u2.T)) * rho.T ).T * constants.w_i

def collision(f, omega):
    rho = f_rho(f)
    u = f_vel(f)
    # to compute the new f, first we need to compute feq
    feq = equilibrium(rho, u)
    # relaxation
    f += omega * (feq - f)
    return f, rho, u


def set_boundry(boundry=constants.boundary, width=constants.width,length=constants.length):
    Mask = np.zeros((width, length))

    Mask[:,  0] = boundry[0]  # left
    Mask[:, -1] = boundry[1]  # right
    Mask[0,  :] = boundry[2]  # top
    Mask[-1, :] = boundry[3]  # bottom

    # convert the boundries to Boolean
    return Mask == 1

    #def plot_velocity(u, steps, milestone, re, lid_vel, omega, figsize = (10,10), width = constants.width, length = constants.length):
    
    #fig = plt.figure(figsize=figsize)
    #if lid_vel:
    #    plt.title("lattice dimensions: (%d * %d), omega: %.1f, lid velocity = %.1f, steps: %d, Re: %0.1f " %(width, length, omega, lid_vel, steps, re))
    #else:
    #    plt.title("lattice dimensions: (%d * %d), omega: %.1f, steps: %d, elapsed_time: %0.1f seconds" %(width, length, omega,steps, elapsed_time))
    #plt.streamplot(np.arange(width), np.arange(length), u[:,:, 0].T, u[:,:, 1].T)
    #plt.xlabel("lenght")
    #plt.ylabel("width")
    #plt.xticks(np.arange(0, length+1, 25))
    #plt.yticks(np.arange(0, width+1, 25))
    #plt.savefig("milestone_%d_%d_%d_%.1f_%d.png" %(milestone, width, length, omega,steps))

def calculate_re(omega, length, lid_vel):
    nu = 1/3 * (1/omega - 1/2)
    return (lid_vel*length) / (nu)



# Before initialising the plankton cells, we need to have a steady state flow, under a certain threshold we say it has been reached:


def check_steady_state(u_at_point, threshold=1e-5, window=500, consecutive=3):
    """
    Check if the system has reached steady state based on the change in fluid velocity magnitude.
    
    Parameters:
        u_at_point (ndarray): Array containing fluid speed at a specific point over time.
        threshold (float): Threshold for the mean change in fluid velocity magnitude to consider steady state.
        window (int): Window size for calculating mean change.
        consecutive (int): Number of consecutive iterations the mean change must be below the threshold.
    
    Returns:
        bool: True if steady state is reached, False otherwise.
    """
    if len(u_at_point) < window + consecutive:
        return False
    
    mean_changes = np.abs(u_at_point[window:] - u_at_point[:-window])  # Calculate mean changes
    
    # Truncate mean_changes to ensure its length is a multiple of window
    num_chunks = len(mean_changes) // window
    mean_changes = mean_changes[:num_chunks * window]
    
    # Reshape and calculate mean
    mean_change = np.mean(mean_changes.reshape(-1, window), axis=1)
    
    # Check for consecutive iterations below threshold
    steady_state_counter = 0
    for change in mean_change:
        if change < threshold:
            steady_state_counter += 1
        else:
            steady_state_counter = 0
        
        if steady_state_counter >= consecutive:
            return True
    
    return False



def calculate_dp_dt(p, vorticity, k, Psi=2):
    dp_dt = 1/(2 * Psi) * (k - np.dot(k, p) * p) + 0.5 * np.cross(vorticity, p) 
    return dp_dt 

def calculate_dX_dt(X, p, u, Phi=np.power(70.0,-3)):
    dX_dt = Phi * p + u 
    return dX_dt 

def k():
    return np.array([0, 1])

