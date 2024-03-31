import numpy as np 

q = 9
w_i = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

# D2Q9 velocity vectors
c = np.array([[ 0,  0],[ 1,  0],[ 0,  1],[-1,  0],[ 0, -1],[ 1,  1],[-1,  1],[-1, -1],[ 1, -1]])

#opposite directions for bouncing back
opposite_direction = np.array([0,3,4,1,2,7,8,5,6], dtype = np.int8)
