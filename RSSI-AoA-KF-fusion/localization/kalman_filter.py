import numpy as np

# Kalman Filter
def kalman_filter(zk, config, initial_position, R):
    
    #Initialize state vector with initial position
    xk = np.zeros((4,len(zk))).transpose()
    xk[0] = [initial_position[0], 0, initial_position[1], 0]
    xk = xk.transpose()
    
    #Initialize Matrices
    P = np.array(config['kalman_filter']['P'])
    A = np.array(config['kalman_filter']['A'])
    Q = np.array(config['kalman_filter']['Q'])
    C = np.array(config['kalman_filter']['C'])

    for i in range (1, len(zk)): # Loop through all time steps
        
        # Predict Step
        xk[:,i:i+1] = A @ xk[:,i-1:i]
        P = A @ P @ A.transpose() + Q
        
        #Measurement Update
        K = P @ C.transpose() @ np.linalg.inv(C @ P @ C.transpose() + R) # Kalman Gain
        xk[:,i:i+1] = xk[:,i:i+1] + K @ (zk[i:i+1,:].transpose() - C @ xk[:,i:i+1])

        P = P - K @ C @ P
    
    xk = xk.transpose()
    
    return xk