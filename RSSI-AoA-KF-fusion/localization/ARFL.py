import numpy as np

"""
Proposed Fusion - ARFL
Note that the measurements vector already possesses the positions 
for all time steps obtained by the AoA+RSSI and AoA-only methods.
"""
def ARFL_fusion(zk_1, zk_2, R_1, R_2, initial_position, config):
    #1 - Trigonometry (AoA+RSSI)
    #2 - Triangulation (AoA-only)
    
    #Initialize state vector with initial position
    xk_1 = np.zeros((4,len(zk_1))).transpose()
    xk_1[0] = [initial_position[0], 0, initial_position[1], 0]
    xk_1 = xk_1.transpose()
    
    xk_2 = np.zeros((4,len(zk_2))).transpose()
    xk_2[0] = [initial_position[0], 0, initial_position[1], 0]
    xk_2 = xk_2.transpose()
    
    xk_arfl = np.zeros((4,len(zk_2))).transpose()
    xk_arfl[0] = [initial_position[0], 0, initial_position[1], 0]
    xk_arfl = xk_arfl.transpose()
    
    
    #Initialize Matrices
    P_1 = np.array(config['kalman_filter']['P'])
    P_2 = np.array(config['kalman_filter']['P'])
    P_arfl = np.array(config['kalman_filter']['P'])
    
    A = np.array(config['kalman_filter']['A'])
    Q = np.array(config['kalman_filter']['Q'])
    C = np.array(config['kalman_filter']['C'])
    
    for i in range (1, len(zk_1)): # Loop through all time steps
        
        # Predict Local x_{k|k-1}   
        xk_1[:,i:i+1] = A @ xk_1[:,i-1:i] 
        xk_2[:,i:i+1] = A @ xk_2[:,i-1:i]
        
        # Predict Local P_{k|k-1}
        P_1 = A @ P_1 @ A.transpose() + Q
        P_2 = A @ P_2 @ A.transpose() + Q
        
        # Predict Fused x_{k|k-1}
        xk_arfl[:,i:i+1] = xk_1[:,i:i+1] + (P_1) @ np.linalg.inv((P_1 + P_2)) @ (xk_2[:,i:i+1]- xk_1[:,i:i+1])
        
        # Predict Fused P_{k|k-1}
        Pf = P_1 - (P_1) @ (np.linalg.inv(P_1 + P_2)) @ (P_1)
        
        # Update Local K_{k}
        S_1 = C @ Pf @ C.transpose() + R_1
        S_2 = C @ Pf @ C.transpose() + R_2
        K_1 = Pf @ C.transpose() @ np.linalg.inv(S_1)
        K_2 = Pf @ C.transpose() @ np.linalg.inv(S_2)
        
        # Update Local P_{k|k}
        P_1 = (np.eye(4) - K_1 @ C) @ Pf
        P_2 = (np.eye(4) - K_2 @ C) @ Pf
    
        # Update Local x_{k|k} 
        zdash = C @ xk_arfl[:,i:i+1]
        xk_1[:,i:i+1] = xk_arfl[:,i:i+1] + K_1 @ (zk_1[i:i+1,:].transpose() - zdash)
        xk_2[:,i:i+1] = xk_arfl[:,i:i+1] + K_2 @ (zk_2[i:i+1,:].transpose() - zdash)
        
        # Update Fused x_{k|k}
        xk_arfl[:,i:i+1] = xk_1[:,i:i+1] + (P_1) @ np.linalg.inv((P_1 + P_2)) @ (xk_2[:,i:i+1]- xk_1[:,i:i+1])
    
    xk_arfl = xk_arfl.transpose()
    
    return xk_arfl