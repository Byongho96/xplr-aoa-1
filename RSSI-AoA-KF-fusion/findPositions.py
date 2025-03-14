import numpy as np
import pandas as pd
import data_processing as dP
import math

#Function to obtain position by multilateration
def multilateration(config, mean_data):
    
    anchors_coordinates = {}

    for anchor in config['anchors']: 
        anchor_id = anchor['id']
        x, y, z = anchor['coordinates'] 
        anchors_coordinates[anchor_id] = {
            'x': x,
            'y': y,
            'z': z,
            'alpha': anchor['alpha'],
            'ref_coordinates': anchor['ref_coordinates']
        }   
        
    # Distance between anchors:
    d12 = np.sqrt(pow(anchors_coordinates[6501]['x'] - anchors_coordinates[6502]['x'], 2) + pow(anchors_coordinates[6501]['y'] - anchors_coordinates[6502]['y'], 2))
    d13 = np.sqrt(pow(anchors_coordinates[6501]['x'] - anchors_coordinates[6503]['x'], 2) + pow(anchors_coordinates[6501]['y'] - anchors_coordinates[6503]['y'], 2))
    d14 = np.sqrt(pow(anchors_coordinates[6501]['x'] - anchors_coordinates[6504]['x'], 2) + pow(anchors_coordinates[6501]['y'] - anchors_coordinates[6504]['y'], 2))
    d23 = np.sqrt(pow(anchors_coordinates[6502]['x'] - anchors_coordinates[6503]['x'], 2) + pow(anchors_coordinates[6502]['y'] - anchors_coordinates[6503]['y'], 2))
    d24 = np.sqrt(pow(anchors_coordinates[6502]['x'] - anchors_coordinates[6504]['x'], 2) + pow(anchors_coordinates[6502]['y'] - anchors_coordinates[6504]['y'], 2))
    d34 = np.sqrt(pow(anchors_coordinates[6503]['x'] - anchors_coordinates[6504]['x'], 2) + pow(anchors_coordinates[6503]['y'] - anchors_coordinates[6504]['y'], 2))

    df_MLT = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) # Dataframe to save the results
    
    for i in range (len(mean_data[6501])): # Loop through measurements
       
        # Distance with RSSI between anchors and tag
        d1 = mean_data[6501].loc[i, 'Dest_RSSI']
        d2 = mean_data[6502].loc[i, 'Dest_RSSI']
        d3 = mean_data[6503].loc[i, 'Dest_RSSI']
        d4 = mean_data[6504].loc[i, 'Dest_RSSI']
        
        '''
        Check if the radii of the circles formed by the distance and position of the anchor are eccentric
        or do not touch other circles. If so, make a correction by decreasing or increasing the distance.
        A loop is implemented to ensure that when a correction is made, other radii are not affected
        '''
        for _ in range(0,100):
            
            d2,d3 = dP.adjust_circle_eccentric(d2,d3,d23)
            d2,d4 = dP.adjust_circle_eccentric(d2,d4,d24)
            d3,d4 = dP.adjust_circle_eccentric(d3,d4,d34)  
            d1,d2 = dP.adjust_circle_eccentric(d1,d2,d12)
            d1,d3 = dP.adjust_circle_eccentric(d1,d3,d13)
            d1,d4 = dP.adjust_circle_eccentric(d1,d4,d14)
                
            d2,d3 = dP.adjust_separate_circle_radii(d2,d3,d23)
            d2,d4 = dP.adjust_separate_circle_radii(d2,d4,d24)
            d3,d4 = dP.adjust_separate_circle_radii(d3,d4,d34)
            d1,d2 = dP.adjust_separate_circle_radii(d1,d2,d12)
            d1,d3 = dP.adjust_separate_circle_radii(d1,d3,d13)
            d1,d4 = dP.adjust_separate_circle_radii(d1,d4,d14)
            
        # Multilateration equations : (x - x1)^2 + (y - y1)^2 = d1^2
        b1 = -pow(anchors_coordinates[6501]['x'],2)-pow(anchors_coordinates[6501]['y'],2)+pow(anchors_coordinates[6504]['x'],2)+pow(anchors_coordinates[6504]['y'],2)+pow(d1,2)-pow(d4,2)
        b2 = -pow(anchors_coordinates[6502]['x'],2)-pow(anchors_coordinates[6502]['y'],2)+pow(anchors_coordinates[6504]['x'],2)+pow(anchors_coordinates[6504]['y'],2)+pow(d2,2)-pow(d4,2)
        b3 = -pow(anchors_coordinates[6503]['x'],2)-pow(anchors_coordinates[6503]['y'],2)+pow(anchors_coordinates[6504]['x'],2)+pow(anchors_coordinates[6504]['y'],2)+pow(d3,2)-pow(d4,2)
       
        m11 = 2*(-anchors_coordinates[6501]['x']+anchors_coordinates[6504]['x'])
        m12 = 2*(-anchors_coordinates[6501]['y']+anchors_coordinates[6504]['y'])
        m21 = 2*(-anchors_coordinates[6502]['x']+anchors_coordinates[6504]['x'])
        m22 = 2*(-anchors_coordinates[6502]['y']+anchors_coordinates[6504]['y'])
        m31 = 2*(-anchors_coordinates[6503]['x']+anchors_coordinates[6504]['x'])
        m32 = 2*(-anchors_coordinates[6503]['y']+anchors_coordinates[6504]['y'])
        
        B = np.array([
            [b1],
            [b2],
            [b3]
            ])
        M = np.array([
            [m11, m12],
            [m21, m22],
            [m31, m32]
            ])
        
        # Perform multilateration with LS
        p = (np.linalg.pinv(M.transpose() @ M)) @ M.transpose() @ B 
        
        # Extract estimated X and Y
        Xest = p[0][0] 
        Yest = p[1][0] 
        
        if 'InitialTime' in mean_data[6501]: # # Determine real coordinates from mean_data
            Xreal = mean_data[6501].iloc[i,1]
            Yreal = mean_data[6501].iloc[i,2]
        else:
            Xreal = mean_data[6501].iloc[i,0]
            Yreal = mean_data[6501].iloc[i,1]
            
        new_row = [Xreal, Yreal, Xest, Yest]
        df_MLT.loc[len(df_MLT)] = new_row # Append the new row with real and estimated coordinates to df_MLT

    return df_MLT

# Function to obtain position by AoA+RSSI
def trigonometry(mean_data, config, df_posMLT):
    
    anchors_coordinates = {}
    for anchor in config['anchors']:
        anchor_id = anchor['id']
        x, y, z = anchor['coordinates']
        anchors_coordinates[anchor_id] = {
            'x': x,
            'y': y,
            'z': z,
            'alpha': anchor['alpha'],
            'ref_coordinates': anchor['ref_coordinates']
        }   
    
    df_trigonometry= pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) # Create dataframe o save the results
    
    posTrigonometry = {anchor_id: {'x': [], 'y': []} for anchor_id in [6501, 6502, 6503, 6504]} #dict to save the result of each anchor
    for anchor_id in [6501, 6502, 6503, 6504]:
        # Calculate the distance from anchors to the position obtained by multilateration
        mean_data[anchor_id]['Dest_MLT'] = np.sqrt((df_posMLT["Xest"] - anchors_coordinates[anchor_id]['x'])**2 + (df_posMLT["Yest"] - anchors_coordinates[anchor_id]['y'])**2 + (anchors_coordinates[anchor_id]['ref_coordinates'][2] - anchors_coordinates[anchor_id]['z'])**2)

        # iven the orientation of each anchor, each calculation requires a different equation, as each must be determined with respect to a common reference axis.
        if anchor_id == 6501:
            Xest = abs(-anchors_coordinates[anchor_id]['x']+mean_data[anchor_id]['Dest_MLT']*np.sin((np.deg2rad(90-mean_data[anchor_id]['AoA_az']))))
            Yest = abs(anchors_coordinates[anchor_id]['y']-mean_data[anchor_id]['Dest_MLT']*np.cos((np.deg2rad(90-mean_data[anchor_id]['AoA_az']))))
        elif anchor_id == 6502:
            Xest = abs(anchors_coordinates[anchor_id]['x']+mean_data[anchor_id]['Dest_MLT']*np.cos((np.deg2rad(90-mean_data[anchor_id]['AoA_az']))))
            Yest = abs(anchors_coordinates[anchor_id]['y']-mean_data[anchor_id]['Dest_MLT']*np.sin((np.deg2rad(90-mean_data[anchor_id]['AoA_az']))))
        elif anchor_id == 6503:
            Xest = abs(-anchors_coordinates[anchor_id]['x']+mean_data[anchor_id]['Dest_MLT']*np.sin((np.deg2rad(90+mean_data[anchor_id]['AoA_az']))))
            Yest = abs(anchors_coordinates[anchor_id]['y']-mean_data[anchor_id]['Dest_MLT']*np.cos((np.deg2rad(90+mean_data[anchor_id]['AoA_az']))))
        else:
            Xest = abs(anchors_coordinates[anchor_id]['x']+mean_data[anchor_id]['Dest_MLT']*np.cos((np.deg2rad(90+mean_data[anchor_id]['AoA_az']))))
            Yest = abs(anchors_coordinates[anchor_id]['y']-mean_data[anchor_id]['Dest_MLT']*np.sin((np.deg2rad(90+mean_data[anchor_id]['AoA_az']))))
        
        # Store the Xest and Yest values for each anchor
        posTrigonometry[anchor_id]['x'].append(Xest)
        posTrigonometry[anchor_id]['y'].append(Yest)
            

    mean_pos = {'Xest_mean': [], 'Yest_mean': []}   
    # Loop through the rows and calculate the mean Xest and Yest across all anchors
    for i in range(len(df_posMLT)):
        
        # Get the Xest and Yest values for each anchor at the current time step
        x_values = [posTrigonometry[anchor_id]['x'][0][i] for anchor_id in posTrigonometry]
        y_values = [posTrigonometry[anchor_id]['y'][0][i] for anchor_id in posTrigonometry]

        # Calculate the mean of Xest and Yest for all 4 anchors at this row
        Xest_mean = np.mean(x_values)
        Yest_mean = np.mean(y_values)

        # Append the mean values to the `mean_pos` dictionary
        mean_pos['Xest_mean'].append(Xest_mean)
        mean_pos['Yest_mean'].append(Yest_mean)
  
    # Convert mean_pos dictionary to a DataFrame
    df_mean_pos = pd.DataFrame(mean_pos)
    
    # Assign results to df_trigonometry
    df_trigonometry['Xest'] = df_mean_pos['Xest_mean']
    df_trigonometry['Yest'] = df_mean_pos['Yest_mean']
    df_trigonometry['Xreal'] = df_posMLT['Xreal']
    df_trigonometry['Yreal'] = df_posMLT['Yreal']
    
    return df_trigonometry

# Function to obtain position by AoA-only
def triangulation(mean_data, config):
    
    anchors_coordinates = {}
    for anchor in config['anchors']:
        anchor_id = anchor['id']
        x, y, z = anchor['coordinates']
        anchors_coordinates[anchor_id] = {
            'x': x,
            'y': y,
            'z': z,
            'alpha': anchor['alpha'],
            'ref_coordinates': anchor['ref_coordinates']
        }   
    
    df_triangulation = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) # Create dataframe o save the results
    
    df_triangulation['Xreal'] = mean_data[6501]['Xcoord'] # Assign real positions
    df_triangulation['Yreal'] = mean_data[6501]['Ycoord']
    
    for i in range(len(mean_data[6501])): # Loop through measurements
        
        aoa1 = math.pi - math.radians(mean_data[6501].loc[i,'AoA_az']) 
        aoa2 = math.pi/2 - math.radians(mean_data[6502].loc[i,'AoA_az']) 
        aoa3 = math.pi - math.radians(mean_data[6503].loc[i,'AoA_az']) 
        aoa4 = math.pi/2 - math.radians(mean_data[6504].loc[i,'AoA_az'])
        
        '''
        The article by Ottoy and Kupper was used as a reference for calculating the triangulation, using the least squares estimation.

        Y_intercept = Y_a - tan(θ)  * X_a
        H * T = Y_intercept
        '''
        h11 = -math.tan(aoa1)
        h21 = -math.tan(aoa2)
        h31 = -math.tan(aoa3)
        h41 = -math.tan(aoa4)
        h12 = 1
        h22 = 1
        h32 = 1
        h42 = 1

        H = np.array([
                [h11, h12],
                [h21, h22],
                [h31, h32],
                [h41, h42]
            ])

        c11 = anchors_coordinates[6501]['y'] - anchors_coordinates[6501]['x']*math.tan(aoa1)
        c21 = anchors_coordinates[6502]['y'] - anchors_coordinates[6502]['x']*math.tan(aoa2)
        c31 = anchors_coordinates[6503]['y'] - anchors_coordinates[6503]['x']*math.tan(aoa3)
        c41 = anchors_coordinates[6504]['y'] - anchors_coordinates[6504]['x']*math.tan(aoa4)

        c = np.array([
                [c11],
                [c21],
                [c31],
                [c41]
            ])

        e = np.linalg.inv(H.transpose().dot(H)).dot(H.transpose()).dot(c) #LS

        Xest = e[0][0] #Estimate in X-axis
        Yest = e[1][0] #Estimate in Y-axis
        df_triangulation.iloc[i,2] = Xest
        df_triangulation.iloc[i,3] = Yest
    
    return df_triangulation

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

def kalman_filter_acceleration(zk, config, initial_position, R):
    
    #Initialize state vector with initial position
    xk = np.zeros((6,len(zk))).transpose()
    xk[0] = [initial_position[0], 0, 0, initial_position[1], 0, 0]
    xk = xk.transpose()
    
    #Initialize Matrices
    P = np.array(config['kalman_filter_acceleration']['P'])
    A = np.array(config['kalman_filter_acceleration']['A'])
    Q = np.array(config['kalman_filter_acceleration']['Q'])
    C = np.array(config['kalman_filter_acceleration']['C'])

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

def extended_kalman_filter(zk, config, initial_position, R):
    
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