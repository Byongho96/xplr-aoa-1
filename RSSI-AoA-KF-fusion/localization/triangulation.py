import numpy as np
import pandas as pd
import math

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
