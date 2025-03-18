import numpy as np
import pandas as pd

def adjust_overlap_circle_radii(r1, r2, d12):    
    if r1 > r2 + d12:
        r1 -= (r1 / (r2 + r1)) * (r1 - d12 - r2)
        r2 += (r2 / (r2 + r1)) * (r1 - d12 - r2)
        r1 = r1 * 0.9
        r2 = r2 * 1.1
    elif r2 > r1 + d12:
        r2 -= (r2 / (r2 + r1)) * (r2 - d12 - r1)
        r1 += (r1 / (r2 + r1)) * (r2 - d12 - r1)
        r1 = r1 * 1.1
        r2 = r2 * 0.9

    return r1, r2

def adjust_separate_circle_radii(r1, r2, d12, tolerance=1e-2):
    if d12 > r1 + r2:
        r1 = (r1 / (r1 + r2)) * d12
        r2 = (r2 / (r1 + r2)) * d12
        r1 = r1 * 1.1
        r2 = r2 * 1.1

    return r1, r2

# Multilateration function
def multilateration(config, mean_data):
    
    anchors_coordinates = {
        anchor['id']: {
            'x': anchor['coordinates'][0],
            'y': anchor['coordinates'][1],
            'z': anchor['coordinates'][2],
            'alpha': anchor['alpha'],
            'ref_coordinates': anchor['ref_coordinates']
        }
        for anchor in config['anchors']
    } 
        
    # Distance between anchors:
    d12 = int(np.sqrt(pow(anchors_coordinates[6501]['x'] - anchors_coordinates[6502]['x'], 2) + pow(anchors_coordinates[6501]['y'] - anchors_coordinates[6502]['y'], 2)))
    d13 = int(np.sqrt(pow(anchors_coordinates[6501]['x'] - anchors_coordinates[6503]['x'], 2) + pow(anchors_coordinates[6501]['y'] - anchors_coordinates[6503]['y'], 2)))
    d14 = int(np.sqrt(pow(anchors_coordinates[6501]['x'] - anchors_coordinates[6504]['x'], 2) + pow(anchors_coordinates[6501]['y'] - anchors_coordinates[6504]['y'], 2)))
    d23 = int(np.sqrt(pow(anchors_coordinates[6502]['x'] - anchors_coordinates[6503]['x'], 2) + pow(anchors_coordinates[6502]['y'] - anchors_coordinates[6503]['y'], 2)))
    d24 = int(np.sqrt(pow(anchors_coordinates[6502]['x'] - anchors_coordinates[6504]['x'], 2) + pow(anchors_coordinates[6502]['y'] - anchors_coordinates[6504]['y'], 2)))
    d34 = int(np.sqrt(pow(anchors_coordinates[6503]['x'] - anchors_coordinates[6504]['x'], 2) + pow(anchors_coordinates[6503]['y'] - anchors_coordinates[6504]['y'], 2)))

    df_MLT = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) # Dataframe to save the results
    
    for i in range (len(mean_data[6501])): # Loop through measurements
       
        # Distance with RSSI between anchors and tag
        d1 = int(mean_data[6501].loc[i, 'Dist_RSSI'])
        d2 = int(mean_data[6502].loc[i, 'Dist_RSSI'])
        d3 = int(mean_data[6503].loc[i, 'Dist_RSSI'])
        d4 = int(mean_data[6504].loc[i, 'Dist_RSSI'])
        
        '''
        Check if the radii of the circles formed by the distance and position of the anchor are eccentric
        or do not touch other circles. If so, make a correction by decreasing or increasing the distance.
        A loop is implemented to ensure that when a correction is made, other radii are not affected
        '''
        for _ in range(0, 100):
            
            d2, d3 = adjust_overlap_circle_radii(d2, d3, d23)
            d2, d4 = adjust_overlap_circle_radii(d2, d4, d24)
            d3, d4 = adjust_overlap_circle_radii(d3, d4, d34)  
            d1, d2 = adjust_overlap_circle_radii(d1, d2, d12)
            d1, d3 = adjust_overlap_circle_radii(d1, d3, d13)
            d1, d4 = adjust_overlap_circle_radii(d1, d4, d14)
                
            d2, d3 = adjust_separate_circle_radii(d2, d3, d23)
            d2, d4 = adjust_separate_circle_radii(d2, d4, d24)
            d3, d4 = adjust_separate_circle_radii(d3, d4, d34)
            d1, d2 = adjust_separate_circle_radii(d1, d2, d12)
            d1, d3 = adjust_separate_circle_radii(d1, d3, d13)
            d1, d4 = adjust_separate_circle_radii(d1, d4, d14)
            
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
        
        if 'InitialTime' in mean_data[6501]: # Determine real coordinates from mean_data
            Xreal = mean_data[6501].iloc[i,1]
            Yreal = mean_data[6501].iloc[i,2]
        else:
            Xreal = mean_data[6501].iloc[i,0]
            Yreal = mean_data[6501].iloc[i,1]
            
        new_row = [Xreal, Yreal, Xest, Yest]
        df_MLT.loc[len(df_MLT)] = new_row # Append the new row with real and estimated coordinates to df_MLT

    return df_MLT
