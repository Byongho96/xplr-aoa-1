import numpy as np
import pandas as pd

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