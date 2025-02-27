import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_processing as dp
from scipy.optimize import minimize

beacons_column_names = ["TimeStamp", "TagID", "1stP","AoA_az", "AoA_el", "2ndP", "Channel", "AnchorID"]
gt_column_names = ["StartTime", "EndTime", "Xcoord","Ycoord"]

def run():
    print('Start Calibration Process')
    
    # Open Configrations File
    with open("config-calibration.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # Load dataset as DataFrames
    beacons_df = pd.read_csv(config['file_paths']['beacons'], header=None, names=beacons_column_names)
    gt_df = pd.read_csv(config['file_paths']['gt'], header=None, names=gt_column_names)

    # Add `Xcoord`` and `Ycoord`` columns to the beacons_df (optimized with merge_asof : binary search)
    beacons_df = pd.merge_asof(
        beacons_df, gt_df[['StartTime', 'EndTime', 'Xcoord', 'Ycoord']], 
        left_on='TimeStamp', right_on='StartTime'
    )
    beacons_df = beacons_df[beacons_df['TimeStamp'] <= beacons_df['EndTime']]

    # Remove unnecessary beacons data
    beacons_df.dropna(subset=['Xcoord', 'Ycoord'], inplace=True)
    
    # Create dictionaries to store the results
    beacons_df_by_anchor = {anchor_id: df for anchor_id, df in beacons_df.groupby("AnchorID")}
    mean_df_by_anchor = {}
    pathloss_by_anchor = {}
    
    # Calculate the path-loss coefficient for each anchor
    print('PATHLOSS COEFFICIENT VALUES:')

    polarization_column = config['rssi']['polarization_column']

    for anchor in config['anchors']:
        id, coordinates, ref_coordinates = anchor.values()

        # Get the corresponding beacons calibration dataframe for the current anchor
        anchor_beacons_df = beacons_df_by_anchor.get(id)
        
        # Add columns to the anchor_beacons_df
        anchor_beacons_df['RSSILin'] = np.power(10, (anchor_beacons_df[polarization_column] - 30) / 10)
        anchor_beacons_df['Distance'] = np.sqrt(
            (anchor_beacons_df['Xcoord'] - coordinates[0])**2 +
            (anchor_beacons_df['Ycoord'] - coordinates[1])**2 +
            (coordinates[2] - ref_coordinates[2]) ** 2
        )
        
        # Filter outliers based on Z_score
        anchor_beacons_df['Z_score'] = anchor_beacons_df.groupby(['Xcoord', 'Ycoord'])[polarization_column].transform(lambda x: (x - x.mean()) / x.std()).abs()
        anchor_beacons_df = anchor_beacons_df[anchor_beacons_df['Z_score'] <= config['rssi']['Z_score']]

        # Calculate the mean dataframe by tag positions
        mean_df_by_positions = anchor_beacons_df.groupby(['Xcoord', 'Ycoord']).agg({
            'AoA_az': 'mean',
            'AoA_el': 'mean',
            'RSSILin': 'mean',
            'Distance': 'mean'
        }).reset_index()
        mean_df_by_positions['RSSI'] = 10 * np.log10(mean_df_by_positions['RSSILin']) + 30
        
        # Calculate the path-loss coefficient with LSM
        d0, rssi_d0 = mean_df_by_positions.loc[(mean_df_by_positions['Xcoord'] == ref_coordinates[0]) & (mean_df_by_positions['Ycoord'] == ref_coordinates[1]), ['Distance', 'RSSI']].mean()

        def objective_function(n_value):
            model = rssi_d0 - 10 * n_value * np.log10(mean_df_by_positions['Distance'] / d0)
            error = np.sum((mean_df_by_positions['RSSI'] - model) ** 2)
            return error

        result = minimize(objective_function, 2.0) # Initial guess is 2.0
        pathloss = result.x[0]

        # Add calculated RSSI values based on the path-loss coefficient results
        mean_df_by_positions['RSSIModel'] = rssi_d0 - 10 * pathloss * np.log10(mean_df_by_positions['Distance'] / d0)

        # Store the results
        mean_df_by_anchor[id] = mean_df_by_positions
        pathloss_by_anchor[id] = pathloss
        
        print(f'Anchor: {id} --- PLc: {pathloss:.3f}') #f"{results[i]:.2f}
            
            
    # Plot the results
    fig_width = config['plot']['fig_size'][0]
    fig_height = config['plot']['fig_size'][1]
    
    _fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))

    for i, anchor_id in enumerate([6501, 6502, 6503, 6504]):   
        row, col = divmod(i, 2)
        ax = axes[row, col]

        # Set subplot properties
        ax.set_title(f"RSSI x Distance - Anchor {id}", fontsize=14)
        ax.set_xlabel("Distance [m]",fontsize=12)
        ax.set_ylabel("RSSI [dBm]", fontsize=12)
                    
        # Plot RSSI measurements
        x_plot = beacons_df_by_anchor[anchor_id].loc[::60, 'Distance'].values / 100  # Slicing every 60th value and dividing by 100 to transform in meters
        y_plot = beacons_df_by_anchor[anchor_id].loc[::60, polarization_column].values  # Slicing every 60th value for RSSI
        ax.plot(x_plot, y_plot,'o')
        
        # Plot mean RSSI
        ax.plot(mean_df_by_anchor[anchor_id]["Distance"]/100, mean_df_by_anchor[anchor_id]["RSSI"],'r*')
        
        # Plot RSSI model
        mean_df_by_anchor[anchor_id].sort_values(by=['Distance'], inplace=True)
        ax.plot(mean_df_by_anchor[anchor_id]["Distance"]/100, mean_df_by_anchor[anchor_id]["RSSIModel"],'k', linewidth=3)
        
        ax.legend(['RSSI Measurements', 'Mean RSSI', 'RSSI Model'],loc=1, fontsize=11)
        ax.grid()

    plt.tight_layout()
    plt.show()