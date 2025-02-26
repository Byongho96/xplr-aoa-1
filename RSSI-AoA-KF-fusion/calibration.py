import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_processing as dp

beacons_column_names = ["TimeStamp", "TagID", "1stP","AoA_az", "AoA_el", "2ndP", "Channel", "AnchorID"]
gt_column_names = ["StartTime", "EndTime", "Xcoord","Ycoord"]

def run():
    print('Start Calibration Process')
    
    # Open Configrations File
    with open("config-calibration.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # Load dataset
    beacons_df = pd.read_csv(config['file_paths']['beacons'], header=None, names=beacons_column_names)
    gt_df = pd.read_csv(config['file_paths']['gt'], header=None, names=gt_column_names)

    # Add ['Xcoord', 'Ycoord'] data to `beacons_df` based on the timestamps
    for _, row in gt_df.iterrows():
        mask = (beacons_df['TimeStamp'] >= row['StartTime']) & (beacons_df['TimeStamp'] <= row['EndTime'])
        beacons_df.loc[mask, ['Xcoord', 'Ycoord']] = row[['Xcoord', 'Ycoord']].values
        
    # Remove unnecessary data where ['Xcoord', 'Ycoord'] are NaN
    beacons_df.dropna(subset=['Xcoord', 'Ycoord'], inplace=True)
    
    # Group the beacons_df by AnchorID
    beacons_df_by_anchor = {anchor_id: df for anchor_id, df in beacons_df.groupby("AnchorID")}

    # Initialize a dictionary to store the DataFrames
    beacons_dataset_dict = {}
    df_mean_dict = {}
    pathloss_dict = {}
    
    # Calculate the pathloss coefficient for each anchor
    print('PATHLOSS COEFFICIENT VALUES:')

    for anchor in config['anchors']:
        id, coordinates, ref_coordinates = anchor.values()
        dH = coordinates[2] - ref_coordinates[2]

        # Get the corresponding beacons calibration dataframe for the current anchor
        anchor_beacons = beacons_df_by_anchor.get(id)
        
        # Add data
        anchor_beacons["Distance"] = np.sqrt(
            (anchor_beacons["Xcoord"] - coordinates[0])**2 +
            (anchor_beacons["Ycoord"] - coordinates[1])**2 +
            (dH) ** 2
        )
        anchor_beacons['RSSILin'] = np.power(10, (anchor_beacons['2ndP'] - 30) / 10)
        anchor_beacons['Z-Score'] = anchor_beacons.groupby(['Xcoord', 'Ycoord'])['2ndP'].transform(lambda x: (x - x.mean()) / x.std()).abs()
        
        # Filter based on Z-Score
        anchor_beacons = anchor_beacons[anchor_beacons['Z-Score'] <= 2]

        # Calculate the mean dataframe by tag positions
        mean_df_by_positions = dp.mean_calibration(anchor_beacons, config)
        
        #calculate the pathloss coefficient
        a = dp.pathloss_calculation(mean_df_by_positions, '2ndP', ref_coordinates, coordinates, dH)
        
        #Calculate the RSSI model
        dp.rssi_model(mean_df_by_positions, '2ndP', coordinates, a, ref_coordinates, dH)
        
        # Update dataframes
        beacons_dataset_dict[id] = anchor_beacons
        
        df_mean_dict[id] = mean_df_by_positions
        
        pathloss_dict[id] = a
        
        print(f'Anchor: {id} --- PLc: {a:.3f}') #f"{results[i]:.2f}
            
            
    # Plot graphics: use the config.yaml to determine whether to display no plots, only the first plot, or all RSSI models.
    
    # Define the aspect ratio
    aspect_ratio = 16 / 9
    # Set the figure size based on the aspect ratio
    fig_width = 8  # You can choose any width
    fig_height = fig_width / aspect_ratio
    
    #for anchor_id in ['a6501', 'a6502', 'a6503', 'a6504']:
        
    if config['additional']['plot_all_RSSI'] == False and config['additional']['plot_first_RSSI'] == False:
        print('RSSI plot not selected in config.yaml')
        
    else:
        if config['additional']['plot_all_RSSI'] == True:
            print('Plotting RSSI for all anchors')
            plot_anchors = [6501, 6502, 6503, 6504]
        else:
            print('Plotting RSSI for Anchor 6501 only')
            plot_anchors = [6501]
            
        for anchor_id in plot_anchors:    
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                        
            x_plot = beacons_dataset_dict[anchor_id].iloc[::60, 10].values / 100  # Slicing every 60th value and dividing by 100 to transform in meters
            y_plot = beacons_dataset_dict[anchor_id].loc[::60, config['additional']['polarization']].values  # Slicing every 60th value for RSSI 2ndP
            plt.plot(x_plot, y_plot,'o')
            
            plt.plot(df_mean_dict[anchor_id]["D_real"]/100, df_mean_dict[anchor_id][config['additional']['polarization']],'r*') # distance (meters) x mean RSSI
            
            df_mean_dict[anchor_id].sort_values(by=['D_real'], inplace=True)
            plt.plot(df_mean_dict[anchor_id]["D_real"]/100, df_mean_dict[anchor_id]["RSSImodel"],'k', linewidth=3) # # distance (meters) x RSSI model
            
            plt.legend(['RSSI Measurements','Mean RSSI', 'RSSI Model'],loc=1, fontsize=11)
            plt.xlabel("Distance [m]",fontsize=12)
            plt.ylabel("RSSI [dBm]", fontsize=12)
            plt.title(f"RSSI x Distance - Anchor {id}", fontsize=14)
            plt.grid()
            plt.show()