from typing import Tuple
import yaml
import pandas as pd
import numpy as np
from localization import multilateration, trigonometry, triangulation, kalman_filter, ARFL_fusion
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 

beacons_column_names = ["TimeStamp", "TagID", "1stP","AoA_az", "AoA_el", "2ndP", "Channel", "AnchorID"]  # Column IDs for Beacons Dataset
gt_column_names = ["StartTime", "EndTime", "Xcoord","Ycoord"] # Column IDs for Ground Truth Dataset


def create_measurement_matrix(position_data, x_key='Xest', y_key='Yest'):
    x_values = position_data[x_key]
    y_values = position_data[y_key]
    return np.array([x_values, y_values]).transpose()


def run(case):
    print('Start Mobility Process')
    
    # Open Configuration File
    with open("config-mobility.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    initial_position = config['initial_position'][f'case_{case}']

    # Load dataset as DataFrames
    beacons_df_by_run = {}
    gt_df_by_run = {}

    for run in range(1, 5):
        beacons_df = pd.read_csv(config['file_paths'][f'case_{case}'][f'run{run}']['beacons'], header=None, names=beacons_column_names)
        gt_df = pd.read_csv(config['file_paths'][f'case_{case}'][f'run{run}']['gt'], header=None, names=gt_column_names)

        '''Interpolate the ground truth data to match the beacons data'''
        # Add 'Vx' and 'Vy' columns to the ground truth DataFrame
        gt_df['Vx'] = ((gt_df.iloc[:, 2].diff()) / (gt_df.iloc[:, 0].diff())) * 10  # cm/ms → m/s
        gt_df['Vy'] = ((gt_df.iloc[:, 3].diff()) / (gt_df.iloc[:, 0].diff())) * 10
        gt_df.fillna(0, inplace=True)  # Handle NaN values in first row
        

        # Filter beacons out of time range
        initial_time, final_time = gt_df.loc[gt_df.index[0], 'StartTime'], gt_df.loc[gt_df.index[-1], 'StartTime']
        beacons_df = beacons_df[(beacons_df['TimeStamp'] >= initial_time) & (beacons_df['TimeStamp'] <= final_time)] # Filter beacons_df out of range

        # Interpolate
        def interpolate_position(row):
            valid_gt = gt_df[gt_df['StartTime'] >= row['TimeStamp']].iloc[0] # Get the last valid ground truth row
            time_elapsed = (row['TimeStamp'] - valid_gt['StartTime']) / 1000
            new_x = valid_gt['Xcoord'] + valid_gt['Vx'] * time_elapsed * 100  # Convert m → cm
            new_y = valid_gt['Ycoord'] + valid_gt['Vy'] * time_elapsed * 100
            return pd.Series([int(new_x), int(new_y)])
        
        beacons_df[['Xcoord', 'Ycoord']] = beacons_df.apply(interpolate_position, axis=1)

        # Group per delta_T seconds -> Take mean values for each time interval (Line 93)
        Tdisc = config['kalman_filter']['delta_T'] 
        beacons_df['InitialTime'] = ((beacons_df['TimeStamp'] - initial_time) / 1000 / Tdisc).astype(int)

        beacons_df_by_run[f'run{run}'] = beacons_df
        gt_df_by_run[f'run{run}'] = gt_df


    # Vector to save results of each run -> Case results
    all_pos_multilateration = []
    all_errors_multilateration = []
    mean_errors_multilateration = []

    all_pos_trigonometry = []
    all_errors_trigonometry = []
    mean_errors_trigonometry = []

    all_pos_triangulation = []
    all_errors_triangulation = []
    mean_errors_triangulation  = []

    all_pos_multilateration_KF = []
    all_errors_multilateration_KF = []
    mean_errors_multilateration_KF = []

    all_pos_trigonometry_KF = []
    all_errors_trigonometry_KF = []
    mean_errors_trigonometry_KF = []

    all_pos_triangulation_KF = []
    all_errors_triangulation_KF = []
    mean_errors_triangulation_KF = []

    all_pos_ARFL = []
    all_errors_ARFL = []
    mean_errors_ARFL = []
    
    for run in range(1, 5):
        print(f'Executing RUN {run}')

        mean_data = {}

        for anchor in config['anchors']:
            anchor_id, alpha, coordinates, ref_coordinates = anchor.values()

            anchor_beacons = beacons_df_by_run[f'run{run}'][beacons_df_by_run[f'run{run}']["AnchorID"] == anchor_id].copy()
        
            # Add 'RSSILin' and 'Distance' columns to the DataFrame
            anchor_beacons['RSSILin'] = np.power(10, (anchor_beacons['2ndP'] - 30) / 10)
            anchor_beacons['Distance'] = np.sqrt(
                (anchor_beacons['Xcoord'] - coordinates[0]) ** 2 +
                (anchor_beacons['Ycoord'] - coordinates[1]) ** 2 +
                (coordinates[2] - ref_coordinates[2]) ** 2
            )

            # Obtain the mean data for each position (discretized time in delta_T)
            anchor_mean_data = anchor_beacons.groupby('InitialTime').agg({
                'Xcoord': 'mean',
                'Ycoord': 'mean',
                'Distance': 'mean',
                'AoA_az': 'mean',
                'AoA_el': 'mean',
                'RSSILin': 'mean'
            }).reset_index()

            # Calculate additional columns
            anchor_mean_data['RSSI'] = 10 * np.log10(anchor_mean_data['RSSILin']) + 30
            anchor_mean_data['InitialTime'] = anchor_mean_data['InitialTime'] * config['kalman_filter']['delta_T']  # Adjust InitialTime to represent the time in seconds

            d0 = anchor_mean_data.loc[anchor_mean_data.index[0], 'Distance']
            rssi_d0 = anchor_mean_data.loc[anchor_mean_data.index[0], 'RSSI']
            anchor_mean_data['Dist_RSSI'] = d0 * np.power(10, ((rssi_d0 - anchor_mean_data["RSSI"]) / (10 * alpha)))

            # Store the mean values for each anchor
            mean_data[anchor_id] =anchor_mean_data


        # Set all the data to the same size
        total_row = max([len(df) for df in mean_data.values()])
        
        for i, df in enumerate(df for df in mean_data.values()):
            if len(df) < total_row:
                rows_to_add = total_row - len(df)
                df = pd.concat([df, pd.DataFrame(np.nan, index=range(rows_to_add), columns=df.columns)], ignore_index=True)
            df.interpolate(method   ='linear', inplace=True)
    
        #######################################################################################################
        ''' Main Logic : Simulate each localization method and calculate the error '''

        def calculate_distance_error(dataframe: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
            errors = np.sqrt((dataframe.loc[:, 'Xreal'] - dataframe.loc[:, 'Xest'])**2 + (dataframe.loc[:, 'Yreal'] - dataframe.loc[:, 'Yest'])**2)
            
            df_error = pd.DataFrame({'Error': errors})
            mean_error = errors.mean()

            return mean_error, df_error

        '''
        1. Estimate position by Multilateration (RSSI)
        '''
        df_pos_multilateration = multilateration(config, mean_data)
        df_pos_multilateration['Xest'] = df_pos_multilateration['Xest'].astype(float) 
        df_pos_multilateration['Yest'] = df_pos_multilateration['Yest'].astype(float)

        mean_error_posMLT, df_all_error_posMLT = calculate_distance_error(df_pos_multilateration)

        all_pos_multilateration.append(df_pos_multilateration)
        all_errors_multilateration.append(df_all_error_posMLT)
        mean_errors_multilateration.append(mean_error_posMLT)        
     
        '''
        2. Estimate position by AoA + RSSI (Trigonometry)
        '''
        df_pos_trigonometry = trigonometry(mean_data, config, df_pos_multilateration)
        df_pos_trigonometry['Xest'] = df_pos_trigonometry['Xest'].astype(float) 
        df_pos_trigonometry['Yest'] = df_pos_trigonometry['Yest'].astype(float)
        
        mean_error_posTrigonometry, df_all_error_posTrigonometry= calculate_distance_error(df_pos_trigonometry)
        
        all_pos_trigonometry.append(df_pos_trigonometry)
        all_errors_trigonometry.append(df_all_error_posTrigonometry)
        mean_errors_trigonometry.append(mean_error_posTrigonometry)

        '''
        3. Estimate position by AoA-only (Triangulation)
        '''
        df_pos_triangulation = triangulation(mean_data, config)
        df_pos_triangulation['Xest'] = df_pos_triangulation['Xest'].astype(float)
        df_pos_triangulation['Yest'] = df_pos_triangulation['Yest'].astype(float)

        mean_error_posTriangulation, df_all_error_posTriangulation= calculate_distance_error(df_pos_triangulation)

        all_pos_triangulation.append(df_pos_triangulation)
        all_errors_triangulation.append(df_all_error_posTriangulation)
        mean_errors_triangulation.append(mean_error_posTriangulation)

        # Announce re-use of the same data
        zk_pos_multilateration = create_measurement_matrix(df_pos_multilateration)
        zk_pos_trigonometry = create_measurement_matrix(df_pos_trigonometry)
        zk_pos_triangulation = create_measurement_matrix(df_pos_triangulation)

        '''
        4. Estimate positions using the results from previously methods with Kalman Filter
        '''
        xk_multilateration = kalman_filter(zk_pos_multilateration, config, initial_position, config['kalman_filter']['R_MLT'])
        df_pos_multilateration_KF = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) #DF to save results
        df_pos_multilateration_KF['Xreal'] = df_pos_multilateration['Xreal']
        df_pos_multilateration_KF['Yreal'] = df_pos_multilateration['Yreal']
        df_pos_multilateration_KF['Xest'] = xk_multilateration[:,0]
        df_pos_multilateration_KF['Yest'] = xk_multilateration[:,2]
        
        mean_error_posMLT_KF, df_all_error_posMLT_KF = calculate_distance_error(df_pos_multilateration_KF)
        
        all_pos_multilateration_KF.append(df_pos_multilateration_KF)
        all_errors_multilateration_KF.append(df_all_error_posMLT_KF)
        mean_errors_multilateration_KF.append(mean_error_posMLT_KF)
        
        """
        5. Positioning obtained with Trigonometry + Kalman filter
        """
        xk_trigonometry = kalman_filter(zk_pos_trigonometry, config, initial_position, config['kalman_filter']['R_AoA_RSSI'])
        df_pos_trigonometry_KF = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) #DF to save results
        df_pos_trigonometry_KF['Xreal'] = df_pos_trigonometry['Xreal']
        df_pos_trigonometry_KF['Yreal'] = df_pos_trigonometry['Yreal']
        df_pos_trigonometry_KF['Xest'] = xk_trigonometry[:,0]
        df_pos_trigonometry_KF['Yest'] = xk_trigonometry[:,2]
        
        mean_error_posTrigonometry_KF, df_all_error_posTrigonometry_KF = calculate_distance_error(df_pos_trigonometry_KF)
        
        all_pos_trigonometry_KF.append(df_pos_trigonometry_KF)  
        all_errors_trigonometry_KF.append(df_all_error_posTrigonometry_KF)
        mean_errors_trigonometry_KF.append(mean_error_posTrigonometry_KF)

        """
        6. Positioning obtained with Triangulation + Kalman filter
        """
        xk_triangulation = kalman_filter(zk_pos_triangulation, config, initial_position, config['kalman_filter']['R_AoA_only'])
        df_pos_triangulation_KF = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) #DF to save results
        df_pos_triangulation_KF['Xreal'] = df_pos_triangulation['Xreal']
        df_pos_triangulation_KF['Yreal'] = df_pos_triangulation['Yreal']
        df_pos_triangulation_KF['Xest'] = xk_triangulation[:,0]
        df_pos_triangulation_KF['Yest'] = xk_triangulation[:,2]
        
        mean_error_posTriangulation_KF, df_all_error_posTriangulation_KF = calculate_distance_error(df_pos_triangulation_KF)
        
        all_pos_triangulation_KF.append(df_pos_triangulation_KF)
        all_errors_triangulation_KF.append(df_all_error_posTriangulation_KF)
        mean_errors_triangulation_KF.append(mean_error_posTriangulation_KF)

        '''
        7. Estimate position using ARFL fusion with AoA+RSSI (Trigonometry) and AoA-only (Triangulation)
        '''
        xk_ARFL = ARFL_fusion(zk_pos_trigonometry, zk_pos_triangulation, config['kalman_filter']['R_AoA_RSSI'] , config['kalman_filter']['R_AoA_only'], initial_position, config)
        
        df_posARFL = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) #DF to save results
        df_posARFL['Xreal'] = df_pos_triangulation['Xreal']
        df_posARFL['Yreal'] = df_pos_triangulation['Yreal']
        df_posARFL['Xest'] = xk_ARFL[:,0]
        df_posARFL['Yest'] = xk_ARFL[:,2]
        
        mean_error_posARFL, df_all_error_posARFL = calculate_distance_error(df_posARFL)

        all_pos_ARFL.append(df_posARFL)
        all_errors_ARFL.append(df_all_error_posARFL)
        mean_errors_ARFL.append(mean_error_posARFL)

    #######################################################################################################
    ''' Plot real and estimated positions '''

    fig_width = config['plot']['fig_size'][0]
    fig_height = config['plot']['fig_size'][1]
    
    # Create a 2x2 subplot layout  
    _fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))

    for run in range(0, 4):
        row, col = divmod(run, 2)  # Determine subplot position
        ax = axes[row, col]

        plt.xlim(-2.00, 14.00)
        plt.ylim(-2.00, 8.00)

        ax.plot([0, 12, 12, 0, 0], [0, 0, 6, 6, 0], 'k', label='Boundary')
        ax.grid(which='major', color='#DDDDDD', linewidth=1) # major grid
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.8) # minor grid
        ax.minorticks_on() # enable minor ticks

        # mark anchors
        ax.plot([0, 6, 12, 6], [3, 0, 3, 6], 'ro', markersize=8, label='Anchors')
        ax.text(-1.30, 2.85, '6501')
        ax.text(5.50, -0.50, '6502')
        ax.text(12.25, 2.85, '6503')
        ax.text(5.50, 6.20, '6504')
        
        df_pos_multilateration_KF = all_pos_multilateration_KF[run]
        df_pos_trigonometry_KF = all_pos_trigonometry_KF[run]
        df_pos_triangulation_KF = all_pos_triangulation_KF[run]
        df_pos_ARFL = all_pos_ARFL[run]

        for i in range(0, len( df_pos_multilateration_KF) ,3):
            # Plot real trajectory
            ax.plot( df_pos_multilateration_KF.loc[i,'Xreal']/100,  df_pos_multilateration_KF.loc[i,'Yreal']/100, 'b*', label='Real Trajectory')

            # Plot estimated positions
            ax.plot(df_pos_multilateration_KF.loc[i,'Xest']/100,df_pos_multilateration_KF.loc[i,'Yest']/100, 'y.', label='RSSI') 
            ax.plot(df_pos_trigonometry_KF.loc[i,'Xest']/100,df_pos_trigonometry_KF.loc[i,'Yest']/100, 'c.', label='RSSI+AoA') 
            ax.plot(df_pos_triangulation_KF.loc[i,'Xest']/100,df_pos_triangulation_KF.loc[i,'Yest']/100, 'm.', label='AoA') 
            ax.plot(df_pos_ARFL.loc[i,'Xest']/100,df_pos_ARFL.loc[i,'Yest']/100, 'g.', label='ARFL')

            # Plot Line between real and estimated positions
            ax.plot([ df_pos_multilateration_KF.loc[i,'Xreal']/100,df_pos_multilateration_KF.loc[i,'Xest']/100], [ df_pos_multilateration_KF.loc[i,'Yreal']/100,df_pos_multilateration_KF.loc[i,'Yest']/100],'y', linewidth=0.2)
            ax.plot([ df_pos_multilateration_KF.loc[i,'Xreal']/100,df_pos_trigonometry_KF.loc[i,'Xest']/100], [ df_pos_multilateration_KF.loc[i,'Yreal']/100,df_pos_trigonometry_KF.loc[i,'Yest']/100],'c', linewidth=0.2)
            ax.plot([ df_pos_multilateration_KF.loc[i,'Xreal']/100,df_pos_triangulation_KF.loc[i,'Xest']/100], [ df_pos_multilateration_KF.loc[i,'Yreal']/100,df_pos_triangulation_KF.loc[i,'Yest']/100],'m', linewidth=0.2) 
            ax.plot([ df_pos_multilateration_KF.loc[i,'Xreal']/100,df_pos_ARFL.loc[i,'Xest']/100], [ df_pos_multilateration_KF.loc[i,'Yreal']/100,df_pos_ARFL.loc[i,'Yest']/100],'g', linewidth=0.2)

        ax.legend(['Area', 'Anchors', 'Real Trajectory', 'RSSI', 'RSSI+AoA', 'AoA', 'ARFL'],loc=1, fontsize='small')
        
        # Set subplot labels and title
        ax.set_title(f'Case: {case} - Run: {run} -- Position Estimation')
        ax.set_xlabel('x [m]', loc='right', fontsize = 12)
        ax.set_ylabel('y [m]', loc='top', fontsize = 12)

        ax.set_xlim(-2, 14)  # Set X-axis limit
        ax.set_ylim(-2, 8)  # Set Y-axis limit

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
        
        
    ###############################################################################################################################
    '''Print the results of the case'''

    # RESULTS WITHOUT KALMAN FILTER
    overall_mean_error_MLT = round(sum(mean_errors_multilateration) / len(mean_errors_multilateration), 4)
    print(f"Mean Err Multilateration (RSSI): {overall_mean_error_MLT}")

    overall_mean_error_Trigonometry = round(sum(mean_errors_trigonometry) / len(mean_errors_trigonometry), 4)
    print(f"Mean Err Trigonometry (AoA + RSSI): {overall_mean_error_Trigonometry}")

    overall_mean_error_Triangulation = round(sum(mean_errors_triangulation) / len(mean_errors_triangulation), 4)
    print(f"Mean Err Triangulation (AoA): {overall_mean_error_Triangulation}")

    # RESULTS WITH KALMAN FILTER
    overall_mean_error_MLT_KF = round(sum(mean_errors_multilateration_KF) / len(mean_errors_multilateration_KF), 4)
    print(f"Mean Err Multilateration (RSSI) + KF: {overall_mean_error_MLT_KF}")
    
    overall_mean_error_Trigonometry_KF = round(sum(mean_errors_trigonometry_KF) / len(mean_errors_trigonometry_KF), 4)
    print(f"Mean Err Trigonometry (AoA + RSSI) + KF: {overall_mean_error_Trigonometry_KF}")

    overall_mean_error_Triangulation_KF = round(sum(mean_errors_triangulation_KF) / len(mean_errors_triangulation_KF), 4)
    print(f"Mean Err Triangulation (AoA) + KF: {overall_mean_error_Triangulation_KF}")

    # RESULTS WITH ARFL FUSION
    overall_mean_error_ARFL = round(sum(mean_errors_ARFL) / len(mean_errors_ARFL), 4)
    print(f"Mean Err ARFL: {overall_mean_error_ARFL}\n")

    
    ##############################################################################
    '''Plot Error Bar Graph'''

    # Set figure size based on the aspect ratio
    _fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Define methods and corresponding results
    methods = ['MLT', 'MLT+KF', 'AoA+RSSI', 'AoA+RSSI+KF', 'AoA', 'AoA+KF', 'ARFL']
    results = [
        overall_mean_error_MLT, overall_mean_error_MLT_KF,
        overall_mean_error_Trigonometry, overall_mean_error_Trigonometry_KF,
        overall_mean_error_Triangulation, overall_mean_error_Triangulation_KF,
        overall_mean_error_ARFL
    ]

    # Define bar colors and labels
    bar_info = [
        ('tab:red', 'Without Filter'),
        ('tab:blue', 'With KF'),
        ('tab:red', '_Without Filter'),
        ('tab:blue', '_White KF'),
        ('tab:red', '_Without Filter'),
        ('tab:blue', '_KF'),
        ('tab:green', 'ARFL')
    ]

    # Set X-axis labels with rotation for better readability
    plt.xticks(range(len(methods)), methods, rotation=30, fontsize=11)

    # Plot the bar graph
    for method, result, (color, label) in zip(methods, results, bar_info):
        ax.bar(method, result, color=color, label=label)

    # Annotate bars with their respective values
    for i, result in enumerate(results):
        plt.text(i, result / 2, f"{result:.2f}", ha='center', va='center')

    # Set labels and title
    ax.set_ylabel('Distance Error [m]', fontsize=12)
    ax.set_title(f'Case: {case} -- Distance Error x Method', fontsize=14)

    # Remove duplicate labels from the legend
    handles, labels = ax.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    ax.legend(unique_legend.values(), unique_legend.keys(), title='Method', title_fontsize=12)

    plt.tick_params(axis='y', labelsize=12)
    plt.show()
                

    ##############################################################################
    '''Plot Cumulative Distribution Function (CDF) for KF'''

    # Concatenate all DataFrames of errors into a single DataFrame
    df_error_multilateration_KF_all = pd.concat(all_errors_multilateration, ignore_index=True)
    df_error_Trigonometry_KF_all = pd.concat(all_errors_trigonometry_KF, ignore_index=True)
    df_error_Triangulation_KF_all = pd.concat(all_errors_triangulation_KF, ignore_index=True)
    df_error_ARFL_all = pd.concat(all_errors_ARFL, ignore_index=True)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Configure Y-axis limits and set percentile values
    ax.set_ylim(0, 1)
    percentile_values = [0.05, 0.25, 0.5, 0.75, 0.95]
    ax.set_yticks(percentile_values)

    # Function to plot cumulative histogram (CDF)
    def plot_cumulative_hist(ax, data, color, label):
        """
        Plots the cumulative distribution function (CDF) for given data.
        
        Parameters:
            ax (matplotlib.axes._subplots.AxesSubplot): Axis object to plot on.
            data (array-like): Data values to create histogram.
            color (str): Color for the plot.
            label (str): Label for the plot legend.
        """
        # Compute histogram and cumulative distribution
        counts, bin_edges = np.histogram(data / 100, bins=100, density=True)
        cdf = np.cumsum(counts) / np.sum(counts)

        # Plot the cumulative distribution curve
        ax.step(bin_edges[:-1], cdf, where='post', color=color, label=label, linewidth=2)

        # Plot horizontal percentile lines
        for p in percentile_values:
            x_value = np.percentile(data, p) / 100  # Convert to match x-axis scale
            y_value = p / 100.0  # Convert to probability scale
            ax.hlines(y=y_value, xmin=0, xmax=x_value, color='grey', linestyle='--', linewidth=1)

    # List of datasets, colors, and labels for plotting
    cdf_data = [
        (df_error_ARFL_all, 'green', 'ARFL'),
        (df_error_Triangulation_KF_all, 'purple', 'AoA+KF'),
        (df_error_Trigonometry_KF_all, 'black', 'AoA+RSSI+KF'),
        (df_error_multilateration_KF_all, 'blue', 'MLT+KF')
    ]

    # Plot all cumulative histograms
    for data, color, label in cdf_data:
        plot_cumulative_hist(ax, data, color, label)

    # Set labels and title
    ax.set_xlabel('Distance Error [m]', fontsize=14)
    ax.set_ylabel('Cumulative Probability', fontsize=14)
    ax.set_title(f'Case: {case} -- Cumulative Distribution Error', fontsize=14)
    ax.set_xlim(0, 3)  # Set X-axis limit
    ax.set_ylim(0, 1)  # Set Y-axis limit

    # Add grid to Y-axis only
    ax.grid(True, axis='y')

    # Create and add legend
    legend_lines = [Line2D([0], [0], color=color, lw=2) for _, color, _ in cdf_data]
    legend_labels = [label for _, _, label in cdf_data]
    ax.legend(legend_lines, legend_labels, handlelength=2, loc=4, fontsize=14)

    # Set tick label sizes
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)

    # Show the plot
    plt.show()

    
        