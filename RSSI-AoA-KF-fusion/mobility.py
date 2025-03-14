import yaml
import pandas as pd
import numpy as np
import data_processing as dP
import matplotlib.pyplot as plt
import findPositions as fP
from matplotlib.lines import Line2D 

beacons_column_names = ["TimeStamp", "TagID", "1stP","AoA_az", "AoA_el", "2ndP", "Channel", "AnchorID"]  # Column IDs for Beacons Dataset
gt_column_names = ["StartTime", "EndTime", "Xcoord","Ycoord"] # Column IDs for Ground Truth Dataset

def run(case):
    print('Start Mobility Process')
    
    # Open Configuration File
    with open("config-mobility.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    initial_position = config['additional'][f'case_{case}']['initial_position']

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


    # Vector to save results of each method
    all_errors_MLT = []
    mean_errors_MLT = []
    all_errors_Trigonometry = []
    mean_errors_Trigonometry = []
    all_errors_Triangulation = []
    mean_errors_Triangulation  = []
    all_errors_MLT_KF = []
    mean_errors_MLT_KF = []
    all_errors_Trigonometry_KF = []
    mean_errors_Trigonometry_KF = []
    all_errors_Triangulation_KF = []
    mean_errors_Triangulation_KF = []
    all_errors_ARFL = []
    mean_errors_ARFL = []
    
    for run in range(1, 2):
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
            anchor_mean_data['Dest_RSSI'] = d0 * np.power(10, ((rssi_d0 - anchor_mean_data["RSSI"]) / (10 * alpha)))

            # Store the mean values for each anchor
            mean_data[anchor_id] =anchor_mean_data


        # Set all the data to the same size...
        total_row = max([len(df) for df in mean_data.values()])
        
        for i, df in enumerate(df for df in mean_data.values()):
            if len(df) < total_row:
                rows_to_add = total_row - len(df)
                df = pd.concat([df, pd.DataFrame(np.nan, index=range(rows_to_add), columns=df.columns)], ignore_index=True)
            df.interpolate(method   ='linear', inplace=True)
    
        # Simulate with different methods
        
        '''
        1. Estimate position by Multilateration(RSSI)
        '''
        df_posMLT = fP.multilateration(config, mean_data)
        df_posMLT['Xest'] = df_posMLT['Xest'].astype(float) 
        df_posMLT['Yest'] = df_posMLT['Yest'].astype(float)

        mean_error_posMLT, df_all_error_posMLT= dP.distance_error(df_posMLT)

        all_errors_MLT.append(df_all_error_posMLT)
        mean_errors_MLT.append(mean_error_posMLT)
        
        print(f'Mean Err MLT(RSSI): {mean_errors_MLT}')
        
     
        '''
        2. Estimate position by AoA + RSSI (Trigonometry)
        '''
        df_posTrigonometry = fP.trigonometry(mean_data, config, df_posMLT)
        df_posTrigonometry['Xest'] = df_posTrigonometry['Xest'].astype(float) 
        df_posTrigonometry['Yest'] = df_posTrigonometry['Yest'].astype(float)
        
        mean_error_posTrigonometry, df_all_error_posTrigonometry= dP.distance_error(df_posTrigonometry)
        
        all_errors_Trigonometry.append(df_all_error_posTrigonometry)
        mean_errors_Trigonometry.append(mean_error_posTrigonometry)
        
        print(f'Mean Err Trigonometry(AoA + RSSI): {mean_errors_Trigonometry}')

 
        '''
        3. Estimate position by AoA-only (Triangulation)
        '''
        df_posTriangulation = fP.triangulation(mean_data, config)
        df_posTriangulation['Xest'] = df_posTriangulation['Xest'].astype(float)
        df_posTriangulation['Yest'] = df_posTriangulation['Yest'].astype(float)

        mean_error_posTriangulation, df_all_error_posTriangulation= dP.distance_error(df_posTriangulation)

        all_errors_Triangulation.append(df_all_error_posTriangulation)
        mean_errors_Triangulation.append(mean_error_posTriangulation)

        print(f'Mean Err Triangulation(AoA): {mean_errors_Triangulation}')

        # Announce re-use of the same data
        zk_posMLT = dP.create_measurement_matrix(df_posMLT)
        zk_posTrigonometry = dP.create_measurement_matrix(df_posTrigonometry)
        zk_posTriangulation = dP.create_measurement_matrix(df_posTriangulation)

        # '''
        # 4. Estimate positions using the results from previously methods with Kalman Filter
        # '''
        # # Create measurements matrices using the helper function
        
        # # Positioning obtained with multilateration + Kalman filter
        # xk_MLT = fP.kalman_filter(zk_posMLT, config, initial_position, config['kalman_filter']['R_MLT'])
        # df_posMLT_KF = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) #DF to save results
        # df_posMLT_KF['Xreal'] = df_posMLT['Xreal']
        # df_posMLT_KF['Yreal'] = df_posMLT['Yreal']
        # df_posMLT_KF['Xest'] = xk_MLT[:,0]
        # df_posMLT_KF['Yest'] = xk_MLT[:,2]
        
        # #Calculate the error
        # mean_error_posMLT_KF, df_all_error_posMLT_KF = dP.distance_error(df_posMLT_KF)
        
        # # Append the DataFrame of errors to the list - used later to get the CDF
        # all_errors_MLT_KF.append(df_all_error_posMLT_KF)
        
        # # Append the average errors to the list
        # mean_errors_MLT_KF.append(mean_error_posMLT_KF)
        
        # ########################################################################################################################3
        # """
        # Positioning obtained with Trigonometry + Kalman filter
        # """
        # print('initial_position', initial_position)
        # xk_Trigonometry = fP.kalman_filter(zk_posTrigonometry, config, initial_position, config['kalman_filter']['R_AoA_RSSI'])
        # df_posTrigonometry_KF = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) #DF to save results
        # df_posTrigonometry_KF['Xreal'] = df_posTrigonometry['Xreal']
        # df_posTrigonometry_KF['Yreal'] = df_posTrigonometry['Yreal']
        # df_posTrigonometry_KF['Xest'] = xk_Trigonometry[:,0]
        # df_posTrigonometry_KF['Yest'] = xk_Trigonometry[:,2]
        
        # #Calculate the error
        # mean_error_posTrigonometry_KF, df_all_error_posTrigonometry_KF = dP.distance_error(df_posTrigonometry_KF)
        
        # # Append the DataFrame of errors to the list - used later to get the CDF
        # all_errors_Trigonometry_KF.append(df_all_error_posTrigonometry_KF)
        
        # # Append the average errors to the list
        # mean_errors_Trigonometry_KF.append(mean_error_posTrigonometry_KF)

        """
        6. Positioning obtained with Triangulation + Kalman filter
        """
        xk_Triangulation = fP.kalman_filter(zk_posTriangulation, config, initial_position, config['kalman_filter']['R_AoA_only'])
        df_posTriangulation_KF = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) #DF to save results
        df_posTriangulation_KF['Xreal'] = df_posTriangulation['Xreal']
        df_posTriangulation_KF['Yreal'] = df_posTriangulation['Yreal']
        df_posTriangulation_KF['Xest'] = xk_Triangulation[:,0]
        df_posTriangulation_KF['Yest'] = xk_Triangulation[:,2]
        
        mean_error_posTriangulation_KF, df_all_error_posTriangulation_KF = dP.distance_error(df_posTriangulation_KF)
        
        all_errors_Triangulation_KF.append(df_all_error_posTriangulation_KF)
        mean_errors_Triangulation_KF.append(mean_error_posTriangulation_KF)

        print(f'Mean Err Triangulation(AoA) + KF: {mean_errors_Triangulation_KF}')

        """
        6-1. Positioning obtained with Triangulation + Kalman filter acceleration
        """
        xk_Triangulation_accel = fP.kalman_filter_acceleration(zk_posTriangulation, config, initial_position, config['kalman_filter_acceleration']['R'])
        df_posTriangulation_KF_accel = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) #DF to save results
        df_posTriangulation_KF_accel['Xreal'] = df_posTriangulation['Xreal']
        df_posTriangulation_KF_accel['Yreal'] = df_posTriangulation['Yreal']
        df_posTriangulation_KF_accel['Xest'] = xk_Triangulation_accel[:,0]
        df_posTriangulation_KF_accel['Yest'] = xk_Triangulation_accel[:,3]
        
        mean_error_posTriangulation_KF_accel, df_all_error_posTriangulation_KF_accel = dP.distance_error(df_posTriangulation_KF_accel)

        print(f'Mean Err Triangulation(AoA) + KF_accel: {mean_error_posTriangulation_KF_accel}')


        """
        7-2. Positioning obtained with Triangulation + Extended Kalman filter
        """
        # xk_Triangulation = fP.kalman_filter_acceleration(zk_posTriangulation, config, initial_position, config['kalman_filter_acceleration']['R'])
        # df_posTriangulation_KF = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) #DF to save results
        # df_posTriangulation_KF['Xreal'] = df_posTriangulation['Xreal']
        # df_posTriangulation_KF['Yreal'] = df_posTriangulation['Yreal']
        # df_posTriangulation_KF['Xest'] = xk_Triangulation[:,0]
        # df_posTriangulation_KF['Yest'] = xk_Triangulation[:,2]
        
        # mean_error_posTriangulation_KF_accel, df_all_error_posTriangulation_KF_accel = dP.distance_error(df_posTriangulation_KF)

        # print(f'Mean Err Triangulation(AoA) + KF: {mean_error_posTriangulation_KF_accel}')


        # ########################################################################################################################
        # '''Estimate position using ARFL fusion with AoA+RSSI (Trigonometry) and AoA-only (Triangulation)'''
        # xk_ARFL = fP.ARFL_fusion(zk_posTrigonometry, zk_posTriangulation, config['kalman_filter']['R_AoA_RSSI'] , config['kalman_filter']['R_AoA_only'], initial_position, config)
        
        # df_posARFL = pd.DataFrame(columns=['Xreal', 'Yreal', 'Xest', 'Yest']) #DF to save results
        # df_posARFL['Xreal'] = df_posTriangulation['Xreal']
        # df_posARFL['Yreal'] = df_posTriangulation['Yreal']
        # df_posARFL['Xest'] = xk_ARFL[:,0]
        # df_posARFL['Yest'] = xk_ARFL[:,2]
        
        # #Calculate the error
        # mean_error_posARFL, df_all_error_posARFL = dP.distance_error(df_posARFL)
        
        # # Append the DataFrame of errors to the list - used later to get the CDF
        # all_errors_ARFL.append(df_all_error_posARFL)
        
        # # Append the average errors to the list
        # mean_errors_ARFL.append(mean_error_posARFL)
        
        
        #######################################################################################################
        '''Plot real and estimated positions'''
        plt.figure()
        plt.title(f'Case: {case} - Run: {run} -- Position Estimation')

        plt.xlim(-2.00,14.00)
        plt.xlabel('x [m]', loc='right', fontsize = 12)

        plt.ylim(-2.00, 8.00)
        plt.ylabel('y [m]', loc='top', fontsize = 12)

        plt.plot([0,12.00,12.00,0,0],[0,0,6.00,6.00,0],'k', label='Boundary')
        plt.grid(which='major', color='#DDDDDD', linewidth=1) # major grid
        plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.8) # minor grid
        plt.minorticks_on() # enable minor ticks

        # mark anchors
        plt.plot([0,6.00,12.00,6.00],[3.00,0,3.00,6.00],'ro', markersize=8, label='Anchors')
        plt.text(-1.30,2.85,'6501')
        plt.text(5.50,-0.50,'6502')
        plt.text(12.25,2.85,'6503')
        plt.text(5.50,6.20,'6504')
        
        for i in range (0,len(df_posMLT),3):
            plt.plot(df_posMLT.loc[i,'Xreal']/100, df_posMLT.loc[i,'Yreal']/100, 'b*', label='Real Trajectory')

            plt.plot(df_posTrigonometry.loc[i,'Xest']/100,df_posTrigonometry.loc[i,'Yest']/100, 'y.', label='RSSI') 
            plt.plot(df_posTriangulation.loc[i,'Xest']/100,df_posTriangulation.loc[i,'Yest']/100, 'c.', label='AoA') 
            plt.plot(df_posTriangulation_KF.loc[i,'Xest']/100,df_posTriangulation_KF.loc[i,'Yest']/100, 'm.', label='AoA+KF') 

            plt.plot([df_posMLT.loc[i,'Xreal']/100,df_posTrigonometry.loc[i,'Xest']/100], [df_posMLT.loc[i,'Yreal']/100,df_posTrigonometry.loc[i,'Yest']/100],'y', linewidth=0.2)
            plt.plot([df_posMLT.loc[i,'Xreal']/100,df_posTriangulation.loc[i,'Xest']/100], [df_posMLT.loc[i,'Yreal']/100,df_posTriangulation.loc[i,'Yest']/100],'c', linewidth=0.2)
            plt.plot([df_posMLT.loc[i,'Xreal']/100,df_posTriangulation_KF.loc[i,'Xest']/100], [df_posMLT.loc[i,'Yreal']/100,df_posTriangulation_KF.loc[i,'Yest']/100],'m', linewidth=0.2) 

        
        plt.legend(['Area', 'Anchors', 'Real Trajectory', 'RSSI', 'AoA', 'AoA+KF'],loc=1, fontsize='small')

        plt.show(block=True)
        
        
    ###############################################################################################################################
    
    # # Calculate the overall mean average error
    # print(f'\nRESULTS WITHOUT KALMAN FILTER')
    # overall_mean_error_MLT = round(sum(mean_errors_MLT) / (len(mean_errors_MLT)*100), 2) #transform to meters and round
    # print(f"Multilateration Average Error across all runs: {overall_mean_error_MLT}")
    # overall_mean_error_Trigonometry = round(sum(mean_errors_Trigonometry) / (len(mean_errors_Trigonometry)*100), 2) #transform to meters and round
    # print(f"AoA+RSSI Average Error across all runs: {overall_mean_error_Trigonometry}")
    # overall_mean_error_Triangulation = round(sum(mean_errors_Triangulation) / (len(mean_errors_Triangulation)*100), 2) #transform to meters and round
    # print(f"AoA-only Average Error across all runs: {overall_mean_error_Triangulation}")
    # overall_mean_error_MLT_KF = round(sum(mean_errors_MLT_KF) / (len(mean_errors_MLT_KF)*100), 2) #transform to meters and round
    # print(f'\nRESULTS WITH KALMAN FILTER')
    # print(f"Multilateration+KF Average Error across all runs: {overall_mean_error_MLT_KF}")
    # overall_mean_error_Trigonometry_KF = round(sum(mean_errors_Trigonometry_KF) / (len(mean_errors_Trigonometry_KF)*100), 2) #transform to meters and round
    # print(f"AoA+RSSI+KF Average Error across all runs: {overall_mean_error_Trigonometry_KF}")
    # overall_mean_error_Triangulation_KF = round(sum(mean_errors_Triangulation_KF) / (len(mean_errors_Triangulation_KF)*100), 2) #transform to meters and round
    # print(f"AoA-only+KF Average Error across all runs: {overall_mean_error_Triangulation_KF}")
    # overall_mean_error_ARFL = round(sum(mean_errors_ARFL) / (len(mean_errors_ARFL)*100), 2) #transform to meters and round
    # print(f'\nRESULTS WITH FUSION')
    # print(f"ARFL Average Error across all runs: {overall_mean_error_ARFL}\n")
    
    
    # # Concatenate all DataFrames of errors into a single DataFrame
    # df_error_MLT_all = pd.concat(all_errors_MLT, ignore_index=True)
    # df_error_Trigonometry_all = pd.concat(all_errors_Trigonometry, ignore_index=True)
    # df_error_Triangulation_all = pd.concat(all_errors_Triangulation, ignore_index=True)
    # df_error_MLT_KF_all = pd.concat(all_errors_MLT, ignore_index=True)
    # df_error_Trigonometry_KF_all = pd.concat(all_errors_Trigonometry_KF, ignore_index=True)
    # df_error_Triangulation_KF_all = pd.concat(all_errors_Triangulation_KF, ignore_index=True)
    # df_error_ARFL_all = pd.concat(all_errors_ARFL, ignore_index=True)
    
    
    ##############################################################################
    '''Start of plots (Erro Bargraph and CDF)'''
    
    #  # Define the aspect ratio
    # aspect_ratio = 16 / 9
    # # Set the figure size based on the aspect ratio
    # fig_width = 8  # You can choose any width
    # fig_height = fig_width / aspect_ratio
    
    # if config['additional']['plot_error_bargraph'] == True:
    #     fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    #     methods = ['MLT', 'MLT+KF', 'AoA+RSSI', 'AoA+RSSI+KF', 'AoA', 'AoA+KF', 'ARFL']
    #     results = [overall_mean_error_MLT, overall_mean_error_MLT_KF, overall_mean_error_Trigonometry, overall_mean_error_Trigonometry_KF, overall_mean_error_Triangulation, overall_mean_error_Triangulation_KF, overall_mean_error_ARFL]
    #     bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:blue', 'tab:red', 'tab:blue', 'tab:green' ]
    #     bar_labels = ['Without Filter', 'With KF', '_Without Filter', '_White KF' ,'_Without Filter', '_KF', 'ARFL']
    #     plt.xticks(range(len(methods)),('MLT', 'MLT+KF', 'AoA+RSSI', 'AoA+RSSI+KF', 'AoA', 'AoA+KF', 'ARFL'), rotation=30,fontsize=11)
    #     ax.bar(methods, results, label=bar_labels, color=bar_colors)
    #     for i in range (len(methods)):
    #         plt.text(i, results[i]/2, f"{results[i]:.2f}", ha = 'center', va='center')

    #     # plt.plot(metodos, resultados, 'k', linewidth=2.0)
    #     ax.set_ylabel('Distance Error [m]', fontsize=12)
    #     ax.set_title(f'Case: {case} -- Distance Error x Method', fontsize=14)
    #     ax.legend(title='Method', title_fontsize=12)
    #     plt.tick_params(axis='y', labelsize=12)
    #     plt.show()
            
    # if config['additional']['plot_CDF'] == True:
    #     # Create a figure and axis
    #     fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
    #     # Config Y-limit and percentiles
    #     ax.set_ylim(0, 1)
    #     percentile_values = [0.05, 0.25, 0.5, 0.75, 0.95]
    #     ax.set_yticks(percentile_values)

    #     # Function do generate percentiles
    #     def plot_cumulative_hist(data, color, label):
    #         counts, bin_edges = np.histogram(data/100, bins=100, density=True)
    #         cdf = np.cumsum(counts) / np.sum(counts)  # Calculate cumulative distribution
    #         plt.step(bin_edges[:-1], cdf, where='post', color=color, label=label, linewidth=2)  # Skip the last bin edge

    #         # Calculate and annotate percentiles
    #         for p in percentile_values:
    #             # Get the corresponding x value (distance error)
    #             value = np.percentile(data, p) / 100  # Divide by 100 to match scaling in plot
                
    #             # Find the corresponding y value on the CDF (percentile in terms of probability)
    #             y_value = p / 100.0
                
    #             # Plot the horizontal line from the y-axis (at y_value) to the CDF curve at 'value'
    #             plt.hlines(y=y_value, xmin=0, xmax=value, color='grey', linestyle='--', linewidth=1)
                              
    #     # Plot the first cumulative histogram
    #     plot_cumulative_hist(df_error_ARFL_all, color='green', label='alg1')

    #     # Plot the second cumulative histogram
    #     plot_cumulative_hist(df_error_Triangulation_KF_all, color='purple', label='alg2')

    #     # Plot the third cumulative histogram
    #     plot_cumulative_hist(df_error_Trigonometry_KF_all, color='black', label='alg3')

    #     # Add labels and title
    #     plt.xlabel('Distance Error [m]', fontsize=14)
    #     plt.ylabel('Cumulative Probability', fontsize=14)
    #     plt.title(f'Case: {case} -- Cumulative Distribution Error', fontsize=14)
    #     plt.xlim(left=0, right=3)  # Ajuste limite do eixo x de 0 a 3
    #     plt.ylim(0, 1)  # Limite do eixo y de 0 a 1

    #     plt.grid(True, axis='y')  # Grade apenas no eixo y

    #     # Create custom legend lines (Line2D objects)
    #     custom_lines = [Line2D([0], [0], color='green', lw=2),
    #                     Line2D([0], [0], color='purple', lw=2),
    #                     Line2D([0], [0], color='black', lw=2)]

    #     # Add a legend using the custom lines
    #     plt.legend(custom_lines, ['ARFL', 'AoA+KF', 'AoA+RSSI+KF'], handlelength=2, loc=4, fontsize=14)
    #     plt.tick_params(axis='y', labelsize=12)
    #     plt.tick_params(axis='x', labelsize=12)

    #     # Show the plot
    #     plt.show()

    
        