import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate

def visualize_error(all_error_df):
    num_dfs = len(all_error_df)
    cols = 4
    rows = (num_dfs + (cols // 2) - 1) // (cols // 2) 

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows)) 
    axes = axes.flatten() 

    for idx, error_df in enumerate(all_error_df):
        ax = axes[idx]

        # Group Data
        grouped = error_df.groupby(["X", "Y"])["AzimuthError"].agg(["mean", "var"]).reset_index()
        grouped["AzimuthErrorAbs"] = np.abs(grouped["mean"]) 

        # Extract Data
        X = grouped["X"].values
        Y = grouped["Y"].values
        Error_mean = grouped["AzimuthErrorAbs"].values  # 절댓값 평균
        Error_var = grouped["var"].values  # 분산

        # Grid for interpolation
        grid_x, grid_y = np.meshgrid(
            np.linspace(X.min(), X.max(), 200),
            np.linspace(Y.min(), Y.max(), 200)
        )

        # Cubic-Interpolate
        grid_z_mean = scipy.interpolate.griddata((X, Y), Error_mean, (grid_x, grid_y), method="cubic")
        grid_z_var = scipy.interpolate.griddata((X, Y), Error_var, (grid_x, grid_y), method="cubic")

        # Heatmap (Mean Error)
        ax1 = axes[idx * 2]
        contour1 = ax1.imshow(grid_z_mean, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                              origin="lower", cmap="jet", aspect="auto", vmin=0, vmax=15)
        cbar1 = plt.colorbar(contour1, ax=ax1)
        cbar1.set_label("Azimuth Error (Abs)")
        ax1.set_xlabel("[cm]")
        ax1.set_ylabel("[cm]")
        ax1.set_xlim(0, 600)
        ax1.set_ylim(0, 480)
        ax1.set_title(f"Dataset {idx + 1} - Mean Error")
        ax1.scatter(X, Y, c='red', s=5, label="Data Points")
        ax1.legend()

        # Heatmap (Variance)
        ax2 = axes[idx * 2 + 1]
        contour2 = ax2.imshow(grid_z_var, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                              origin="lower", cmap="jet", aspect="auto", vmin=0, vmax=20)
        cbar2 = plt.colorbar(contour2, ax=ax2)
        cbar2.set_label("Azimuth Error Variance")
        ax2.set_xlabel("[cm]")
        ax2.set_ylabel("[cm]")
        ax2.set_xlim(0, 600)
        ax2.set_ylim(0, 480)
        ax2.set_title(f"Dataset {idx + 1} - Variance")
        ax2.scatter(X, Y, c='red', s=5, label="Data Points")
        ax2.legend()

    # Remove unused axes
    for i in range(num_dfs * 2, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def calculate_error(gt_df, ms_df, anchor_config ):
    position = anchor_config["position"]   
    orientation = anchor_config["orientation"]

    filtered_data = []

    for _, row in gt_df.iterrows():
        start_time_iso, start_timestamp, end_time_iso, end_timestamp, x, y, z, obstacle = row

        if obstacle != 4:
            continue

        # calculate ground truth azimuth and elevation
        azimuth_gt = math.atan((x - position[0]) / (y - position[1])) - math.radians(orientation[2])
        while azimuth_gt < -math.pi / 2:
            azimuth_gt += math.pi
        azimuth_gt = math.degrees(azimuth_gt)

        # calculate ground truth elevation
        elevation_gt = math.atan((z - position[2]) / math.sqrt( (x - position[0]) ** 2 + (y - position[1]) ** 2))
        elevation_gt = math.degrees(elevation_gt)

        filtered_df = ms_df[
            (ms_df["Timestamp"] >= start_timestamp) & (ms_df["Timestamp"] <= end_timestamp)
        ].copy()
        filtered_df["X"] = x
        filtered_df["Y"] = y
        filtered_df["Z"] = z
        filtered_df["AzimuthGT"] = azimuth_gt
        filtered_df["ElevationGT"] = elevation_gt      

        filtered_df["AzimuthError"] = azimuth_gt - filtered_df["Azimuth"]
        filtered_df["ElevationError"] = elevation_gt- filtered_df["Elevation"]

        filtered_data.append(filtered_df)
        
    result_df = pd.concat(filtered_data, ignore_index=True) if filtered_data else pd.DataFrame()
    return result_df