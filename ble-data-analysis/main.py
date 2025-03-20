import os
import sys
import yaml
import pandas as pd
import aoa
import rssi

dataset_dir = "dataset"

def main(type: str):
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    gt_path = os.path.join(dataset_dir, f"gt/{type}.csv")
    gt_df = pd.read_csv(gt_path)

    all_anchor_error_df = []

    for anchor_name in os.listdir(dataset_dir):
        if not anchor_name.startswith("anchor"):
            continue

        anchor_path = os.path.join(dataset_dir, f"{anchor_name}/{type}.csv")
        if not os.path.exists(anchor_path):
            raise FileNotFoundError(f"{anchor_path} does not exist")

        anchor_df = pd.read_csv(anchor_path)

        anchor_error_df = aoa.calculate_error(gt_df, anchor_df, config[type][anchor_name])
        # anchor_error_df = rssi.calculate_error(anchor_path, gt_path)

        all_anchor_error_df.append(anchor_error_df)

    aoa.visualize_error(all_anchor_error_df)
    # rssi.visualize_error(type)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <type> <channel>")
        sys.exit(1)

    arg1 = sys.argv[1]

    main(arg1)