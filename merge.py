import os
import pandas as pd

def merge_csv_files(folder_path, output_file=None):
    # List to store DataFrames
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = []

    # Iterate through all CSV files and read them into DataFrames
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate all DataFrames into one
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Optionally save the merged DataFrame to a new CSV file
    if output_file:
        merged_df.to_csv(output_file, index=False)
        print(f"Merged CSV saved to {output_file}")

    return merged_df

# Usage
folder_path = './'  # Replace with the folder path containing your CSV files
output_file = './merged'  # Optional: Replace with the desired output file path
merged_csv = merge_csv_files(folder_path, output_file)

# To inspect the merged DataFrame
print(merged_csv.head())  # Print the first few rows of the merged CSV

