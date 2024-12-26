
"""
import pandas as pd

def merge_csv_data(output_data,metrics_data):
    # Convert the string data into pandas DataFrames
    metrics_df = pd.read_csv(metrics_data)
    output_df = pd.read_csv(output_data)
    
    # Merge the dataframes on Video ID
    merged_df = pd.merge(output_df,metrics_df ,on=['Video ID', 'Sequence'], how='inner')
    
    # Sort by Sequence and Video ID for better readability
    # merged_df = merged_df.sort_values(['Sequence', 'Video ID'])
    merged_df.to_csv('combined-08-12-2024.csv', index=False)

    return merged_df

merge_csv_data("Compression bframes-ref-quantizer-key_int_max-bitrate_08-12-2024.csv","Metrics bframes-ref-quantizer-key_int_max-bitrate_08-12-2024.csv")
"""
import pandas as pd
import os
import glob
def merge_csv_data():
    """
    Merges corresponding Compression and Metrics CSV files from respective folders.
    Only creates new combined files if they don't already exist.
    """
    # Create combined directory if it doesn't exist
    os.makedirs("csv_files/combined", exist_ok=True)
    
    # Get all files from both directories
    output_data_files = glob.glob("csv_files/compression/Compression*.csv")
    metrics_data_files = glob.glob("csv_files/metrics/Metrics*.csv")
    
    if not output_data_files or not metrics_data_files:
        raise FileNotFoundError("Required files not found. Ensure files start with 'Compression' and 'Metrics' respectively.")
    
    merged_dataframes = []
    
    # Process each Compression file
    for output_data in output_data_files:
        output_file = output_data.split("/")[-1]
        # Extract the unique identifier part (everything after "Compression")
        file_identifier = output_file[12:]
        
        # Find corresponding Metrics file
        matching_metrics = [f for f in metrics_data_files if file_identifier in f]
        
        if not matching_metrics:
            print(f"No matching Metrics file found for {output_file}")
            continue
            
        metrics_data = matching_metrics[0]
        
        # Construct the output filename
        combined_filename = f"csv_files/combined/combined {file_identifier}"
        
        # Check if combined file already exists
        if os.path.exists(combined_filename):
            print(f"Combined file already exists: {combined_filename}")
            # Read existing file to return in the list
            merged_df = pd.read_csv(combined_filename)
            merged_dataframes.append(merged_df)
            continue
            
        try:
            # Read the CSV files
            metrics_df = pd.read_csv(metrics_data)
            output_df = pd.read_csv(output_data)
            
            # Merge the dataframes
            merged_df = pd.merge(output_df, metrics_df, 
                               on=['Video ID', 'Sequence'], 
                               how='inner')
            
            # Save the merged DataFrame
            merged_df.to_csv(combined_filename, index=False)
            print(f"Merged data saved to {combined_filename}")
            
            merged_dataframes.append(merged_df)
            
        except Exception as e:
            print(f"Error processing {output_file}: {str(e)}")
            continue
    
    return merged_dataframes
merge_csv_data()
