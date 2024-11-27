import pandas as pd

def merge_csv_data(output_data,metrics_data):
    # Convert the string data into pandas DataFrames
    metrics_df = pd.read_csv(metrics_data)
    output_df = pd.read_csv(output_data)
    
    # Merge the dataframes on Video ID
    merged_df = pd.merge(output_df,metrics_df ,on=['Video ID', 'Sequence'], how='inner')
    
    # Sort by Sequence and Video ID for better readability
    # merged_df = merged_df.sort_values(['Sequence', 'Video ID'])
    merged_df.to_csv('combined_output_metric.csv', index=False)

    return merged_df

merge_csv_data("compressionData.csv","compressionMetrics.csv")