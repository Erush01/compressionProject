import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def load_and_prepare_data(csv_file):
    """
    Load and prepare the video metrics data from CSV file
    """
    df = pd.read_csv(csv_file)
    return df

# Load the data
df = load_and_prepare_data('combined_output_metric.csv')
encoding_params = ('Bitrate',"B-Frames","Ref Number","RC Lookahead")
quality_metrics = ('PSNR(dB)', 'SSIM', 'Cbleed', 'Ringing', 'VIF')
marker_shapes = {
    "ant": "circle",
    "duck": "square",
    # Add as many shapes as needed for each sequence
}
# Create initial scatter plot
fig = go.Figure()

def update_scatter_plot(param, metric):
    """
    Update scatter plot based on selected encoding parameter and quality metric.
    """
    fig.data = []  # Clear existing data
    for sequence in df['Sequence'].unique():
        print(sequence)
        sequence_data = df[df['Sequence'] == sequence]
        fig.add_trace(go.Scatter(
            x=sequence_data[param],
            y=sequence_data[metric],
            mode='markers',
            marker=dict(size=10,
                        symbol=marker_shapes.get(sequence, "circle"),
                        opacity=0.7),  # Use "circle" as default shape),
            name=sequence.capitalize(),
            # Define hovertemplate to show all encoding parameter values
            hovertemplate=
                'ID:%{customdata[3]}<br>'            
                'Metric:%{y}<br>'+
                "Bitrate: %{customdata[0]}<br>" +               # Display x-axis (Bitrate)
                "B-Frames: %{customdata[1]}<br>" + # Display Quantizer
                "Ref: %{customdata[2]}<br>"+
                "RC Lookahead: %{customdata[3]}<br>"           # Display Subme

            ))
        fig.data[-1].customdata = sequence_data[['Bitrate','B-Frames', 'Ref Number','RC Lookahead']].values

    fig.update_layout(
        title=f'{metric} vs {param}',
        xaxis_title=param,
        yaxis_title=metric,
        template='plotly_white',
        
    )

# Initialize with default values
initial_param = encoding_params[0]
initial_metric = quality_metrics[0]
update_scatter_plot(initial_param, initial_metric)

# Add dropdown menus for selecting encoding parameters and quality metrics
fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "method": "update",
                    "label": param,
                    "args": [{"x": [df[df['Sequence'] == seq][param] for seq in df['Sequence'].unique()]}]
                } for param in encoding_params
            ],
            "direction": "down",
            "showactive": True,
            "x": 0.17,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top",
            "pad": {"r": 10, "t": 10},
            "active": 0
        },
        {
            "buttons": [
                {
                    "method": "update",
                    "label": metric,
                    "args": [{"y": [df[df['Sequence'] == seq][metric] for seq in df['Sequence'].unique()]}]
                } for metric in quality_metrics
            ],
            "direction": "down",
            "showactive": True,
            "x": 0.3,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top",
            "pad": {"r": 10, "t": 10},
            "active": 0
        }
    ]
)

# Create correlation heatmap
def create_correlation_heatmap(df):
    """
    Create a correlation heatmap between encoding parameters and quality metrics.
    """
    # Select relevant columns for correlation
    columns_of_interest = ['Bitrate', 'Quantizer', 'Subme', 'PSNR(dB)', 'SSIM', 'Cbleed', 'Ringing', 'VIF']
    correlation_matrix = df[columns_of_interest].corr()
    
    # Generate heatmap using plotly express
    heatmap_fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale='coolwarm',
        title="Correlation between Encoding Parameters and Quality Metrics"
    )
    return heatmap_fig

# Display the scatter plot and the heatmap
scatter_fig = fig
heatmap_fig = create_correlation_heatmap(df)

# Show both figures
scatter_fig.show()
# heatmap_fig.show()