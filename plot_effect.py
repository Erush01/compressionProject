import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import title_renderer

def load_and_prepare_data(csv_file):
    """
    Load and prepare the video metrics data from CSV file
    """
    df = pd.read_csv(csv_file)
    return df

# Load the data
df = load_and_prepare_data('combined-05-12-2024.csv')
encoding_params = ('Bitrate',"B-Frames","Ref Number","Quantizer","RC Lookahead")
quality_metrics = ('PSNR(dB)', 'SSIM', 'Cbleed', 'Ringing', 'VIF','Compression Ratio (%)')
marker_shapes = {
    "ant": "circle",
    "duck": "square",
    "referee":"cross",
    "misato":"x",
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
                'ID:%{customdata[5]}<br>'            
                'Metric:%{y}<br>'+
                "Bitrate: %{customdata[0]}<br>" +               # Display x-axis (Bitrate)
                "B-Frames: %{customdata[1]}<br>" + # Display Quantizer
                "Ref: %{customdata[2]}<br>"+
                "RC Lookahead: %{customdata[3]}<br>"+
                "Quantizer: %{customdata[4]}<br>"

            ))
        fig.data[-1].customdata = sequence_data[['Bitrate','B-Frames', 'Ref Number','RC Lookahead','Quantizer','Video ID']].values

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
    ],
    # sliders=[
    #     {
    #         "steps": [
    #             {
    #                 "method": "update",
    #                 "label": f"{val}",
    #                 "args": [{"visible": [seq == val for seq in df['Sequence'].unique()]}],
    #             }
    #             for val in df['Sequence'].unique()
    #         ],
    #         "active": 0,
    #     }
    # ]
)

# Create correlation heatmap
def create_correlation_heatmap(df):
    """
    Create a correlation heatmap between encoding parameters (on the left) 
    and quality metrics (at the bottom).
    """
    # Define the parameters and metrics
    parameters = ['Bitrate',"B-Frames","Ref Number","Quantizer","RC Lookahead"]
    metrics = ['PSNR(dB)', 'SSIM', 'Cbleed', 'Ringing', 'VIF',"Compression Ratio (%)"]
    
    # Rearrange the correlation matrix to have parameters on rows and metrics on columns
    correlation_matrix = df[parameters + metrics].corr().loc[parameters, metrics]
    
    # Generate heatmap using Plotly Express
    heatmap_fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale='amp',
        title="Correlation between Encoding Parameters (Left) and Quality Metrics (Bottom)"
    )
    
    return heatmap_fig
def create_pairwise_scatter_matrix(df):
    """
    Create a pairwise scatter plot matrix for encoding parameters and metrics.
    """
    parameters = ['Bitrate', "B-Frames", "Ref Number", "Quantizer", "RC Lookahead"]
    metrics = ['PSNR(dB)', 'SSIM', 'Cbleed', 'Ringing', 'VIF', "Compression Ratio (%)"]

    fig = px.scatter_matrix(
        df,
        dimensions=parameters + metrics,
        color="Sequence",  # Color by sequence or another categorical feature
        title="Pairwise Scatter Matrix: Encoding Parameters and Quality Metrics",
        labels={col: col for col in parameters + metrics},
    )
    
    fig.update_traces(diagonal_visible=False)  # Hide histograms on the diagonal
    fig.update_layout(template="plotly_white")
    return fig


from sklearn.ensemble import RandomForestRegressor
def calculate_feature_importance(df):
    """
    Calculate feature importance of encoding parameters on all quality metrics.
    """
    parameters = ['Bitrate', "B-Frames", "Ref Number", "Quantizer", "RC Lookahead"]
    metrics = ['PSNR(dB)', 'SSIM', 'Cbleed', 'Ringing', 'VIF', "Compression Ratio (%)"]
    importance_matrix = pd.DataFrame(index=parameters, columns=metrics)

    for metric in metrics:
        X = df[parameters]
        y = df[metric]

        # Handle missing or invalid values (if any)
        X = X.fillna(0)
        y = y.fillna(0)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Save feature importance
        importance_matrix[metric] = model.feature_importances_

    fig = px.imshow(
    importance_matrix,
    text_auto=".2f",
    color_continuous_scale="Blues",
    labels=dict(color="Importance"),
    title="Feature Importance of Encoding Parameters on Quality Metrics"
    )
    fig.update_layout(template="plotly_white")
    
    return fig

# Display the scatter plot and the heatmap
scatter_fig = fig
heatmap_fig = create_correlation_heatmap(df)
scatter_matrix_fig = create_pairwise_scatter_matrix(df)
importance_matrix = calculate_feature_importance(df)

# Show both figures
scatter_fig.show(renderer="titleBrowser",browser_tab_title="Scatter Figure")
scatter_matrix_fig.show(renderer="titleBrowser",browser_tab_title="Scatter Matrix Figure")
importance_matrix.show(renderer="titleBrowser",browser_tab_title="Importance Matrix")
heatmap_fig.show(renderer="titleBrowser",browser_tab_title="Correlation Matrix")