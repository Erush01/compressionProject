import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the CSV files
metrics_df = pd.read_csv("metrics.csv")
output_df = pd.read_csv("output.csv")

# Merge the data on 'Video ID' and 'Sequence' columns
merged_df = pd.merge(metrics_df, output_df, on=["Video ID", "Sequence"])

# Define encoding parameters and quality metrics
encoding_params = ["Bitrate", "Quantizer", "Subme", "B-Frames", "IP Factor", "PB Factor", "Ref Number", "RC Lookahead"]
quality_metrics = ["PSNR(dB)", "SSIM", "VIF", "Cbleed", "Ringing"]

# Start by creating a base plot with the first parameter-metric pair
x_param = encoding_params[0]
y_metric = quality_metrics[0]
fig = px.scatter(
    merged_df,
    x=x_param,
    y=y_metric,
    color="Sequence",
    symbol="Sequence",
    size="Quantizer",
    hover_data=["SSIM", "VIF", "Ringing", "Cbleed"],
    title=f"Relationship between {x_param} and {y_metric}",
    labels={
        x_param: x_param,
        y_metric: y_metric,
        "Subme": "Subme Level",
        "Quantizer": "Quantizer Level"
    }
)

# Update layout for clarity
fig.update_layout(
    xaxis_title=x_param,
    yaxis_title=y_metric,
    legend_title="Subme Level",
)

# Add dropdown menus for selecting x and y axes dynamically
fig.update_layout(
    updatemenus=[
        # Dropdown for selecting encoding parameters (x-axis)
        dict(
            buttons=list([
                {
                    "method": "restyle",
                    "label": param,
                    "args": [
                        {"x": [merged_df[param]]},
                        {"xaxis": {"title": param}}
                    ]
                } for param in encoding_params
            ]),
            "direction": "down",
            "showactive": True,
            "x": 0.17,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top",
            "pad": {"r": 10},
            "active": 0,
        ),
        # Dropdown for selecting quality metrics (y-axis)
        {
            "buttons": [
                {
                    "method": "restyle",
                    "label": metric,
                    "args": [
                        {"y": [merged_df[metric]]},
                        {"yaxis": {"title": metric}}
                    ]
                } for metric in quality_metrics
            ],
            "direction": "down",
            "showactive": True,
            "x": 0.3,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top",
            "pad": {"r": 10},
            "active": 0,
        },
    ]
)

# Save as an HTML file for interactive viewing
output_filename = "encoding_quality_relationships_dropdown.html"
fig.write_html(output_filename)
print(f"\nPlot saved as {output_filename}. Open this file in a web browser to explore the plot interactively.")
