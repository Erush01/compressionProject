import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict, Any
import title_renderer

class VideoMetricsAnalyzer:
    def __init__(self, csv_file: str):

        # Define all potential encoding parameters
        self.ALL_ENCODING_PARAMS = [
            'Bitrate', 'Quantizer', 'QP Step', 'B-Frames', 
            'IP Factor', 'PB Factor', 'Ref Number', 
            'Subme', 'RC Lookahead'
        ]
        
        self.QUALITY_METRICS = ['PSNR(dB)', 'SSIM', 'Cbleed', 'Ringing', 'VIF', 'Compression Ratio (%)']
        
        # Marker shapes for different sequences
        self.MARKER_SHAPES = {
            "ant": "circle",
            "duck": "square",
            "referee": "cross",
            "misato": "x"
        }
        
        # Load and prepare data
        self.df = self._load_and_prepare_data(csv_file)
        
        # Dynamically filter encoding parameters with variation
        self.ENCODING_PARAMS = self._get_params_with_variation()
        
        # Initialize figures
        self.scatter_fig = None
        self.correlation_heatmap = None
        self.importance_matrix = None

    def _load_and_prepare_data(self, csv_file: str) -> pd.DataFrame:
        """
        Load and prepare the video metrics data from CSV file
        
        :param csv_file: Path to the CSV file
        :return: Prepared DataFrame
        """
        return pd.read_csv(csv_file)

    def _get_params_with_variation(self) -> List[str]:
        """
        Dynamically filter encoding parameters that have variation in the dataset
        
        :return: List of parameters with more than one unique value
        """
        # Filter parameters with more than one unique value
        params_with_variation = [
            param for param in self.ALL_ENCODING_PARAMS 
            if self.df[param].nunique() > 1
        ]
        
        print("Parameters with variation:", params_with_variation)
        return params_with_variation

    def create_scatter_plot(self, 
                             param: str = None, 
                             metric: str = 'PSNR(dB)') -> go.Figure:
        """
        Create an interactive scatter plot of encoding parameters vs quality metrics
        
        :param param: Encoding parameter for x-axis (uses first varying param if not specified)
        :param metric: Quality metric for y-axis
        :return: Plotly Figure object
        """
        # Use first varying parameter if not specified
        if param is None:
            if not self.ENCODING_PARAMS:
                raise ValueError("No encoding parameters with variation found")
            param = self.ENCODING_PARAMS[0]

        # Create a new figure
        fig = go.Figure()

        # Prepare columns for hover data
        hover_columns = self.ENCODING_PARAMS + ['Video ID']

        # Add traces for each unique sequence
        for sequence in self.df['Sequence'].unique():
            sequence_data = self.df[self.df['Sequence'] == sequence]
            
            # Prepare hover data
            hover_data = sequence_data[hover_columns].values

            fig.add_trace(go.Scatter(
                x=sequence_data[param],
                y=sequence_data[metric],
                mode='markers',
                marker=dict(
                    size=10,
                    symbol=self.MARKER_SHAPES.get(sequence, "circle"),
                    opacity=0.7
                ),
                name=sequence.capitalize(),
                customdata=hover_data,
                hovertemplate=(
                    'ID:%{customdata[-1]}<br>'
                    'Metric:%{y}<br>' +
                    ''.join(f"{p}: %{{customdata[{i}]}}<br>" for i, p in enumerate(self.ENCODING_PARAMS))
                )
            ))

        # Update layout
        fig.update_layout(
            title=f'{metric} vs {param}',
            xaxis_title=param,
            yaxis_title=metric,
            template='plotly_white'
        )
        fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "method": "update",
                        "label": param,
                        "args": [{"x": [self.df[self.df['Sequence'] == seq][param] for seq in self.df['Sequence'].unique()]}]
                    } for param in self.ENCODING_PARAMS
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
                        "args": [{"y": [self.df[self.df['Sequence'] == seq][metric] for seq in self.df['Sequence'].unique()]}]
                    } for metric in self.QUALITY_METRICS
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
    )

        
        # Store the figure for later use
        self.scatter_fig = fig
        return fig

    def create_correlation_heatmap(self):
        """
        Create a correlation heatmap between encoding parameters and quality metrics
        
        :return: Plotly Express Figure object
        """
        # Combine parameters and metrics
        all_columns = self.ENCODING_PARAMS + self.QUALITY_METRICS
        
        # Calculate correlation matrix
        correlation_matrix = self.df[all_columns].corr().loc[self.ENCODING_PARAMS, self.QUALITY_METRICS]
        
        # Generate heatmap
        heatmap_fig = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale='RdBu_r',
            title="Correlation between Encoding Parameters and Quality Metrics"
        )
        
        # Store the figure
        self.correlation_heatmap = heatmap_fig
        return heatmap_fig

    def calculate_feature_importance(self):
        """
        Calculate and visualize feature importance of encoding parameters on quality metrics
        
        :return: Plotly Express Figure object with feature importance
        """
        # Use only parameters with variation
        parameters = self.ENCODING_PARAMS
        metrics = self.QUALITY_METRICS

        # Initialize importance matrix
        importance_matrix = pd.DataFrame(index=parameters, columns=metrics)

        # Calculate feature importance for each metric
        for metric in metrics:
            X = self.df[parameters].fillna(0)
            y = self.df[metric].fillna(0)

            # Train Random Forest Regressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            # Store feature importances
            importance_matrix[metric] = model.feature_importances_

        # Create heatmap of feature importances
        fig = px.imshow(
            importance_matrix,
            text_auto=".2f",
            color_continuous_scale="Blues",
            labels=dict(color="Importance"),
            title="Feature Importance of Encoding Parameters on Quality Metrics"
        )
        fig.update_layout(template="plotly_white")
        
        # Store the figure
        self.importance_matrix = fig
        return fig

    def display_all_visualizations(self):
        """
        Display all generated visualizations
        """
        # Ensure all visualizations are created
        if self.scatter_fig is None:
            self.create_scatter_plot()
        if self.correlation_heatmap is None:
            self.create_correlation_heatmap()
        if self.importance_matrix is None:
            self.calculate_feature_importance()

        # Display the figures
        self.scatter_fig.show(renderer="titleBrowser", browser_tab_title="Encoding vs Metrics")
        self.correlation_heatmap.show(renderer="titleBrowser", browser_tab_title="Correlation Matrix")
        self.importance_matrix.show(renderer="titleBrowser", browser_tab_title="Importance Matrix")

def main():

    # File path
    csv_file = 'combined-05-12-2024.csv'
    
    # Create analyzer
    analyzer = VideoMetricsAnalyzer(csv_file)
    
    # Generate and display visualizations
    analyzer.create_scatter_plot()
    analyzer.create_correlation_heatmap()
    analyzer.calculate_feature_importance()
    analyzer.display_all_visualizations()

if __name__ == "__main__":
    main()