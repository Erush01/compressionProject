import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import RadioButtons
import matplotlib
from mplcursors import cursor  # separate package must be installed

def load_and_prepare_data(csv_file):
    """
    Load and prepare the video metrics data from CSV file
    """
    df = pd.read_csv(csv_file)
    return df

fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.3)  # Make room for radio buttons
df = load_and_prepare_data('combined_output_metric.csv')
encoding_params = ('Bitrate', 'Quantizer', 'Subme')
quality_metrics = ('PSNR(dB)', 'SSIM', 'Cbleed', 'Ringing', 'VIF')
current_param=encoding_params[0]
current_metric=quality_metrics[0]
def update_plot(_,current_param=current_param, current_metric=current_metric):
    global ax,df
    ax.clear()
    
    
    # Plot for each sequence
    for sequence in df['Sequence'].unique():
        sequence_data = df[df['Sequence'] == sequence]
        param_column = current_param if current_param != 'Bitrate' else 'Bitrate'
        
        ax.scatter(sequence_data[param_column], sequence_data[current_metric],
                    label=sequence.capitalize(), alpha=0.7)
        
        # ax.plot(sequence_data[param_column], sequence_data[current_metric], 
        #         linestyle='--', alpha=0.5)

    ax.set_xlabel(current_param)
    ax.set_ylabel(current_metric)
    ax.set_title(f'{current_metric} vs {current_param}')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.draw()

def param_clicked(label):
    global current_param
    current_param = label
    update_plot(None,current_param, current_metric)

def metric_clicked(label):
    global current_metric
    current_metric = label
    update_plot(None,current_param, current_metric)

# Set initial values


def create_correlation_heatmap(df):
    """
    Create a correlation heatmap between encoding parameters and quality metrics
    """
    # Select relevant columns for correlation
    columns_of_interest = ['Bitrate', 'Quantizer', 'Subme', 'PSNR(dB)', 'SSIM', 'Cbleed', 'Ringing', 'VIF']
    correlation_matrix = df[columns_of_interest].corr()

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True)
    plt.title('Correlation between Encoding Parameters and Quality Metrics')
    plt.tight_layout()
    
    return plt.gcf()


if __name__ == "__main__":
        # Load the data
    
    # Create interactive plot
    """
    Create an interactive plot with radio buttons to select metrics
    """
    # Set up the figure and axis


    # Define encoding parameters and quality metrics

    # Create radio buttons for parameters
    ax_param = plt.axes([0.05, 0.7, 0.15, 0.15])
    radio_param = RadioButtons(ax_param, encoding_params, active=0)
    ax_param.set_title("Encoding Parameter")

    # Create radio buttons for metrics
    ax_metric = plt.axes([0.05, 0.4, 0.15, 0.15])
    radio_metric = RadioButtons(ax_metric, quality_metrics, active=0)
    ax_metric.set_title("Quality Metric")
    
    current_param = encoding_params[0]
    current_metric = quality_metrics[0]

    # Connect radio button events
    radio_param.on_clicked(param_clicked)
    radio_metric.on_clicked(metric_clicked)

    # Initial plot
    update_plot(None)

    # Create correlation heatmap

    # Show both plots
    plt.show()
