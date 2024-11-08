import numpy as np
import cv2
import os
import plotly.graph_objects as go
from glob import glob

# Function to compute the FFT magnitude spectrum for a grayscale image
def compute_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    magnitude_spectrum = np.log(magnitude_spectrum + 1)
    return magnitude_spectrum

# Directory containing the sequence of images
image_folder = 'ant'
image_files = sorted(glob(os.path.join(image_folder, '*.bmp')))  # Modify extension as needed

# Initialize Plotly figure
fig = go.Figure()

# Create 3D plot for each image's FFT
for idx, image_file in enumerate(image_files[0:1]):
    # Load the image in grayscale
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    
    # Compute FFT of the image
    magnitude_spectrum = compute_fft(image)

    # Generate meshgrid for frequency coordinates
    rows, cols = magnitude_spectrum.shape
    x = np.fft.fftfreq(cols)  # Frequency values along x-axis
    y = np.fft.fftfreq(rows)  # Frequency values along y-axis
    X, Y = np.meshgrid(x, y)
    
    # Add surface trace to plot
    fig.add_trace(go.Surface(z=magnitude_spectrum, x=X, y=Y, colorscale='Viridis', name=f'Image {os.path.basename(image_file)}'))

# Update layout to make it interactive
fig.update_layout(
    title='3D FFT Magnitude Spectrum',
    scene=dict(
        xaxis_title='Frequency (X)',
        yaxis_title='Frequency (Y)',
        zaxis_title='Magnitude'
    ),
    autosize=True
)

# Show the plot
print("showing")
fig.show()
