import numpy as np
import cv2
import os
import pandas as pd
from glob import glob

# Function to compute FFT magnitude spectrum for a given channel
def compute_fft(channel):
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

# Function to compute radial average of a magnitude spectrum
def radial_profile(data):
    y, x = np.indices(data.shape)
    center = np.array(data.shape) // 2
    r = np.hypot(x - center[1], y - center[0]).astype(int)
    radial_mean = np.bincount(r.ravel(), data.ravel()) / np.bincount(r.ravel())
    return radial_mean

# Directory containing the sequence of images
image_folder = 'ant'
image_files = sorted(glob(os.path.join(image_folder, '*.bmp')))  # Modify extension as needed

# Initialize a list to store results
results = []

# Process each image in the sequence
for idx, image_file in enumerate(image_files):
    # Load the image
    image = cv2.imread(image_file)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split into R, G, B channels
    red_channel, green_channel, blue_channel = cv2.split(image_rgb)

    # Calculate FFT magnitude spectra
    magnitude_spectrum_red = compute_fft(red_channel)
    magnitude_spectrum_green = compute_fft(green_channel)
    magnitude_spectrum_blue = compute_fft(blue_channel)

    # Calculate radial averages
    radial_avg_red = radial_profile(magnitude_spectrum_red)
    radial_avg_green = radial_profile(magnitude_spectrum_green)
    radial_avg_blue = radial_profile(magnitude_spectrum_blue)

    # Store the results
    results.append({
        'Image': os.path.basename(image_file),
        'Radial_Avg_Red': radial_avg_red,
        'Radial_Avg_Green': radial_avg_green,
        'Radial_Avg_Blue': radial_avg_blue
    })

# Convert results to a DataFrame for easy export
data_records = []
for result in results:
    max_len = max(len(result['Radial_Avg_Red']), len(result['Radial_Avg_Green']), len(result['Radial_Avg_Blue']))
    for i in range(max_len):
        data_records.append({
            'Image': result['Image'],
            'Radial_Freq_Index': i,
            'Radial_Avg_Red': result['Radial_Avg_Red'][i] if i < len(result['Radial_Avg_Red']) else None,
            'Radial_Avg_Green': result['Radial_Avg_Green'][i] if i < len(result['Radial_Avg_Green']) else None,
            'Radial_Avg_Blue': result['Radial_Avg_Blue'][i] if i < len(result['Radial_Avg_Blue']) else None
        })

df = pd.DataFrame(data_records)
df.to_csv('radial_averages_sequence_ant.csv', index=False)

print("Radial averages saved to 'radial_averages_sequence.csv'")
