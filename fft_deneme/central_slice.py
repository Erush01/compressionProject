import numpy as np
import cv2
import os
import pandas as pd
from glob import glob

# Function to compute the FFT magnitude spectrum for a given channel
def compute_fft(channel):
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

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

    # Calculate FFT magnitude spectra for each channel
    magnitude_spectrum_red = compute_fft(red_channel)
    magnitude_spectrum_green = compute_fft(green_channel)
    magnitude_spectrum_blue = compute_fft(blue_channel)

    # Take the central horizontal slice for each channel
    center_row_red = magnitude_spectrum_red[magnitude_spectrum_red.shape[0] // 2, :]
    center_row_green = magnitude_spectrum_green[magnitude_spectrum_green.shape[0] // 2, :]
    center_row_blue = magnitude_spectrum_blue[magnitude_spectrum_blue.shape[0] // 2, :]

    # Store the results
    results.append({
        'Image': os.path.basename(image_file),
        'Center_Slice_Red': center_row_red,
        'Center_Slice_Green': center_row_green,
        'Center_Slice_Blue': center_row_blue
    })

# Convert results to a DataFrame for easy export
data_records = []
for result in results:
    max_len = max(len(result['Center_Slice_Red']), len(result['Center_Slice_Green']), len(result['Center_Slice_Blue']))
    for i in range(max_len):
        data_records.append({
            'Image': result['Image'],
            'Freq_Index': i,
            'Center_Slice_Red': result['Center_Slice_Red'][i] if i < len(result['Center_Slice_Red']) else None,
            'Center_Slice_Green': result['Center_Slice_Green'][i] if i < len(result['Center_Slice_Green']) else None,
            'Center_Slice_Blue': result['Center_Slice_Blue'][i] if i < len(result['Center_Slice_Blue']) else None
        })

df = pd.DataFrame(data_records)
df.to_csv('central_slices_sequence_ant.csv', index=False)

print("Central slices saved to 'central_slices_sequence.csv'")
