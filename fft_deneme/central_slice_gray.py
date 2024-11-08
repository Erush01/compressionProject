import numpy as np
import cv2
import os
import pandas as pd
from glob import glob

# Function to compute the FFT magnitude spectrum for a grayscale image
def compute_fft(image):
    f = np.fft.fft2(image)
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
    # Load the image in grayscale
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # Calculate FFT magnitude spectrum
    magnitude_spectrum = compute_fft(image)

    # Take the central horizontal slice
    center_row = magnitude_spectrum[magnitude_spectrum.shape[0] // 2, :]

    # Store the results
    results.append({
        'Image': os.path.basename(image_file),
        'Center_Slice': center_row
    })

# Convert results to a DataFrame for easy export
data_records = []
for result in results:
    max_len = len(result['Center_Slice'])
    for i in range(max_len):
        data_records.append({
            'Image': result['Image'],
            'Freq_Index': i,
            'Center_Slice': result['Center_Slice'][i]
        })

df = pd.DataFrame(data_records)
df.to_csv('central_slices_grayscale_sequence_ant.csv', index=False)

print("Central slices saved to 'central_slices_grayscale_sequence.csv'")
