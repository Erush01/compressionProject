import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file containing the central slice data
df = pd.read_csv('central_slices_grayscale_sequence_ant.csv')

# Get the unique frequency indices
freq_indices = df['Freq_Index'].unique()

# Initialize an array to store the average central slice across all images
avg_central_slice = np.zeros(len(freq_indices))

# Iterate over each frequency index and calculate the average central slice
for i in freq_indices:
    # Get the central slice values for the current frequency index across all images
    central_slice_values = df[df['Freq_Index'] == i]['Center_Slice'].values
    
    # Calculate the mean of the central slice values for the current frequency index
    avg_central_slice[i] = np.mean(central_slice_values)

# Plot the average central slice across all images
plt.figure(figsize=(15, 8))
plt.plot(freq_indices, avg_central_slice, label='Average Central Slice', color='blue', linewidth=2)

# Add labels, title, and grid
plt.xlabel('Frequency Index')
plt.ylabel('Magnitude')
plt.title('Average Central Slice of FFT Magnitudes Across All Grayscale Images')
plt.grid(True)
plt.tight_layout()

plt.show()
