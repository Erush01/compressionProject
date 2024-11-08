import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('central_slices_grayscale_sequence_ant.csv')

# Get a list of unique image names
image_names = df['Image'].unique()

# Plotting Central Slices for each Image in the Sequence
plt.figure(figsize=(15, 10))

# Iterate over each image and plot the central slice
for image in image_names:
    # Filter data for the current image
    image_data = df[df['Image'] == image]

    # Plot the central slice for the grayscale image
    plt.plot(image_data['Freq_Index'], image_data['Center_Slice'], label=image, alpha=0.7)

# Add labels, title, and grid
plt.xlabel('Frequency Index')
plt.ylabel('Magnitude')
plt.title('Central Slice of FFT Magnitudes Across Grayscale Image Sequence')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.grid(True)
plt.tight_layout()

plt.show()
