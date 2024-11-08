import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('central_slices_sequence_ant.csv')

# Get a list of unique image names
image_names = df['Image'].unique()

# Plot Central Slices for each Image in the Sequence
plt.figure(figsize=(15, 10))

# Iterate over each image and plot the central slices
for image in image_names:
    # Filter data for the current image
    image_data = df[df['Image'] == image]

    # Plot the central slice for Red, Green, and Blue channels
    plt.plot(image_data['Freq_Index'], image_data['Center_Slice_Red'], label=f'Red - {image}', color='red', alpha=0.5)
    plt.plot(image_data['Freq_Index'], image_data['Center_Slice_Green'], label=f'Green - {image}', color='green', alpha=0.5)
    plt.plot(image_data['Freq_Index'], image_data['Center_Slice_Blue'], label=f'Blue - {image}', color='blue', alpha=0.5)

# Add labels, title, and grid
plt.xlabel('Frequency Index')
plt.ylabel('Magnitude')
plt.title('Central Slice of FFT Magnitudes Across Image Sequence')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.grid(True)
plt.tight_layout()

plt.show()
