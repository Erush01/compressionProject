import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('radial_averages_sequence.csv')

# Get a list of unique images
image_names = df['Image'].unique()

# Plotting Radial Averages for each Image in the Sequence
plt.figure(figsize=(15, 10))

for image in image_names:
    # Filter data for the current image
    image_data = df[df['Image'] == image]

    # Plot the radial averages for Red, Green, and Blue channels
    plt.plot(image_data['Radial_Freq_Index'], image_data['Radial_Avg_Red'], label=f'Red - {image}', color='red', alpha=0.5)
    plt.plot(image_data['Radial_Freq_Index'], image_data['Radial_Avg_Green'], label=f'Green - {image}', color='green', alpha=0.5)
    plt.plot(image_data['Radial_Freq_Index'], image_data['Radial_Avg_Blue'], label=f'Blue - {image}', color='blue', alpha=0.5)

# Add labels and legend
plt.xlabel('Radial Frequency Index')
plt.ylabel('Magnitude')
plt.title('Radial Average of FFT Magnitudes Across Image Sequence')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.grid(True)
plt.tight_layout()

plt.show()