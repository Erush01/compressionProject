import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_gradient_magnitude

def radial_profile(data):
    y, x = np.indices(data.shape)
    center = np.array(data.shape) // 2
    r = np.hypot(x - center[1], y - center[0])
    r = r.astype(int)

    # Compute radial profile
    radial_mean = np.bincount(r.ravel(), data.ravel()) / np.bincount(r.ravel())
    return radial_mean

def compute_fft(channel):
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Adding 1 to avoid log(0)
    return magnitude_spectrum


duck_folder=os.listdir("duck")
ant_folder=os.listdir("ant")
image=cv2.imread(os.path.join("duck",ant_folder[1]))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
red_channel, green_channel, blue_channel = cv2.split(image_rgb)

# Load image in grayscale

# Perform 2D FFT and shift the zero frequency component to the center
magnitude_spectrum_red = compute_fft(red_channel)
magnitude_spectrum_green = compute_fft(green_channel)
magnitude_spectrum_blue = compute_fft(blue_channel)

radial_avg_red = radial_profile(magnitude_spectrum_red)
radial_avg_green = radial_profile(magnitude_spectrum_green)
radial_avg_blue = radial_profile(magnitude_spectrum_blue)


# Plot the original image and its magnitude spectrum
plt.figure(figsize=(18, 12))

# Original channels
plt.subplot(3, 3, 1)
plt.imshow(red_channel, cmap='Reds')
plt.title('Red Channel')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(green_channel, cmap='Greens')
plt.title('Green Channel')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(blue_channel, cmap='Blues')
plt.title('Blue Channel')
plt.axis('off')

# FFT magnitude spectra
plt.subplot(3, 3, 2)
plt.imshow(magnitude_spectrum_red, cmap='gray')
plt.title('FFT Magnitude Spectrum (Red)')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(magnitude_spectrum_green, cmap='gray')
plt.title('FFT Magnitude Spectrum (Green)')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.imshow(magnitude_spectrum_blue, cmap='gray')
plt.title('FFT Magnitude Spectrum (Blue)')
plt.axis('off')

# Radial averages
plt.subplot(3, 3, 3)
plt.plot(radial_avg_red, color='red')
plt.title('Radial Average (Red Channel)')
plt.xlabel('Radial Frequency')
plt.ylabel('Magnitude')

plt.subplot(3, 3, 6)
plt.plot(radial_avg_green, color='green')
plt.title('Radial Average (Green Channel)')
plt.xlabel('Radial Frequency')
plt.ylabel('Magnitude')

plt.subplot(3, 3, 9)
plt.plot(radial_avg_blue, color='blue')
plt.title('Radial Average (Blue Channel)')
plt.xlabel('Radial Frequency')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
# # image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# # img_b=image[:,:,0]

# #Green channel
# # img_g=image[:,:,1]

# #Red channel
# # img_r=image[:,:,2]

# img_fft=np.fft.fft2(image).real
# plt.figure(figsize=(6, 6))
# plt.imshow(img_fft, cmap='gray')
# plt.colorbar(label='Amplitude')
# plt.xlabel("n")
# plt.ylabel("m")
# plt.show()
