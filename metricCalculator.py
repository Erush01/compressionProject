import numpy as np
import math
import cv2
import os
from skimage.metrics import structural_similarity,peak_signal_noise_ratio

import scipy

def PSNR(img1, img2): 
    mse = np.mean((img1 - img2) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return float("inf")
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse)) 
    return psnr 
    
def SSIM(img1,img2):
    
    
    img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY).astype(np.float64)
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel = scipy.ndimage.gaussian_filter
    mu1 = kernel(img1, sigma=1.5)
    mu2 = kernel(img2, sigma=1.5)
    sigma1 = kernel(img1 ** 2, sigma=1.5) - mu1 ** 2
    sigma2 = kernel(img2 ** 2, sigma=1.5) - mu2 ** 2
    sigma12 = kernel(img1 * img2, sigma=1.5) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return ssim_map.mean()

if __name__=="__main__":
    original_folder="original"
    original_image=os.listdir(original_folder)
    compressed_folder="compressed"
    original = cv2.imread("/home/erush/ele490Project/original/ant/00000000.bmp") 
    compressed = cv2.imread("/home/erush/ele490Project/compressed/ant/38I6C8KL/00000000.bmp", 1) 
    
    value = PSNR(original, compressed) 
    custom_score=SSIM(original,compressed)
    print(f"PSNR value is {value} dB") 
    print("SSIM", custom_score)
