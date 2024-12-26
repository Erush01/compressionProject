import numpy as np
import math
import cv2
import os
from skimage.metrics import structural_similarity,peak_signal_noise_ratio
import pywt
import scipy
from scipy import ndimage, special
import glob
import csv
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn, TaskProgressColumn, ProgressColumn
import rich
from rich.panel import Panel
from rich.table import Table
import multiprocessing as mp
from functools import partial
from itertools import repeat

def calculate_folder_size(path):
    """Calculate the total size of all files in the folder and return it in MB with 4 decimal places."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    
    # Convert size to MB and format to 4 decimal places
    total_size_mb = total_size / (1024 ** 2)  # Convert from bytes to MB
    return round(total_size_mb, 4)

def process_image_pair(image_paths):
    """Process a single pair of original and compressed images"""
    original_path, compressed_path = image_paths
    original = cv2.imread(original_path)
    compressed = cv2.imread(compressed_path)
    
    return calculate_metrics(original, compressed)

def calculate_metrics(original, compressed):
    """Calculate all metrics for a pair of images"""
    def _PSNR(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if(mse == 0):
            return float("inf")
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return psnr

    def _SSIM(img1, img2):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
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

    def _ColorBleeding(original, compressed):
        if len(original.shape) != 3:
            return 0.0
            
        original_ycbcr = cv2.cvtColor(original, cv2.COLOR_RGB2YCrCb)
        compressed_ycbcr = cv2.cvtColor(compressed, cv2.COLOR_RGB2YCrCb)
        
        cb_diff = np.abs(original_ycbcr[:,:,1] - compressed_ycbcr[:,:,1])
        cr_diff = np.abs(original_ycbcr[:,:,2] - compressed_ycbcr[:,:,2])
        
        bleeding_score = (np.mean(cb_diff) + np.mean(cr_diff)) / 2
        return bleeding_score

    def _Ringing(original, compressed):
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(original_gray, 100, 200)

        sobel_original = cv2.Laplacian(original_gray, cv2.CV_64F)
        sobel_compressed = cv2.Laplacian(compressed_gray, cv2.CV_64F)
        edge_regions = edges > 0

        gradient_diff = np.abs(sobel_original - sobel_compressed)
        ringing_score = np.mean(gradient_diff[edge_regions])

        return ringing_score

    def _Blurring(original, compressed):
        """Detect blurring by analyzing the edge sharpness"""
        original_edges = cv2.Laplacian(original, cv2.CV_64F)
        compressed_edges = cv2.Laplacian(compressed, cv2.CV_64F)
        blurring_score = np.mean(np.abs(original_edges - compressed_edges))
        return blurring_score

    def _VIF(ref, dist):
        def compute_vif_channel(ref, dist, sigma_nsq=0.4):
            """
            Compute VIF for a single channel with multi-scale analysis.
            
            Parameters:
            -----------
            ref : ndarray
                Reference image channel
            dist : ndarray
                Distorted image channel
            sigma_nsq : float, optional
                Noise variance (default: 0.4)
            
            Returns:
            --------
            float
                VIF score for the channel
            """
            # Wavelet decomposition levels
            levels = 4
            
            # Wavelet decomposition
            ref_pyramid = pywt.wavedec2(ref, 'db5', level=levels)
            dist_pyramid = pywt.wavedec2(dist, 'db5', level=levels)
            
            # Initialize VIF components
            vif_total = 0.0
            
            # Process each scale
            for scale in range(levels + 1):
                if scale == 0:
                    # Approximation coefficients at the lowest scale
                    ref_scale = ref_pyramid[0]
                    dist_scale = dist_pyramid[0]
                else:
                    # Detail coefficients at each scale
                    ref_scale = ref_pyramid[scale][0]  # Horizontal details
                    dist_scale = dist_pyramid[scale][0]
                
                # Local statistics
                ref_mean = np.mean(ref_scale)
                dist_mean = np.mean(dist_scale)
                
                ref_var = np.var(ref_scale)
                dist_var = np.var(dist_scale)
                
                # Correlation
                numerator = np.sum((ref_scale - ref_mean) * (dist_scale - dist_mean))
                denominator = np.sqrt(np.sum((ref_scale - ref_mean)**2) * 
                                    np.sum((dist_scale - dist_mean)**2))
                
                # Prevent division by zero
                correlation = numerator / (denominator + 1e-10)
                
                # Information content calculation
                g = ref_var / (dist_var + 1e-10)
                
                # Compute VIF components
                num = np.log(1 + (g * correlation * ref_var / (sigma_nsq + 1e-10)))
                den = np.log(1 + (ref_var / (sigma_nsq + 1e-10)))
                
                # Accumulate scale-dependent VIF
                vif_total += num / (den + 1e-10)
            
            # Normalize VIF score
            return vif_total / levels

    # Preprocess images
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        dist_gray = cv2.cvtColor(dist, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        
        # Compute VIF
        vif_score = compute_vif_channel(ref_gray, dist_gray)
        
        return vif_score


    return {
        'psnr': _PSNR(original, compressed),
        'ssim': _SSIM(original, compressed),
        'vif': _VIF(original, compressed)
    }

class Metrics():
    def __init__(self, original_path, compressed_path, num_processes=None):
        self.compressed_path = compressed_path
        self.original_path = original_path
        self.num_processes = num_processes or mp.cpu_count()
        

    def process_sequence(self, sequence_info):
        """Process a single sequence"""
        sequence, video_id = sequence_info
        

        original_folder = os.path.join(self.original_path, sequence)
        compressed_folder = os.path.join(self.compressed_path, sequence, video_id)
    

        original_size = calculate_folder_size(original_folder)  # In MB
        compressed_size = calculate_folder_size(compressed_folder)  # In MB
        
        # Calculate size difference and compression ratio
        size_difference = original_size - compressed_size
        compression_ratio = (size_difference / original_size) * 100 if original_size > 0 else 0

        # Get sorted lists of image paths
        original_images = sorted([os.path.join(self.original_path, sequence, x) 
                                for x in os.listdir(os.path.join(self.original_path, sequence))])
        #comp_images = sorted([os.path.join(self.compressed_path, sequence, video_id, x) 
                            #for x in os.listdir(os.path.join(self.original_path, sequence))])
        

        comp_images = sorted([os.path.join(self.compressed_path, sequence, video_id, x.replace(".bmp", ".png")) 
                        for x in os.listdir(os.path.join(self.original_path, sequence))])
        

     

        # Create image pairs for processing
        image_pairs = list(zip(original_images, comp_images))
        
        # Process all images in the sequence
        with mp.Pool(processes=self.num_processes) as pool:
            results = pool.map(process_image_pair, image_pairs)
        
        # Calculate averages
        seq_len = len(results)
        avg_metrics = {
            'psnr': sum(r['psnr'] for r in results) / seq_len,
            'ssim': sum(r['ssim'] for r in results) / seq_len,
            'vif': sum(r['vif'] for r in results) / seq_len,
            'compression_ratio': compression_ratio
        }
        
        return sequence, video_id, avg_metrics

    def saveCsv(self, name, video_id, psnr, ssim, vif,compression_ratio, filepath='compressionMetrics.csv'):
        data = {
            "Sequence": name,
            "Video ID": video_id,
            "PSNR(dB)": psnr,
            "SSIM": ssim,
            "VIF": vif,
            "Compression Ratio (%)": compression_ratio
        }

        file_exists = os.path.isfile(filepath)
        
        with open(filepath, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

    def calculateMetrics(self,sequence,csv_path):
            # Process each sequence

        video_ids = [x for x in os.listdir(os.path.join(self.compressed_path, sequence)) 
                if not x.endswith(".mp4")]
        
        # Create sequence processing pool
        sequence_data = list(zip(repeat(sequence), video_ids))
        
        # Process sequences one at a time, but process images within each sequence in parallel
        for seq_info in sequence_data:
            sequence, video_id, metrics = self.process_sequence(seq_info)
            
            # Save results and update progress
            self.saveCsv(
                sequence, video_id,
                metrics['psnr'],
                metrics['ssim'],
                metrics['vif'],
                metrics['compression_ratio'],
                filepath=csv_path
            )

if __name__ == "__main__":
    metrics = Metrics("original", "/media/parslab2/harddisk1/compressionProjectUncompressed")
    metrics.calculateMetrics()