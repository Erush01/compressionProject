import numpy as np
import math
import cv2
import os
from skimage.metrics import structural_similarity,peak_signal_noise_ratio
import pywt
import scipy
from scipy import ndimage, special
import sewar
import glob
import csv
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn, TaskProgressColumn, ProgressColumn
import rich
from rich.panel import Panel
from rich.table import Table
import multiprocessing as mp
from functools import partial
from itertools import repeat

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

    def _VIF(ref, dist):
        def compute_vif(ref_coeffs, dist_coeffs, sigma_nsq):
            num = 0.0
            den = 0.0

            for ref, dist in zip(ref_coeffs, dist_coeffs):
                ref_var = np.var(ref)
                dist_var = np.var(dist)
                g = ref_var / (dist_var + 1e-10)

                num += np.log(1 + (g * ref_var / (sigma_nsq + 1e-10)))
                den += np.log(1 + (ref_var / (sigma_nsq + 1e-10)))

            vif_value = num / (den + 1e-10)
            return vif_value

        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) / 255.0
        dist = cv2.cvtColor(dist, cv2.COLOR_BGR2GRAY) / 255.0
        
        sigma_nsq = 0.4
        ref_coeffs = pywt.wavedec2(ref, 'haar', level=4)
        dist_coeffs = pywt.wavedec2(dist, 'haar', level=4)
        
        ref_coeffs = ref_coeffs[1:]
        dist_coeffs = dist_coeffs[1:]
        
        vifp = sum(compute_vif(ref_band, dist_band, sigma_nsq) 
                  for ref_band, dist_band in zip(ref_coeffs, dist_coeffs))
        return vifp

    return {
        'psnr': _PSNR(original, compressed),
        'ssim': _SSIM(original, compressed),
        'cbleed': _ColorBleeding(original, compressed),
        'ringing': _Ringing(original, compressed),
        'vif': _VIF(original, compressed)
    }

class Metrics():
    def __init__(self, original_path, compressed_path, num_processes=None):
        self.compressed_path = compressed_path
        self.original_path = original_path
        self.num_processes = num_processes or mp.cpu_count()
        
        self.metricProgress = Progress(
            "{task.description}",
            SpinnerColumn('aesthetic', speed=0.4, style=rich.style.Style(color='yellow')),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )    
        
        self.sequenceProgress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn()
        )
        
        self.progress_table = Table.grid()
        self.progress_table.add_row(
            Panel.fit(
                self.sequenceProgress, title="Overall Progress", border_style="yellow", padding=(2, 2)
            ),
            Panel.fit(self.metricProgress, title="[b]Sequence Progress", border_style="cyan", padding=(1, 2)),
        )

    def process_sequence(self, sequence_info):
        """Process a single sequence"""
        sequence, video_id = sequence_info
        
        # Get sorted lists of image paths
        original_images = sorted([os.path.join(self.original_path, sequence, x) 
                                for x in os.listdir(os.path.join(self.original_path, sequence))])
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
            'cbleed': sum(r['cbleed'] for r in results) / seq_len,
            'ringing': sum(r['ringing'] for r in results) / seq_len,
            'vif': sum(r['vif'] for r in results) / seq_len
        }
        
        return sequence, video_id, avg_metrics

    def saveCsv(self, name, video_id, psnr, ssim, cbleed, ringing, vif, filepath='metrics.csv'):
        data = {
            "Sequence": name,
            "Video ID": video_id,
            "PSNR(dB)": psnr,
            "SSIM": ssim,
            "Cbleed": cbleed,
            "Ringing": ringing,
            "VIF": vif
        }

        file_exists = os.path.isfile(filepath)
        
        with open(filepath, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

    def calculateMetrics(self):
        with rich.live.Live(self.progress_table, refresh_per_second=10):
            # Process each sequence
            for sequence in os.listdir(self.original_path):
                video_ids = [x for x in os.listdir(os.path.join(self.compressed_path, sequence)) 
                           if not x.endswith(".mp4")]
                
                overall_task = self.sequenceProgress.add_task(f"[b]{sequence}", total=len(video_ids))
                task = self.metricProgress.add_task(f"[bold blue]Processing", total=len(video_ids))
                
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
                        metrics['cbleed'],
                        metrics['ringing'],
                        metrics['vif']
                    )
                    self.sequenceProgress.update(overall_task, advance=1)
                    self.metricProgress.update(task, advance=1)

if __name__ == "__main__":
    metrics = Metrics("original", "compressed")
    metrics.calculateMetrics()