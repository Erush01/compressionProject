import numpy as np
import math
import cv2
import os
from skimage.metrics import structural_similarity,peak_signal_noise_ratio
import pywt
import scipy
from scipy import ndimage, special
import sewar
import  glob
import csv
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn,MofNCompleteColumn,TimeElapsedColumn,TaskProgressColumn,ProgressColumn
import rich
from rich.panel import Panel
from rich.table import Table

class Metrics():
    def __init__(self,original_path,compressed_path):
        self.compressed_path=compressed_path
        self.original_path=original_path
        self.metricProgress = Progress(
        "{task.description}",
        SpinnerColumn('aesthetic',speed=0.4,style=rich.style.Style(color='yellow')),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        )    
        
        self.sequenceProgress = Progress(TextColumn("[progress.description]{task.description}"),
                                         BarColumn(),
                                         TaskProgressColumn(),
                                         MofNCompleteColumn(),
                                         TimeElapsedColumn())
        self.progress_table = Table.grid()
        self.progress_table.add_row(
            Panel.fit(
                self.sequenceProgress, title="Overall Progress", border_style="yellow", padding=(2, 2)
            ),
            Panel.fit(self.metricProgress, title="[b]Sequence Progress", border_style="cyan", padding=(1, 2)),
        )
        
    def calculateMetrics(self):
        with rich.live.Live(self.progress_table,refresh_per_second=10):
            for sequence in os.listdir(self.original_path):
                video_id=[x for x in os.listdir(os.path.join(self.compressed_path,sequence)) if not x.endswith(".mp4")]
                overall_task= self.sequenceProgress.add_task(f"[b]{sequence}", total=len(video_id))
                original_images=sorted([os.path.join(self.original_path,sequence,x) 
                                        for x in os.listdir(os.path.join(self.original_path,sequence))])
                for id in video_id:

                    comp_images=sorted([os.path.join(self.compressed_path,sequence,id,x) 
                                        for x in os.listdir(os.path.join(self.original_path,sequence))])
                    psnr,ssim,cbleed,ringing,vif=0,0,0,0,0
                    seq_len=len(original_images)
                    task=self.metricProgress.add_task(f"[bold blue]{id}",total=seq_len)
                    for i in range(0,seq_len):
                        original = cv2.imread(original_images[i]) 
                        compressed = cv2.imread(comp_images[i])

                        psnr+=self._PSNR(original,compressed)
                        ssim+=self._SSIM(original,compressed)
                        cbleed+=self._ColorBleeding(original,compressed)
                        ringing+=self._Ringing(original,compressed)
                        vif+=self._VIF(original,compressed)
                        
                        self.metricProgress.update(task,advance=1)
                        
                    psnr/=seq_len
                    ssim/=seq_len
                    cbleed/=seq_len
                    ringing/=seq_len
                    vif/=seq_len
                    self.saveCsv(sequence,id,psnr,ssim,cbleed,ringing,vif)
                    self.sequenceProgress.update(overall_task,advance=1)
                
    def saveCsv(self,name,video_id,psnr,ssim,cbleed,ringing,vif,filepath='metrics.csv'):
        # Generate a random 8-character ID
        # Define the data row to save
        data = {
            "Sequence":name,
            "Video ID": video_id,
            "PSNR(dB)": psnr,
            "SSIM": ssim,
            "Cbleed": cbleed,
            "Ringing": ringing,
            "VIF": vif}

        # Check if file exists to write headers only once
        file_exists = os.path.isfile(filepath)

        # Write the data to a CSV file
        with open(filepath, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()  # Write headers if file doesn't exist
            writer.writerow(data)  # Write the data row

        print(f"Data saved with Video ID: {video_id}")                
                    
    def _PSNR(self,img1, img2): 
        mse = np.mean((img1 - img2) ** 2) 
        if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                    # Therefore PSNR have no importance. 
            return float("inf")
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse)) 
        return psnr 
    
    def _SSIM(self,img1,img2):
    
        img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY).astype(np.float32)
        img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY).astype(np.float32)
        
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
    
    def _ColorBleeding(self,original, compressed):
        """
        Measures color bleeding artifacts in chrominance channels.
        Works with YCbCr color space where bleeding is most visible.
        """
        if len(original.shape) != 3:
            return 0.0
            
        # Convert to YCbCr
        original_ycbcr = cv2.cvtColor(original, cv2.COLOR_RGB2YCrCb)
        compressed_ycbcr = cv2.cvtColor(compressed, cv2.COLOR_RGB2YCrCb)
        
        # Analyze chrominance channels (Cb and Cr)
        cb_diff = np.abs(original_ycbcr[:,:,1] - compressed_ycbcr[:,:,1])
        cr_diff = np.abs(original_ycbcr[:,:,2] - compressed_ycbcr[:,:,2])
        
        # Calculate color bleeding score
        bleeding_score = (np.mean(cb_diff) + np.mean(cr_diff)) / 2
        return bleeding_score

    def _Ringing(self,original, compressed):
        """
        Measures the ringing effect in a compressed image by comparing edge oscillations.
        """
        # Convert images to grayscale
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)

        # Edge detection on the original image
        edges = cv2.Canny(original_gray, 100, 200)

        # Sobel filter to measure intensity gradients (captures oscillations)
        sobel_original = cv2.Laplacian(original_gray, cv2.CV_64F)
        sobel_compressed = cv2.Laplacian(compressed_gray, cv2.CV_64F)
        # Focus on regions around edges for ringing effect measurement
        edge_regions = edges > 0

        # Calculate difference in gradients (oscillations) between original and compressed
        gradient_diff = np.abs(sobel_original - sobel_compressed)
        ringing_score = np.mean(gradient_diff[edge_regions])

        return ringing_score

    def _VIF(self,ref, dist):
        def compute_vif(ref_coeffs, dist_coeffs, sigma_nsq):
            num = 0.0
            den = 0.0

            for ref, dist in zip(ref_coeffs, dist_coeffs):
                ref_var = np.var(ref)
                dist_var = np.var(dist)
                g = ref_var / (dist_var + 1e-10)  # Gain factor

                # Calculate the VIF numerator and denominator
                num += np.log(1 + (g * ref_var / (sigma_nsq + 1e-10)))
                den += np.log(1 + (ref_var / (sigma_nsq + 1e-10)))

            vif_value = num / (den + 1e-10)
            return vif_value

        # Convert images to grayscale and normalize
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) / 255.0
        dist = cv2.cvtColor(dist, cv2.COLOR_BGR2GRAY) / 255.0
        
        # Wavelet decomposition using Haar wavelet
        sigma_nsq = 0.4  # Noise variance in natural images (adjustable based on image properties)
        ref_coeffs = pywt.wavedec2(ref, 'haar', level=4)
        dist_coeffs = pywt.wavedec2(dist, 'haar', level=4)
        
        # Remove the approximation coefficients
        ref_coeffs = ref_coeffs[1:]
        dist_coeffs = dist_coeffs[1:]
        
        # Calculate VIF for each sub-band level
        vifp = sum(compute_vif(ref_band, dist_band, sigma_nsq) for ref_band, dist_band in zip(ref_coeffs, dist_coeffs))
        return vifp
        
        
    def _PSNR_Grayscale(self,img1, img2):
        # Convert images to grayscale
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Calculate MSE
        mse = np.mean((img1_gray - img2_gray) ** 2)
        if mse == 0:
            return float("inf")
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return psnr

    def _PSNR_RGB(self,img1, img2):
        psnr_values = {}
        for i, color in enumerate(['R', 'G', 'B']):
            mse = np.mean((img1[:, :, i] - img2[:, :, i]) ** 2)
            if mse == 0:
                psnr_values[color] = float("inf")
            else:
                max_pixel = 255.0
                psnr_values[color] = 20 * math.log10(max_pixel / math.sqrt(mse))
        return psnr_values    



if __name__=="__main__":
    metrics=Metrics("original","compressed")
    
