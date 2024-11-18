import os
import subprocess
import time
from h264 import H264
import random
import string
import json
from itertools import product
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn,MofNCompleteColumn,TimeElapsedColumn,TaskProgressColumn,ProgressColumn
import rich
import shutil
from FrameExtractor import FrameExtractor
from metricCalculator import Metrics
class VideoCreator:
    def __init__(self, dataset_directory, output_directory):
        self.dataset_directory = dataset_directory
        self.output_directory = output_directory
    
        self.folders_to_process = os.listdir(dataset_directory)


    def create_video_from_images(self, image_folder, output_video,sequence_name,codec):
        """Creates an MP4 video from BMP images in the specified folder."""
        video_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        gst_source_command = [
            "/usr/bin/gst-launch-1.0","-q",
            "multifilesrc", f"location={image_folder}/%08d.bmp",
        ]
        
        gst_sink_command = [
            "!", "filesink", f"location={output_video}/{video_id}.mp4"
        ]
        gst_command = gst_source_command + codec.create_line_bmp() + gst_sink_command

        try:
            # print(codec)
            subprocess.run(gst_command, check=True)
            codec.save_to_csv(video_id,sequence_name)

        except subprocess.CalledProcessError as e:
            print(f"Error creating video from {image_folder}: {e}")

if __name__ == "__main__":
    dataset_directory = 'original'  # Replace with your actual path
    output_directory = 'compressed'   # Replace with your desired output path
    video_creator = VideoCreator(dataset_directory, output_directory)
    video_creator.run()
    