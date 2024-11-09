import os
import subprocess
import time
from EoCutils import H264
import random
import string

class VideoCreator:
    def __init__(self, dataset_directory, output_directory):
        self.dataset_directory = dataset_directory
        self.output_directory = output_directory
    
        self.codec=H264()
        self.folders_to_process = os.listdir(dataset_directory)
    def create_video_from_images(self, image_folder, output_video):
        """Creates an MP4 video from BMP images in the specified folder."""
        video_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        gst_source_command = [
            "gst-launch-1.0",
            "multifilesrc", f"location={image_folder}/%08d.bmp",
        ]
        
        gst_sink_command = [
            "!", "filesink", f"location={output_video}/{video_id}.mp4"
        ]
        gst_command = gst_source_command + self.codec.create_line_bmp() + gst_sink_command

        try:
            print("Executing command:", ' '.join(gst_command))
            subprocess.run(gst_command, check=True)
            self.codec.save_to_csv(video_id)
        except subprocess.CalledProcessError as e:
            print(f"Error creating video from {image_folder}: {e}")

    def process_folder(self, folder):
        """Processes a single folder to create a video."""
        output_video = os.path.join(self.output_directory,folder)
        print("---------------->", output_video)
        self.create_video_from_images(os.path.join(self.dataset_directory,folder), output_video)

    def get_gst_encoder_command(self, gst_encoder_command):
        self.gst_encoder_command = gst_encoder_command

    def run(self):
        """Main function to find all image folders and process them sequentially."""
        start_time = time.time()  # Start time

        # Sequentially process each folder
        for folder in self.folders_to_process:
            print("Folder is --->> ",folder)
            self.process_folder(folder)

        end_time = time.time()  # End time
        total_time = end_time - start_time  # Calculate total execution time
        print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    dataset_directory = 'original'  # Replace with your actual path
    output_directory = 'compressed'   # Replace with your desired output path
    video_creator = VideoCreator(dataset_directory, output_directory)
    video_creator.run()
