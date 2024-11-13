import os
import subprocess
import time
from h264 import H264
import random
import string
import json
class VideoCreator:
    def __init__(self, dataset_directory, output_directory):
        self.dataset_directory = dataset_directory
        self.output_directory = output_directory
    
        self.folders_to_process = os.listdir(dataset_directory)
        self.json_path='h264_parameters_small_test.json'
    def compress_all(self,image_folder, output_video,sequence_name):
        with open(self.json_path, 'r') as json_file:
            parameters = json.load(json_file)
            for bitrate in parameters["bitrate"]:
                for quantizer in parameters["quantizer"]:
                    for subme in parameters["subme"]:
                            codec=H264(bitrate=bitrate,
                                        quantizer=quantizer,
                                        subme=subme)
                            self.create_video_from_images(image_folder, output_video,sequence_name,codec)

    def create_video_from_images(self, image_folder, output_video,sequence_name,codec):
        """Creates an MP4 video from BMP images in the specified folder."""
        video_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        gst_source_command = [
            "gst-launch-1.0","-q",
            "multifilesrc", f"location={image_folder}/%08d.bmp",
        ]
        
        gst_sink_command = [
            "!", "filesink", f"location={output_video}/{video_id}.mp4"
        ]
        gst_command = gst_source_command + codec.create_line_bmp() + gst_sink_command

        try:
            print(codec)
            subprocess.run(gst_command, check=True)
            codec.save_to_csv(video_id,sequence_name)
        except subprocess.CalledProcessError as e:
            print(f"Error creating video from {image_folder}: {e}")

    def process_folder(self, folder):
        """Processes a single folder to create a video."""
        output_video = os.path.join(self.output_directory,folder)
        self.compress_all(os.path.join(self.dataset_directory,folder), output_video,folder)

    def run(self):
        """Main function to find all image folders and process them sequentially."""
        start_time = time.time()  # Start time

        # Sequentially process each folder
        for folder in self.folders_to_process:
            self.process_folder(folder)

        end_time = time.time()  # End time
        total_time = end_time - start_time  # Calculate total execution time
        print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    dataset_directory = 'original'  # Replace with your actual path
    output_directory = 'compressed'   # Replace with your desired output path
    video_creator = VideoCreator(dataset_directory, output_directory)
    video_creator.run()
