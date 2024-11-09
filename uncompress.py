import cv2
import os
import glob
from multiprocessing import Pool, cpu_count

class FrameExtractor:
    def __init__(self, dataset_folder, max_threads=4):
        self.sequences_folder = dataset_folder
        self.max_threads = max_threads or cpu_count()
    
    def extract_frames(self, video_path, output_folder):
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Save frame as BMP image with %08d format
            frame_filename = os.path.join(output_folder, f"{frame_num:08d}.bmp")
            cv2.imwrite(frame_filename, frame)
            frame_num += 1
        
        cap.release()
        print(f"Finished extracting frames for {video_path}")

    def process_video_folder(self, video_folder):
        videos=os.listdir(video_folder)
        
        for video in videos:
            video_id=video.split(".")[0]
            video_path = os.path.join(video_folder, f"{video_id}.mp4")
            output_folder = os.path.join(video_folder, f"{video_id}")
            
            if os.path.exists(video_path):
                self.extract_frames(video_path, output_folder)

    def run(self):
        # Get all folders in the sequences directory
        video_folders = glob.glob(os.path.join(self.sequences_folder, "*"))
        
        # Prepare the arguments list for multiprocessing
        with Pool(processes=self.max_threads) as pool:
            pool.map(self.process_video_folder, video_folders)

if __name__ == "__main__":
    sequences_folder = "compressed"  # Replace with your path
    max_threads = 4  # Set your max threads here
    
    extractor = FrameExtractor(sequences_folder, max_threads)
    extractor.run()
