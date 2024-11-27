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
            frame_filename = os.path.join(output_folder, f"{frame_num:08d}.png")
            cv2.imwrite(frame_filename, frame)
            frame_num += 1
        
        cap.release()


    def process_video(self,video_path):
            output_video_path=video_path.replace('.mp4','')
            if os.path.exists(video_path):
                self.extract_frames(video_path, output_video_path)

    def run(self,folder):
        # Get all folders in the sequences directory
        videos=[os.path.join(folder,x)for x in os.listdir(folder)]
        # Prepare the arguments list for multiprocessing
        with Pool(processes=self.max_threads) as pool:
            pool.map(self.process_video, videos)
    

if __name__ == "__main__":
    sequences_folder = "/media/parslab2/harddisk1/compressionProjectUncompressed"  # Replace with your path
    max_threads = 16  # Set your max threads here
    
    extractor = FrameExtractor(sequences_folder, max_threads)
    extractor.run()
