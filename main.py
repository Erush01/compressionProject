from VideoCreator import VideoCreator
from h264 import H264
from MetricCalculator import Metrics
from FrameExtractor import FrameExtractor
import json
from itertools import product
import rich
import time
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn,MofNCompleteColumn,TimeElapsedColumn,TaskProgressColumn,ProgressColumn
import rich
import shutil
import os
from multiprocessing import Pool, cpu_count
import itertools
from datetime import datetime

def clearFolder(folder):
    shutil.rmtree(folder)
    os.makedirs(folder)

class AllInOne:
    def __init__(self,input_directory,output_directory,batchSize):
            
        self.input_dir=input_directory 
        self.output_directory=output_directory
        self.compressor=VideoCreator(input_directory,output_directory)
        self.extractor=FrameExtractor(output_directory,16)
        self.metrics=Metrics(input_directory,output_directory)
        self.json_path='h264_parameters_small_old.json'
        self.sequenceProgress = Progress(TextColumn("[progress.description]{task.description}"),
                                    BarColumn(),
                                    TaskProgressColumn(),
                                    MofNCompleteColumn(),
                                    TimeElapsedColumn())
        self.console=rich.console.Console()
        self.current_date = datetime.now().strftime("%d-%m-%Y")

        self.batchSize=batchSize


    def processFolder(self,input_dir,output_dir,name):
        with open(self.json_path, 'r') as json_file:
            parameters = json.load(json_file)  # Load all parameters as a dictionary

            # Get all parameter combinations dynamically
            parameter_keys = list(parameters.keys())
            compression_csv_path=f"csv_files/compression/Compression  {'-'.join(parameter_keys)}_{self.current_date}.csv"
            metrics_csv_path=f"csv_files/metrics/Metrics  {'-'.join(parameter_keys)}_{self.current_date}.csv"
            parameter_values = list(parameters.values())
            parameter_combinations = list(itertools.product(*parameter_values))  # All combinations
            num_combinations = len(parameter_combinations)

            self.batchSize=min(num_combinations,self.batchSize)
            batchNumber = num_combinations // self.batchSize
            batchCounter = 1
            totalBatchCounter = 1
            with rich.live.Live(self.sequenceProgress, console=self.console, refresh_per_second=10):
                self.console.log(f'Processing sequence: {name}')
                overall_task = self.sequenceProgress.add_task(f"[b]{name}", total=num_combinations)

                # Iterate over parameter combinations
                for combination in parameter_combinations:
                    # Map combination back to parameter names
                    parameter_dict = dict(zip(parameter_keys, combination))
                    # Dynamically create the codec object using unpacked parameters
                    codec = H264(**parameter_dict,csv_path=compression_csv_path)
                    self.compressor.create_video_from_images(input_dir, output_dir, name, codec)
                    if batchCounter == self.batchSize:

                        self.console.log(f"Starting Uncompression of batch {totalBatchCounter}/{batchNumber}.")
                        self.extractor.run(output_dir)

                        self.console.log(f"Starting Metrics calculation of batch {totalBatchCounter}/{batchNumber}.")
                        self.metrics.calculateMetrics(name,metrics_csv_path)

                        clearFolder(output_dir)
                        self.console.log(f"Batch {totalBatchCounter}/{batchNumber} is completed.")
                        batchCounter = 0
                        totalBatchCounter += 1
                    self.sequenceProgress.update(overall_task,advance=1)
                    batchCounter+=1

        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    def run(self):
        start_time = time.time()  # Start time 
        for folder in os.listdir(self.input_dir):
            output_video = os.path.join(self.output_directory,folder)
            clearFolder(output_video)
            self.processFolder(os.path.join(self.input_dir,folder), output_video,folder)
            clearFolder(output_video)
        end_time = time.time()  # End time
        total_time = end_time - start_time  # Calculate total execution time
        print(f"Total execution time: {total_time:.2f} seconds")

if __name__=="__main__":


    dataset_directory = 'original'  # Replace with your actual path
    output_directory = 'compressed'   # Replace with your desired output path
    batchsize=64
    main=AllInOne(dataset_directory,output_directory,batchsize)
    main.run()


