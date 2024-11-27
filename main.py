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

class AllInOne:
    def __init__(self,input_directory,output_directory,batchSize):
            
        self.input_dir=input_directory 
        self.output_directory=output_directory
        self.compressor=VideoCreator(input_directory,output_directory)
        self.extractor=FrameExtractor(output_directory,16)
        self.metrics=Metrics(input_directory,output_directory)
        self.json_path='h264_parameters_small.json'
        self.sequenceProgress = Progress(TextColumn("[progress.description]{task.description}"),
                                    BarColumn(),
                                    TaskProgressColumn(),
                                    MofNCompleteColumn(),
                                    TimeElapsedColumn())
        self.console=rich.console.Console()
        self.batchSize=batchSize


    def processFolder(self,input_dir,output_dir,name):
        with open(self.json_path, 'r') as json_file:
            parameters = json.load(json_file)
            parameter_lists = parameters.values()
            num_combinations = len(list(product(*parameter_lists)))
            batchNumber=num_combinations//self.batchSize
            batchCounter=0
            totalBatchCounter=1
            with rich.live.Live(self.sequenceProgress,console=self.console,refresh_per_second=10):
                    self.console.log(f'Processing sequence: {name}')
                    overall_task= self.sequenceProgress.add_task(f"[b]{name}", total=num_combinations)
                    for bitrate in parameters["bitrate"]:
                        # bitrate_t=self.metricProgress.add_task(f"[bold blue]{id}",total=5)
                        for bframes in parameters["bframes"]:
                            # bframes_t=self.metricProgress.add_task(f"[bold blue]{id}",total=5)
                            for ref in parameters["ref"]:
                                # ref_t=self.metricProgress.add_task(f"[bold blue]{id}",total=5)
                                for rc_lookahead in parameters["rc_lookahead"]:
                                    # rc_t=self.metricProgress.add_task(f"[bold blue]{id}",total=5)
                                    codec=H264(bitrate=bitrate,
                                                bframes=bframes,
                                                ref=ref,
                                                rc_lookahead=rc_lookahead)
                                    self.compressor.create_video_from_images(input_dir, output_dir,name,codec)
                                    if batchCounter==self.batchSize:
                                        self.console.log(f"Starting Uncompression of batch {totalBatchCounter}/{batchNumber}.")
                                        self.extractor.run(output_dir)
                                        self.console.log(f"Starting Metrics calculation of batch {totalBatchCounter}/{batchNumber}.")
                                        self.metrics.calculateMetrics(name)
                                        shutil.rmtree(output_dir)
                                        os.makedirs(output_dir)
                                        self.console.log(f"Batch {totalBatchCounter}/{batchNumber} is completed.")
                                        batchCounter=0
                                        totalBatchCounter+=1
                                    self.sequenceProgress.update(overall_task,advance=1)
                                    batchCounter+=1

        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    def run(self):
        start_time = time.time()  # Start time 
        for folder in os.listdir(self.input_dir):
            output_video = os.path.join(self.output_directory,folder)
            shutil.rmtree(output_video)
            os.makedirs(output_video)
            self.processFolder(os.path.join(self.input_dir,folder), output_video,folder)
            shutil.rmtree(output_video)
            os.makedirs(output_video)
        end_time = time.time()  # End time
        total_time = end_time - start_time  # Calculate total execution time
        print(f"Total execution time: {total_time:.2f} seconds")

if __name__=="__main__":


    dataset_directory = 'original'  # Replace with your actual path
    output_directory = 'compressed'   # Replace with your desired output path
    batchsize=2
    main=AllInOne(dataset_directory,output_directory,batchsize)
    main.run()


