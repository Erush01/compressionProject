import os
import csv
import random
import string

class H264:
    def __init__(self):
        self.name = None
        #Default 30
        self.fps = 30
        #Automatically decide how many B-frames to use , Default value : true
        self.b_adapt = True
        #Keep some B-frames as references , Default value : false
        self.b_pyramid = False
        #Number of B-frames between I and P , Default value : 0
        self.bframes = 0
        self.bframes_step = [0]
        
        #Adaptive spatial transform size Default value : false
        self.dct8x8 = False
        
        #Quantizer factor between I- and P-frames , Default value : 1.4
        self.ipfactor = 1.4
        self.ipfactor_step = [1.4]
        #Quantizer factor between P- and B-frames , Default value : 1.3
        self.pbfactor = 1.3
        self.pbfactor_step = [1.3]
        
        #Maximum quantizer difference between frames , Default value : 4
        self.qp = 4
        self.qp_step = [4]

        #Constant quantizer or quality to apply , Default value : 21
        self.quantizer = 21

        #Number of frames for frametype lookahead , Default value : 40
        self.rc_lookahead = 40
        #Number of reference frames , Default value : 3
        self.ref_number = 3
        self.ref_number_step = [3]

        #Subpixel motion estimation and partition decision quality: 1=fast, 10=best
        self.subme = 5
        self.subme_step = [5]

        #Bitrate in kbit/sec , Default value : 2048
        self.bitrate= 2048
        #self.bitrate_step=[576,1088,1536,2176,3072,4992,7552,20000]
        self.bitrate_step=[20000]
        # Name              Res         Link    Bitrate Video   Audio
        #                               (Mbps)  (Mbps)  (Kbps)  (Kbps)
        # 240p	            424x240	    1.0	    0.64	576 	64
        # 360p	            640x360	    1.5	    0.96	896	    64
        # 432p	            768x432	    1.8	    1.15	1088	64
        # 480p	            848x480	    2.0	    1.28	1216	64
        # 480p HQ	        848x480	    2.5	    1.60	1536	64
        # 576p	            1024x576	3.0	    1.92	1856	64
        # 576p HQ	        1024x576	3.5	    2.24	2176	64
        # 720p	            1280x720	4.0	    2.56	2496	64
        # 720p HQ	        1280x720	5.0	    3.20	3072	128
        # 1080p	            1920x1080	8.0	    5.12	4992	128
        # 1080p HQ	        1920x1080	12.0    7.68	7552	128
        # 1080p Superbit	1920x1080	N/A	    20.32	20000	320


        #Target location

        self.pipelines=[]

        self.gst_encoder_command = []
    
    def generate_gst_encoder_command(self):
        gst_encoder_command = [
            "!", "x264enc",
            f"bitrate={self.bitrate}"
        ]
        return gst_encoder_command
    
    def video_name(self,average_size_diff_percent,tail1=None,tail2=None,tail3=None):
        self.name="bitrate-"+str(self.bitrate) + "_fps-"+str(int(self.fps*100))+"_bframes-"+str(self.bframes) +"_ipfactor-"+str(self.ipfactor) + "_pbfactor-"+str(self.pbfactor) + "_qpstep-"+str(self.qp) + "_refnum-"+str(self.ref_number) + "_subme-"+str(self.subme) + "_averagesizediffpercent-" +str(average_size_diff_percent)
        pass
    
    def create_line_png(self,source_location,sink_location,begin):
        loc = '/png/%05d.png'
        source_location = source_location + loc
        ### Bad implementation
        
        pipeline= f'multifilesrc location={source_location} index={begin} caps="image/png,framerate={int(self.fps*100)}/100" ! pngdec ! videoconvert ! queue ! x264enc bitrate={self.bitrate} bframes={self.bframes} ip-factor={self.ipfactor} pb-factor={self.pbfactor} qp-step={self.qpstep} ref={self.ref} subme={self.subme} ! queue ! mp4mux ! queue ! filesink location={sink_location}/{self.name}.mp4'
        print("Pipeline is :    ", pipeline)
        return pipeline
    
    def create_line_bmp(self):
        ### Bad implementation
        pipeline= ["index=0",
                   "caps=\"image/bmp,framerate=30/1\"",
                   "!" "avdec_bmp" ,
                   "!","videoconvert","!" ,"queue",
                   "!" ,"x264enc" ,
                   f"bitrate={self.bitrate}" ,
                   f"bframes={self.bframes}", 
                   f"ip-factor={self.ipfactor}",
                   f"pb-factor={self.pbfactor}",
                   f"qp-step={self.qp}", 
                   f"ref={self.ref_number}",
                   f"subme={self.subme}",
                   "!" "queue" "!" "mp4mux" "!" "queue"]
        return pipeline

    def save_to_csv(self, video_id,filepath='output.csv'):
        # Generate a random 8-character ID
        # Define the data row to save
        data = {
            "Video ID": video_id,
            "Bitrate": self.bitrate,
            "B-Frames": self.bframes,
            "IP Factor": self.ipfactor,
            "PB Factor": self.pbfactor,
            "QP Step": self.qp,
            "Ref Number": self.ref_number,
            "Subme": self.subme
        }

        # Check if file exists to write headers only once
        file_exists = os.path.isfile(filepath)

        # Write the data to a CSV file
        with open(filepath, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()  # Write headers if file doesn't exist
            writer.writerow(data)  # Write the data row

        print(f"Data saved with Video ID: {video_id}")
