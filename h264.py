import os
import csv
import random
import string

class H264:
    def __init__(self,
                 bitrate=2048,
                 quantizer=21,
                 qp_step=4,
                 bframes=0,
                 ipfactor=1.4,
                 pbfactor=1.3,
                 ref=3,
                 subme=5,
                 rc_lookahead=40,
                 csv_path='compressionData.csv'
                 ):
        #Default 30
        self.csv_path=csv_path
        self.fps = 30
        
        
        #Bitrate in kbit/sec , Default value : 2048
        #[576,1088,1536,2176,3072,4992,7552,20000]
        self.bitrate= bitrate        
        
        
        #Constant quantizer or quality to apply , Default value : 21
        #[0,50]
        self.quantizer = quantizer        
                
        #Maximum quantizer difference between frames , Default value : 4
        #[0,63]
        self.qp_step = qp_step
        
                
        #Number of B-frames between I and P , Default value : 0
        #[0,16]
        self.bframes = bframes
                
        #Quantizer factor between I- and P-frames , Default value : 1.4
        #[0.0,2.0]
        self.ipfactor = ipfactor

        #Quantizer factor beween P- and B-frames , Default value : 1.3
        #[0.0,2.0]
        self.pbfactor = pbfactor
        
        #Number of reference frames , Default value : 3
        #[0,16]
        self.ref = ref

        #Subpixel motion estimation and partition decision quality: 1=fast, 10=best
        #[1,10]
        self.subme = subme


        #Number of frames for frametype lookahead , Default value : 40
        #[0,250]
        self.rc_lookahead = rc_lookahead
        
        #self.bitrate_step=[576,1088,1536,2176,3072,4992,7552,20000]
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

    def __repr__(self):
        return (f"\nbitrate={self.bitrate}\n"
                f"quantizer={self.quantizer}\n"
                f"qp-step={self.qp_step}\n"
                f"bframes={self.bframes}\n" 
                f"ip-factor={self.ipfactor}\n"
                f"pb-factor={self.pbfactor}\n"
                f"ref={self.ref}\n"
                f"subme={self.subme}\n"
                f"rc-lookahead={self.rc_lookahead}\n")

    def create_line_bmp(self):
        ### Bad implementation
        pipeline= ["index=0",
                   "caps=\"image/bmp,framerate=30/1\"",
                   "!" "avdec_bmp" ,
                   "!","videoconvert","!" ,"queue",
                   "!" ,"x264enc" ,
                   f"bitrate={self.bitrate}" ,
                   f"quantizer={self.quantizer}",
                   f"qp-step={self.qp_step}", 
                   f"bframes={self.bframes}", 
                   f"ip-factor={self.ipfactor}",
                   f"pb-factor={self.pbfactor}",
                   f"ref={self.ref}",
                   f"subme={self.subme}",
                   f"rc-lookahead={self.rc_lookahead}",
                   "!" "queue" "!" "mp4mux" "!" "queue"]
        return pipeline

    def save_to_csv(self, video_id,name):
        # Generate a random 8-character ID
        # Define the data row to save
        data = {
            "Sequence":name,
            "Video ID": video_id,
            "Bitrate": self.bitrate,
            "Quantizer":self.quantizer,
            "QP Step": self.qp_step,
            "B-Frames": self.bframes,
            "IP Factor": self.ipfactor,
            "PB Factor": self.pbfactor,
            "Ref Number": self.ref,
            "Subme": self.subme,
            "RC Lookahead":self.rc_lookahead,
        }

        # Check if file exists to write headers only once
        file_exists = os.path.isfile(self.csv_path)

        # Write the data to a CSV file
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()  # Write headers if file doesn't exist
            writer.writerow(data)  # Write the data row
