import utils
from ultralytics import YOLO
import time

import os
from dotenv import load_dotenv
load_dotenv()


def main():

    srcs = os.getenv('VIDEO_SOURCES', "./data/vids/video1.mp4")
    sources = srcs.split(", ")

    # creates a new yolo model + video feed for each separate source provided above
    processors = [  utils.YOLOVideoProcessor(YOLO('model/yolo-license-plates.pt'), source, framerate=60, 
                    capped_fps=True, restart_on_end=True, img_width=1280, img_height=720,
                    confidence=0.65, pixel_padding=10) for source in sources  ]

    try:
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        for p in processors:
            p.stop()

if __name__ == "__main__":
    main()