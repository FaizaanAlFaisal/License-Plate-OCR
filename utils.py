import threading
import cv2
import time
from typing import Tuple
from ultralytics import YOLO
import numpy as np


class VideoCapture:
    """
    VideoCapture
    ------------
    This class is designed to be a wrapper for cv2.VideoCapture to streamline
    video feed handling by using threading.

    The primary function is that the read() function is run constantly in a separate thread.
    Locks and Events are used to allow this class to be thread-safe.


    Parameters:
        video_source (str): Path to a video file or a video feed URL (rtsp/rtmp/http).
        capped_fps (bool): If True, caps the frame rate (default is False). Set to true for file playback. 
        framerate (int): Frame rate for video file playback (used if capped_fps is True).
        restart_on_end (bool): If True, restarts video file playback when it ends (default is False).
    """
    
    last_frame = None
    last_ready = None
    lock = threading.Lock()
    stop_event = threading.Event()
    start_event = threading.Event()
    fps = 30
    video_source = None
    capped_fps = False
    restart_on_end = False

    ## ensure capped_fps is False for case of rstp/rtmp url 
    def __init__(self, video_source:str, framerate:int=30, capped_fps:bool=False, restart_on_end:bool=False):
        self.fps : int = framerate
        self.video_source : str = video_source
        self.capped_fps : bool = capped_fps
        self.restart_on_end : bool = restart_on_end
        self.cap : cv2.VideoCapture = cv2.VideoCapture(video_source)
        self.thread : threading.Thread = threading.Thread(target=self.__capture_read_thread__)
        self.thread.daemon = True
        self.thread.start()

    def __capture_read_thread__(self):
        """
        Continuously reads frames from the video source in a separate thread.

        This method is intended to run in a separate thread and is not meant to be called directly.
        It reads frames as soon as they are available, and handles video restart if specified.
        If capped_fps is True, it waits to maintain the specified frame rate.

        The method stops when stop_event is set or if the video source cannot provide frames and restart_on_end is False.
        """
        while not self.stop_event.is_set():
            if self.start_event.is_set():
                
                self.last_ready, last_frame = self.cap.read()
                if self.last_ready:
                    with self.lock:
                        self.last_frame = last_frame

                # print(self.video_source, " frame read: ", self.last_ready)
            
                if not self.last_ready and self.restart_on_end:  # restart if file playback video ended
                    with self.lock:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
                # only wait in case of video files, else keep reading frames without delay to prevent packet drop/burst receive
                if self.capped_fps:
                    time.sleep(1 / self.fps)
        return

    def read(self):
        """
        Retrieve the latest frame from the video capture. Skips frames automatically 
        if they are not read in time. Allows for realtime video playback.

        Returns:
            [success boolean, frame or None]
        """
        try:
            if self.start_event.is_set():
                # with self.lock:
                if (self.last_ready is not None) and (self.last_frame is not None):
                    return [self.last_ready, self.last_frame.copy()]
            else:
                if not self.stop_event.is_set():
                    print("Capture thread has started.")
                    self.start_event.set()

            return [False, None]
        except Exception as e:
            raise ValueError(f"Error encountered by read() function: {e}")
          
    def isOpened(self):
        """
        Check if the video source is opened.
        """
        return self.cap.isOpened()    

    def open(self, video_source):
        """
        Open a new video source.
        
        Args:
            video_source (str): Path to the new video source.
        """
        self.video_source = video_source
        with self.lock:
            self.cap.open(self.video_source)

    def release(self):
        """
        Stop the video capture and release resources.
        """
        self.stop_event.set()
        self.restart_on_end = False
        self.thread.join(2)
        
        with self.lock:
            self.cap.release()




class YOLOVideoProcessor:
    """
    YOLOVideoProcessor
    ------------------
    This class processes video frames using YOLO for object detection.
    It uses the VideoCapture class to manage video feeds and applies YOLO
    detection on each frame. Processing is done in separate threads.

    Parameters:
        video_source (str): Path to a video file or a video feed URL.
        yolo_model (YOLO): An instance of a YOLO model from Ultralytics.
        framerate (int): Frame rate for video file playback.
        capped_fps (bool): If True, caps the frame rate.
        restart_on_end (bool): If True, restarts video file playback when it ends.
    """
    
    def __init__(self, yolo_model: YOLO, video_source: str, framerate: int = 30, 
                 capped_fps: bool = False, restart_on_end: bool = False, confidence:float=0.75, 
                 classes:list=[], pixel_padding:int=5, img_width:int=800, img_height:int=600):
        
        self.video_capture = VideoCapture(video_source, framerate, capped_fps, restart_on_end)
        self.img_width = img_width
        self.img_height = img_height

        self.yolo_model = yolo_model
        self.classes = classes
        self.confidence = confidence
        self.padding = pixel_padding

        # frame processing thread
        self.stop_event = threading.Event()
        self.processing_thread = threading.Thread(target=self.__process_frames__)
        self.processing_thread.daemon = True
        self.processing_thread.start()


    def __process_frames__(self):
        """
        Continuously processes and displays frames from the video source using YOLO.
        Run in a separate thread. Enables ease of running multiple feeds very easily.
        """
        feed_name = self.video_capture.video_source

        while not self.stop_event.is_set():
            ret, frame = self.video_capture.read()
            
            if not ret:
                print("Frame not read in yolo vid proc")
                time.sleep(0.5)
                continue

            frame = cv2.resize(frame, (self.img_width, self.img_height))
            annotated_frame = frame

            results = self.yolo_model.track(frame, stream=True, persist=True, verbose=False, classes=[0])
            
            for res in results:

                # if no one detected in single result, skip
                if res.boxes.id is None:
                    continue
                
                annotated_frame = res.plot() # plot all that are detected

                for detection in res.boxes:
                    if detection.conf[0] <= self.confidence:
                        continue
                    xmin, ymin, xmax, ymax = detection.xyxy[0]
                    bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
                    
                    cropped_image = self.__crop_image(frame, bbox, self.padding)
                    if cropped_image is not None:
                        timestamp = int(time.time())
                        filename = self.__save_cropped_image(cropped_image, detection.id[0], timestamp)
            
            cv2.imshow(feed_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


    def __crop_image(self, frame:np.ndarray, bbox: Tuple[int, int, int, int], padding:int=0) -> np.ndarray:
        """
        Crop the image based on the bounding box coordinates, with optional padding.
        
        Args:
            frame (np.ndarray): The original frame.
            bbox (Tuple[int, int, int, int]): Bounding box coordinates (xmin, ymin, xmax, ymax).
            padding (int): Number of pixels to add as padding around the bounding box.
        
        Returns:
            np.ndarray: Cropped image with padding.
        """
        x_min, y_min, x_max, y_max = bbox
        
        # make sure coordinates (with optional padding) are within the frame boundaries
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        # if the adjusted box is invalid
        if x_min >= x_max or y_min >= y_max:
            return None
        
        return frame[y_min:y_max, x_min:x_max]


    def __save_cropped_image(self, cropped_image: np.ndarray, name: str, timestamp:int, output_dir:str="./detections/"):
        """
        Save the cropped image with a unique name based on the detected class.
        
        Args:
            cropped_image (np.ndarray): The cropped image to save.
            name (str): Some semi-unique name or identifier.
            timestamp (int): UNIX timestamp for unique filename generation.
            output_dir (str): Relative path to output folder.

        Returns:
            filename (str): Returns the generated filename.
        """
        timestamp = int(time.time())
        filename = f"{name}_{timestamp}.jpg"
        filepath = f"./{output_dir}/{filename}"
        cv2.imwrite(filepath, cropped_image)
        return filename


    def stop(self):
        """
        Stop the video processing and release resources.
        """
        self.stop_event.set()
        self.processing_thread.join()
        self.video_capture.release()
