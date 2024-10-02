import threading
import cv2
import time
from typing import Tuple
from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np
import easyocr
import os
from dotenv import load_dotenv
from collections import defaultdict
load_dotenv()


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
        confidence (float): Confidence threshold of YOLO model to perform OCR on track object.
        classes (list[int]): The class list of YOLO of objects to track. Empty list/track all by default.
        pixel_padding (int): Number of pixels to pad the bounding box region.
        img_width/img_height (int): Dimensions of output display window.
        top_k (int): Store the k highest confidence images for sake of performing OCR. Default is 5.
    """
    
    def __init__(self, yolo_model: YOLO, video_source: str, framerate: int = 30, 
                 capped_fps: bool = False, restart_on_end: bool = False, confidence:float=0.75, 
                 classes:list=[], pixel_padding:int=5, img_width:int=800, img_height:int=600,
                 top_k:int=5):
        
        # video capture
        self.video_capture = VideoCapture(video_source, framerate, capped_fps, restart_on_end)
        self.img_width = img_width
        self.img_height = img_height

        # yolo basics
        self.yolo_model = yolo_model
        self.classes = classes
        self.confidence = confidence
        self.padding = pixel_padding

        # frame processing thread
        self.stop_event = threading.Event()
        self.processing_thread = threading.Thread(target=self.__process_frames__)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # ocr additions
        self.ocr_reader = easyocr.Reader(['en'], model_storage_directory="model/easyocr/")
        self.top_k = top_k
        self.top_k_images = {}


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
                # all ids detected by yolo
                current_ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else []

                # items previously tracked that are no longer tracked ==> they have left the frame(/are obstructed)
                ids_left_frame = set(self.top_k_images.keys()) - set(current_ids)
                for id in ids_left_frame:
                    self.__object_left_frame(id)
                    
                # if no one detected in single result, skip
                if res.boxes.id is None:
                    continue
                
                self.__yolo_detection_processing(res, frame)
                annotated_frame = res.plot() # plot all objects that are detected
            
            cv2.imshow(feed_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


    def __new_object_in_frame(self, id:int, conf:float, cropped_image_path:str):
        """
        All functionality for when a new object is being tracked. 
        
        Adds entry to the dict/
        """

        print(f"New object in frame func: id={id}")
        self.top_k_images[id] = []

        return None
    
    
    def __object_being_tracked(self, id:int, confidence:float, cropped_image_path:str):
        """
        All functionality for an object that is actively being tracked.

        Updates the top_k_images dict to maintain which images are best to store.
        """
        
        print(f"object being tracked func: id={id}")
        # if fewer than k images are stored, simply append a conf/image tuple to the associated id
        if len(self.top_k_images[id]) < self.top_k:
            self.top_k_images[id].append([confidence, cropped_image_path])

        else:
            # find entry with least confidence, replace it with new entry if possible
            smallest_index = 0
            for i in range(1, len(self.top_k_images[id])):
                if self.top_k_images[id][i][0] < self.top_k_images[id][smallest_index][0]:
                  smallest_index = i
            
            # if confidence of new element is larger, replace the least confident element
            if confidence > self.top_k_images[id][smallest_index][0]:
                self.top_k_images[id][smallest_index] = [confidence, cropped_image_path]

        # sort the entries
        self.top_k_images[id].sort(key=lambda x: x[0])
        return None
    
    
    def __object_left_frame(self, id:int):
        """
        All functionality to deal with a tracked object that has left the feed. Pass ids of all objects to start with.

        Deletes the entry from the top_k_images dict, as well as send the popped element for OCR recognition.
        """
        
        # delete the entry from top_k images and send all files for ocr


        
        return None
    

    def __yolo_detection_processing(self, res : Results, original_frame : np.ndarray):
        """
        Separate out the detection processing code for modularity.

        Args:
            res (ultralytics Results): single item in the list of results returned by yolo.track()
            original_frame (np.ndarray): the original frame being processed
        """
        # for each detection in the result

        for detection in res.boxes:

            det_conf = detection.conf[0]
            if det_conf <= self.confidence:
                continue

            # id of detection by yolo, always unique
            det_id = detection.id[0]
            
            # bounding box and crop of object detected
            xmin, ymin, xmax, ymax = detection.xyxy[0]
            bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
            cropped_image = self.__crop_image(original_frame, bbox, self.padding)
            
            if cropped_image is not None:
                timestamp = int(time.time())
                filename = self.__save_cropped_image(cropped_image, detection.id[0], timestamp)
                
                # if object has not been detected till now
                if det_id not in self.top_k_images.keys():
                    self.__new_object_in_frame(det_id, det_conf, filename)

                else:
                    self.__object_being_tracked(det_id, det_conf, filename)

                # # perform ocr
                # plate_num, plate_conf = self.license_plate_ocr(cropped_image)
                
                # print("Detected License Plate: ", plate_num)


    def license_plate_ocr(self, plate_img:np.ndarray):
        """
        Extracts the license plate number from a cropped image.
        
        Args:
            plate_image (np.ndarray): The cropped image of the license plate.
            
        Returns:
            tuple: A tuple containing the extracted license plate number (str) and the average confidence score (float).
        """

        gray_image = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # perform OCR
        results = self.ocr_reader.readtext(binary_image)

        license_plate_number = ""
        confidence_scores = []

        for (bbox, text, prob) in results:
            license_plate_number += text + " "
            confidence_scores.append(prob)

        license_plate_number = license_plate_number.strip()
        average_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        return license_plate_number, average_confidence
    

    def multi_license_plate_ocr(self, plate_image_paths: list):
        """
        Extracts the license plate numbers from a list of cropped images and aggregates the results.
        Pass paths of images of the same item (from different angles).
        
        Args:
            plate_image_paths (list): A list of file paths to the cropped images of license plates.
            
        Returns:
            tuple: A tuple containing the most likely extracted license plate number (str) and the average confidence score (float).
        """
        
        results = defaultdict(list)  # To hold results for aggregation

        for path in plate_image_paths:
            # Read the image from the file path
            plate_img = cv2.imread(path)

            if plate_img is None:
                print(f"Warning: Unable to read image at {path}. Skipping.")
                continue

            gray_image = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Perform OCR
            ocr_results = self.ocr_reader.readtext(binary_image)

            for (bbox, text, prob) in ocr_results:
                results[text].append(prob)

        # Aggregate results
        aggregated_results = {}
        
        for text, confidences in results.items():
            avg_confidence = np.mean(confidences)
            aggregated_results[text] = avg_confidence

        # Find the best result based on average confidence
        if aggregated_results:
            best_result = max(aggregated_results.items(), key=lambda x: x[1])
            license_plate_number, average_confidence = best_result
        else:
            license_plate_number, average_confidence = "", 0.0

        return license_plate_number.strip(), average_confidence    


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
        Creates the specified folder at the path if it does not already exist.
        
        Args:
            cropped_image (np.ndarray): The cropped image to save.
            name (str): Some semi-unique name or identifier.
            timestamp (int): UNIX timestamp for unique filename generation.
            output_dir (str): Relative path to output folder.

        Returns:
            filepath (str): Returns the generated filepath.
        """
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = int(time.time())
        filename = f"{name}_{timestamp}.jpg"
        filepath = f"./{output_dir}/{filename}"
        cv2.imwrite(filepath, cropped_image)
        return filepath


    def stop(self):
        """
        Stop the video processing and release resources.
        """
        self.stop_event.set()
        self.processing_thread.join()
        self.video_capture.release()
