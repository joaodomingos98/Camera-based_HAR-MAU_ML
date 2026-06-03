# imports
from ultralytics import YOLO 
import cv2 # used for video/image processing and showing frames
import cvzone # for showing information on the frames
import pandas as pd # for easier handling the data results from YOLO
import FileSaveHandler
import time
from datetime import datetime
import os

# specify the detection settings
#They're outside the method because we want to use them in GUI
MIN_CONFIDENCE = 0.35
MIN_AREA = 100
MAX_AREA = float('inf')
ASPECT_RATIO_MIN = 0.25
ASPECT_RATIO_MAX = 4.0
DRAW_BOXES = True
VIDEO_FPS_VALUE = 5


def run(on_frame_callback, status_callback, stop_event, fps_callback):
    # ------------------------------------------------------------
    # ---------------------- 1. Setting up -----------------------
    # ------------------------------------------------------------

    # class variables
    # IP = "169.254.41.207"
    # ScheduledTime = "2024-06-10 15:30:00"

    # we are using YOLO version 12 (medium)
    MODEL_FILENAME = "yolo12m.pt"

    #prev_time saves the timestamp from the last frame
    prev_time = time.time()

    #fps is set to 0 because we need a start value.
    fps = 0

    try:
        if not os.path.exists(MODEL_FILENAME):
            raise FileNotFoundError
        model = YOLO(MODEL_FILENAME)
    except FileNotFoundError:
        raise SystemExit(
            "Error: YOLO model file 'yolo12m.pt' was not found.\n"
        )
    except Exception as e:
        raise SystemExit(f"Error occurred while initializing YOLO model: {e}")

    # the camera is hosted in this URL (IP might change each time)
    # "root" is the username
    # "YoloTracking" is the password
    # /axis-media/media.amp = axis fixed path
    IP = "169.254.41.208"
    CAM_URL = f"rtsp://root:YoloTracking@{IP}/axis-media/media.amp"

    # load the COCO classes + save in an array
    # (the file coco.txt contains different classes that YOLO can detect)
    try:
        with open("coco.txt", "r") as f:
            class_list = f.read().split("\n")
    except FileNotFoundError:
        raise SystemExit("Error: 'coco.txt' file not found.")

    # frame resolution
    FRAME_WIDTH = 690
    FRAME_HEIGHT = 388

    # var used to count frames and ID
    frame_count = 0
    frame_id = 1

    # saving files using file_save_handler
    log_cluster_id = datetime.now().strftime("ID_%Y%m%d_%H%M%S_%f")
    fsh = FileSaveHandler.file_save_handler(status_callback)
    fsh.create_new_folder(log_cluster_id)

    fsh.add_log_to_txt(f"Log Cluster: {log_cluster_id}")
    fsh.add_log_to_txt("------------------------------------------------")
    fsh.add_log_to_txt("SETTINGS:")
    fsh.add_log_to_txt("Minimum Confidence: " + str(MIN_CONFIDENCE))
    fsh.add_log_to_txt("Minimum Area: " + str(MIN_AREA))
    fsh.add_log_to_txt("Aspect Ratio Range: " + str(ASPECT_RATIO_MIN) + " - " + str(ASPECT_RATIO_MAX))
    fsh.add_log_to_txt("Frame Resolution: " + str(FRAME_HEIGHT) + " x " + str(FRAME_WIDTH))
    fsh.add_log_to_txt("------------------------------------------------")

    # create and start video logging
    video_filename = f"{log_cluster_id}.mp4"
    video_path = fsh.start_video_log(video_filename, VIDEO_FPS_VALUE, frame_size=(FRAME_WIDTH, FRAME_HEIGHT))


    # ------------------------------------------------------------
    # ---------------- 2. Processing the stream ------------------
    # ------------------------------------------------------------

    # continuous loop - processes each frame until stopped.
    # ...for each result when running YOLO on the RTSP stream:
    try:
        for results in model.predict(CAM_URL, stream=True, conf=0.15, iou=0.45, verbose=False):
            
            if stop_event.is_set():
                break
            
            # increment frame count
            frame_count += 1

            # for the text log:
            # record start time of processing this frame
            frame_start_time = time.time()
            timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # get a copy of the actual/original frame image from the stream:
            frame = results.orig_img.copy()

            # get the detected objects and their bounding boxes
            # converts the results to a pandas dataframe -> easier to handle 
            a = results.boxes.data
            pandas_dataframe = pd.DataFrame(a).astype("float")

            # var used to count people (in this specific frame)
            people_count = 0

            # loop through the detected objects (stored in pandas dataframe)
            for index, row in pandas_dataframe.iterrows():

                # extract the data from each index
                x1, y1, x2, y2 = map(int, row[:4]) # the bounding box coordinates
                confidence = float(row[4]) # confidence of this detection
                cls_id = int(row[5]) # the class id (for example 0 for "person")
                cld_id_string = class_list[cls_id] # convert cls_id to its corresponding string representation

                # only show objects of class "person" that meet minimum confidence requirement
                if cld_id_string == "person" and confidence > MIN_CONFIDENCE:
                    
                    # calculate aspect ratio + area of the bounding box
                    width = x2 - x1
                    height = y2 - y1
                    bbox_area = width * height
                    aspect_ratio = height / width if width > 0 else 0

                    # bool used to determine if the detection is valid
                    valid_detection = True

                    # if the detection doesn't meet the defined critera, mark invalid
                    if bbox_area < MIN_AREA:
                        valid_detection = False
                    if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
                        valid_detection = False
                    if width < 8 or height < 16:  # for small/partial detections
                        valid_detection = False

                    # if the detection is valid, draw it on the frame
                    if valid_detection:
                        people_count += 1

                        # choose the frame color based on confidence
                        if confidence > 0.7:
                            color = (0, 255, 0) #green
                        elif confidence > 0.5:
                            color = (0, 255, 255) #yellow
                        else:
                            color = (0, 0, 255) #red

                        # draw the bounding box + info (using cv2/cvzone) on the image
                        cx = (x1 + x2)//2
                        cy = (y1 + y2)//2
                        if DRAW_BOXES:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cvzone.putTextRect(frame, f'Person {confidence:.2f}', (x1, y1-10), scale=2, thickness=2)
                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)


            # ------------------------------------------------------------
            # --------------- 3. Overlay Info Preferences ----------------
            # ------------------------------------------------------------

            # display total people count, frame number + frame timestamp on the top-left corner
            cvzone.putTextRect(frame, f'Total People: {people_count}', (20, 50), scale=3, thickness=3, colorR=(0, 255, 0))
            cvzone.putTextRect(frame, f'Frame: {frame_count}, ID: {frame_id}', (20, 120), scale=2.5, thickness=2, colorR=(255, 0, 0))
            # cvzone.putTextRect(frame, f'Time: {timestamp_start}', (20, 180), scale=2.5, thickness=2, colorR=(0, 255, 0))

            # display frame ID (corresponding to the log ID) on the image
            # cvzone.putTextRect(frame, f'ID: {frame_id}', (20, 120), scale=2.5, thickness=2, colorR=(255, 0, 0))

            # calculate total frame processing time
            frame_end_time = time.time()
            timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            process_time = round(frame_end_time - frame_start_time, 3)

            if VIDEO_FPS_VALUE > 0:
                #calculates how long a frame should take at the selected FPS
                #E.g. 10 fps would take 0.1 sec per frame.
                target_frame_time = 1.0 / VIDEO_FPS_VALUE

                #sleep time is the time that remains after the processing.
                sleep_time = target_frame_time - process_time

                #pauses the loop so it doesn't run too fast
                if sleep_time > 0:
                    time.sleep(sleep_time)

            #The time for when the current frame is fully processed.
            current_time = time.time()

            #how long it took to get from the previous frame to this one
            fps = 1 / (current_time - prev_time) if current_time != prev_time else 0

            # updated so the next frame can be compared correctly
            prev_time = current_time

            fps_callback(fps)

            # resize the frame + open a window to show the stream 
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            on_frame_callback(frame)
            
            # add every frame to mp4
            fsh.add_frame_to_video(frame)  

            # add every frame to the csv file
            fsh.add_row_to_csv(f"{timestamp_start},{people_count}")

            # only save every 3rd frame as image log + txt log (for human reviewing)
            if frame_count % 3 == 0:        
                
                # log info to text file
                log_line = (
                    f"ID: {frame_id} | frame: {frame_count} | start: {timestamp_start} | "
                    f"end: {timestamp_end} | process: {process_time:.3f}s | people detected: {people_count}"
                )
                fsh.add_log_to_txt(log_line)

                # save the frame image
                image_name = f"{frame_id}.jpg"
                fsh.add_image_log(frame, image_name)

                # increment the id
                frame_id += 1



    except Exception as e:
        status_callback(f"Error: Camera stream stopped - {str(e)}")
        return

    finally:
        fsh.stop_video_log()

    # clean terminate application after breaking out the loop


#Test mode: allows running pmain3 standalone to verify camera works
if __name__ == "__main__":
    import threading
    from threading import Event
    
    #Create simple display window for testing
    stop_event = Event()
    
    def on_frame_callback(frame):
        #Display frame in a window
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
    
    def status_callback(message):
        #Print status messages to console
        print(f"[STATUS] {message}")
    
    def fps_callback(fps):
        #Print FPS to console
        print(f"[FPS] {fps:.1f}")
    
    try:
        #Run the main detection loop
        run(on_frame_callback, status_callback, stop_event, fps_callback)
    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()
    finally:
        cv2.destroyAllWindows()
