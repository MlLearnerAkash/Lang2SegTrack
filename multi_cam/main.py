import torch
import numpy as np
import cv2
from pathlib import Path
import sort
import json
# import utilities
import homography_tracker
from models.yolo.detection import YOLODetector
from models.sam2.sam import SAM
from concurrent.futures import ThreadPoolExecutor
import track_utilities
import time
import numpy as np
from icecream import ic
import os
class BboxDrawer:
    def __init__(self, image):
        self.image = image.copy()
        self.original = image.copy()
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        self.bbox = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x = x
            self.start_y = y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.image = self.original.copy()
                # Draw semi-transparent rectangle while dragging
                cv2.rectangle(self.image, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)
                # Add corner circles for clarity
                cv2.circle(self.image, (self.start_x, self.start_y), 5, (255, 0, 0), -1)
                cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)
                # Add text showing coordinates
                cv2.putText(self.image, f"({self.start_x}, {self.start_y})", 
                           (self.start_x - 50, self.start_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(self.image, f"({x}, {y})", 
                           (x + 10, y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow('Draw Incision Area', self.image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_x = x
            self.end_y = y
            self.bbox = [self.start_x, self.start_y, self.end_x, self.end_y]
            # Display final box
            self.image = self.original.copy()
            cv2.rectangle(self.image, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 255, 0), 3)
            cv2.circle(self.image, (self.start_x, self.start_y), 7, (255, 0, 0), -1)
            cv2.circle(self.image, (self.end_x, self.end_y), 7, (0, 0, 255), -1)
            cv2.putText(self.image, f"({self.start_x}, {self.start_y})", 
                       (self.start_x - 50, self.start_y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(self.image, f"({self.end_x}, {self.end_y})", 
                       (self.end_x + 10, self.end_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Draw Incision Area', self.image)
            
    def get_bbox(self):
        window_name = 'Draw Incision Area'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, self.image)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("Draw bounding box on the image (click and drag).")
        print("Press 'c' to confirm, 'r' to reset, or 'q' to quit.")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # Confirm
                if self.bbox is not None:
                    print(f"Bounding box confirmed: {self.bbox}")
                    cv2.destroyWindow(window_name)
                    return self.bbox
                else:
                    print("Please draw a bounding box first.")
            elif key == ord('r'):  # Reset
                self.image = self.original.copy()
                self.bbox = None
                cv2.imshow(window_name, self.image)
                print("Reset. Draw again.")
            elif key == ord('q'):  # Quit
                cv2.destroyWindow(window_name)
                return None


def run_tracker(tracker, incision_area, prompt):
    tracker.set_incision_area(incision_area)
    tracker.current_text_prompt = prompt
    tracker.track()

def main(opts):
    video1 = cv2.VideoCapture(opts.video1)
    assert video1.isOpened(), f"Could not open video1 source {opts.video1}"
    video2 = cv2.VideoCapture(opts.video2)
    assert video2.isOpened(), f"Could not open video2 source {opts.video2}"

    cam4_H_cam1 = np.load(opts.homography)#np.eye(3)#
    cam1_H_cam4 = np.linalg.inv(cam4_H_cam1)

    homographies = list()
    homographies.append(np.eye(3))
    homographies.append(cam1_H_cam4)


    ic("Initializing Global tracker")
    global_tracker= homography_tracker.MultiCameraTracker(homographies, iou_thres=0.2)
    # Initialize shared models once
    print("Initializing shared YOLO model...")
    shared_yolo = YOLODetector(
        # "/data/dataset/weights/base_weight/weights/best_wo_specialised_training.pt", 
        "/data/dataset/weights/opervu_seg_46SIs_211125/opervu_46SIs_21112025_2/weights/best.pt",
        conf_thres=0.45,
        iou_thres=0.25,
        imgsz= 2496
    )
    print("YOLO classes: ", shared_yolo.names)
    print("Initializing shared SAM model...")
    shared_sam = SAM()
    shared_sam.build_model(
        "sam2.1_hiera_large",
        "/data/opervu/ws/sam2/checkpoints/model_large_99000.pt",
        predictor_type="video",
        device="cuda:0",
        use_txt_prompt=True
    )

    output_dirs = [
        Path("./output/camera1"),
        Path("./output/camera2")
    ]

    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "frames").mkdir(exist_ok=True)
        (out_dir / "class_counts").mkdir(exist_ok=True)

    # Create trackers with shared models
    trackers = [
        sort.Sort(
            shared_yolo=shared_yolo,
            shared_sam=shared_sam,
            video_path= video,
            save_video=True,
            use_txt_prompt=True,
            mode="video",
            output_path= output_path,
        )
        for video, output_path in zip([opts.video1, opts.video2], ["video1.mp4",
                                                                   "video2.mp4"])
    ]

    # Configure trackers with interactive bounding box drawing
    for i, tracker in enumerate(trackers):
        print(f"\n{'='*50}")
        print(f"Setting incision area for Camera {i+1}")
        print(f"{'='*50}")
        
        # Read first frame to draw bbox on
        cap = cv2.VideoCapture([opts.video1, opts.video2][i])
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            drawer = BboxDrawer(frame)
            bbox = drawer.get_bbox()
            
            if bbox is not None:
                # Convert [x1, y1, x2, y2] to [x1, y1, x2, y2] format expected
                incision_area = [bbox[0], bbox[1], bbox[2], bbox[3]]
                tracker.set_incision_area(incision_area)
                print(f"Camera {i+1} incision area set to: {incision_area}")
            else:
                print(f"No bounding box drawn for Camera {i+1}, using default")
                tracker.set_incision_area([700, 1040+200, 1490-200, 1790])
        else:
            print(f"Could not read frame from video {i+1}, using default")
            tracker.set_incision_area([700, 1040+200, 1490-200, 1790])
        
        tracker.current_text_prompt = 'car'
        tracker.initialize_tracking()
    
    # Process frames synchronously
    combined_video_writer = None
    fps = 10  # Default FPS

    frame_count = 0
    while True:
        active_trackers = 0
        
        per_frame_class_count= []
        frames= []
        all_tracks= []
        for i, tracker in enumerate(trackers):
            
            frame ,class_count, ret = tracker.process_frame(opts.classes)
            per_frame_class_count.append(class_count)

            #NOTE:testing framedata_manager
            bboxes, obj_ids= tracker.get_current_frame_bboxes_and_ids()
            if len(bboxes)>0 and len(obj_ids)>0:
                tracks_array = np.zeros((len(bboxes), 5))
                for idx, (bbox, obj_id) in enumerate(zip(bboxes, obj_ids)):
                    tracks_array[idx]= [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], obj_id]
                all_tracks.append(tracks_array)
            else:
                all_tracks.append(np.empty((0, 5)))
            frames.append(frame)
            #Saving the result
            if frame is not None:
                frame_filename= "frame.jpg"
                frame_path= output_dirs[i]/"frames"/frame_filename
                cv2.imwrite(str(frame_path), frame)
            
            if class_count:
                counts_data = {
                    "frame_number": frame_count,
                    "classes": {}
                    }
                count_filename= "counts.json"
                count_path = output_dirs[i]/"class_counts"/count_filename
                with open(count_path, 'w') as f:
                    for category, counts in class_count.items():
                        counts_data["classes"][category] = {
                            "IN": int(counts['IN']),
                            "OUT": int(counts['OUT']),
                            "total": int(counts['IN'] + counts['OUT'])
                        }
                    json.dump(counts_data, f, indent=2)

            if ret:
                active_trackers += 1
            else:
                print(f"Tracker {i+1} finished")
        
        if active_trackers == 0:
            break
        
        # Update global tracker with tracks from all cameras
        if len(all_tracks) == len(trackers):
            global_ids = global_tracker.update(all_tracks)
            
            # Log global tracking results
            print(f"\n{'='*50}")
            print(f"Frame {frame_count} - Global Tracking Results:")
            print(f"{'='*50}")
            for cam_idx, (local_to_global, tracks) in enumerate(zip(global_ids, all_tracks)):
                print(f"\nCamera {cam_idx + 1}:")
                print(f"  Active tracks: {len(tracks)}")
                if len(tracks) > 0:
                    print(f"  Local ID -> Global ID mapping:")
                    for local_id, global_id in local_to_global.items():
                        print(f"    Local ID {local_id} -> Global ID {global_id}")

        homography_dict = {
        (0, 1): cam4_H_cam1,
        (1, 0): cam1_H_cam4,
        }
        frames = track_utilities.draw_all_projected_tracks(frames, all_tracks, global_ids, homography_dict)


        if all(f is not None for f in frames):
            # Resize frames to same height if needed
            max_height = max(f.shape[0] for f in frames)
            resized_frames = []
            for f in frames:
                if f.shape[0] != max_height:
                    scale = max_height / f.shape[0]
                    new_width = int(f.shape[1] * scale)
                    f = cv2.resize(f, (new_width, max_height))
                resized_frames.append(f)
            
            # Stack frames horizontally
            vis = np.hstack(resized_frames)
            
            # Initialize video writer on first frame
            if combined_video_writer is None:
                video_height, video_width = vis.shape[:2]
                combined_video_path = Path("./output") / "combined_tracking.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                combined_video_writer = cv2.VideoWriter(
                    str(combined_video_path),
                    fourcc,
                    fps,
                    (video_width, video_height)
                )
                print(f"\nCreating combined video: {combined_video_path}")
                print(f"Resolution: {video_width}x{video_height}, FPS: {fps}")
            
            # Write frame to video
            combined_video_writer.write(vis)
                        
        frame_count += 1
        print(f"\nProcessed frame {frame_count}")
        print("="*50)
    
        # Now save the frame and the output
        
        # for i, class_count in enumerate(per_frame_class_count):
        #     print(f"Tracker {i+1} class counts: {class_count}")

        print("="*20)

        time.sleep(0.5)
    
    # Cleanup
    for tracker in trackers:
        tracker.cleanup_tracking()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video1", type=str, default="./epfl/cam1.mp4")
    parser.add_argument("--video2", type=str, default="./epfl/cam4.mp4")
    parser.add_argument("--homography", type=str, default="./cam4_H_cam1.npy")
    parser.add_argument("--classes", type=int, nargs='*', default=[], help="List of class IDs to detect.")
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.3,
        help="IOU threshold to consider a match between two bounding boxes.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Max age of a track, i.e., how many frames will we keep a track alive.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Minimum number of matches to consider a track.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence value for the YoloV5 detector.",
    )

    opts = parser.parse_args()

    main(opts)
