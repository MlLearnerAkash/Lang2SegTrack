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
from icecream import ic


def run_tracker(tracker, incision_area, prompt):
    tracker.set_incision_area(incision_area)
    tracker.current_text_prompt = prompt
    tracker.track()

def main(opts):
    video1 = cv2.VideoCapture(opts.video1)
    assert video1.isOpened(), f"Could not open video1 source {opts.video1}"
    video2 = cv2.VideoCapture(opts.video2)
    assert video2.isOpened(), f"Could not open video2 source {opts.video2}"

    cam4_H_cam1 = np.eye(3)#np.load(opts.homography)
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
        "/home/kamiar/ws/Lang2SegTrack/multi_cam/checkpoints/yolo_model/best.pt",
        conf_thres=0.45,
        # iou_thres= 0.15
    )
    # print(">>>>>>>>>", shared_yolo.names)
    print("Initializing shared SAM model...")
    shared_sam = SAM()
    shared_sam.build_model(
        "sam2.1_hiera_large",
        "/home/kamiar/ws/Lang2SegTrack/multi_cam/checkpoints/sam_model/model_large_38000.pt",
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

  # Configure trackers
    for i, tracker in enumerate(trackers):
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
            
            frame ,class_count, ret = tracker.process_frame()
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
        for track in range(len(all_tracks)):
            frames[track]= track_utilities.draw_projected_tracks(
                frames[track],
                all_tracks[track],
                global_ids[track],
                cam1_H_cam4,
                # src=track
            )
            # vis= np.hstack(frames)
            
            #NOTE: Optionally: Draw global IDs on frames
            # for cam_idx, (frame, tracks) in enumerate(zip(frames, all_tracks)):
            #     if frame is not None and len(tracks) > 0:
            #         for track in tracks:
            #             x1, y1, x2, y2, local_id = map(int, track)
            #             local_id = int(local_id)
            #             global_id = global_ids[cam_idx].get(local_id, -1)
                        
            #             # Draw global ID on frame
            #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #             label = f"G:{global_id} L:{local_id}"
            #             cv2.putText(frame, label, (x1, y1 - 10), 
            #                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
            #         # Save frame with global IDs
            #         # global_frame_filename = f"frame_global_{frame_count:06d}.jpg"
            #         # global_frame_path = output_dirs[cam_idx] / "frames" / global_frame_filename
            #         # cv2.imwrite(str(global_frame_path), frame)
            
            # # Optional: Create side-by-side visualization
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
