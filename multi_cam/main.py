import torch
import numpy as np
import cv2
from pathlib import Path
import sort
import json
# import utilities
# import homography_tracker
from models.yolo.detection import YOLODetector
from models.sam2.sam import SAM
from concurrent.futures import ThreadPoolExecutor
import time
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

    # detector = torch.hub.load("ultralytics/yolov5", "yolov5m")
    # detector.agnostic = True

    # # Class 0 is Person
    # detector.classes = [67]
    # detector.conf = opts.conf

    # Initialize shared models once
    print("Initializing shared YOLO model...")
    shared_yolo = YOLODetector(
        # "/data/dataset/weights/base_weight/weights/best_wo_specialised_training.pt", 
        "/data/dataset/weights/opervu_seg_46SIs_211125/opervu_46SIs_21112025_2/weights/best.pt",
        conf_thres=0.25,
        iou_thres= 0.15
    )
    print(">>>>>>>>>", shared_yolo.names)
    print("Initializing shared SAM model...")
    shared_sam = SAM()
    shared_sam.build_model(
        "sam2.1_hiera_large",
        "/data/opervu/ws/sam2/checkpoints/model_large_38000.pt",
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
    frame_count = 0
    while True:
        active_trackers = 0
        
        per_frame_class_count= []
        frames= []
        for i, tracker in enumerate(trackers):
            frame ,class_count, ret = tracker.process_frame()
            per_frame_class_count.append(class_count)

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
        
        frame_count += 1
        print(f"Processed frame {frame_count}")
        print("="*20)
        # Now save the frame and the output
        
        # for i, class_count in enumerate(per_frame_class_count):
        #     print(f"Tracker {i+1} class counts: {class_count}")

        print("="*20)

        time.sleep(0.5)
    
    # Cleanup
    for tracker in trackers:
        tracker.cleanup_tracking()
    # global_tracker = homography_tracker.MultiCameraTracker(homographies, iou_thres=0.20)

    # num_frames1 = video1.get(cv2.CAP_PROP_FRAME_COUNT)
    # num_frames2 = video2.get(cv2.CAP_PROP_FRAME_COUNT)
    # num_frames = min(num_frames2, num_frames1)
    # num_frames = int(num_frames)

    # # NOTE: Second video 'cam4.mp4' is 17 frames behind the first video 'cam1.mp4'
    # video2.set(cv2.CAP_PROP_POS_FRAMES, 17)

    # video = None
    # for idx in range(num_frames):
    #     # Get frames
    #     frame1 = video1.read()[1]
    #     frame2 = video2.read()[1]

    #     # NOTE: YoloV5 expects the images to be RGB instead of BGR
    #     frames = [frame1[:, :, ::-1], frame2[:, :, ::-1]]

    #     anno = detector(frames)

    #     dets, tracks = [], []
    #     for i in range(len(anno)):
    #         # Sort Tracker requires (x1, y1, x2, y2) bounding box shape
    #         det = anno.xyxy[i].cpu().numpy()
    #         det[:, :4] = np.int0(det[:, :4])
    #         dets.append(det)

    #         # Updating each tracker measures
    #         tracker = trackers[i].update(det[:, :4], det[:, -1])
    #         tracks.append(tracker)

    #     global_ids = global_tracker.update(tracks)

    #     for i in range(2):
    #         frames[i] = utilities.draw_tracks(
    #             frames[i][:, :, ::-1],
    #             tracks[i],
    #             global_ids[i],
    #             i,
    #             classes=detector.names,
    #         )

    #     vis = np.hstack(frames)

    #     cv2.namedWindow("Vis", cv2.WINDOW_NORMAL)
    #     cv2.imshow("Vis", vis)
    #     key = cv2.waitKey(1)

    #     if key == ord("q"):
    #         break

    # video1.release()
    # video2.release()
    # cv2.destroyAllWindows()


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
