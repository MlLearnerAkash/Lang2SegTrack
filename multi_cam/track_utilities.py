import cv2
import numpy as np
from icecream import ic


centroids = {}


def apply_homography(uv, H):
    uv_ = np.zeros_like(uv)

    for idx, (u, v) in enumerate(uv):
        uvs = H @ np.array([u, v, 1]).reshape(3, 1)
        u_, v_, s_ = uvs.reshape(-1)
        u_ = u_ / s_
        v_ = v_ / s_

        uv_[idx] = [u_, v_]

    return uv_


def apply_homography_xyxy(xyxy, H):
    xyxy_ = np.zeros_like(xyxy)
    for idx, (x1, y1, x2, y2) in enumerate(xyxy):
        x1, y1, s1 = H @ np.array([x1, y1, 1]).reshape(3, 1)
        x1 = float(x1 / s1)
        y1 = float(y1 / s1)

        x2, y2, s2 = H @ np.array([x2, y2, 1]).reshape(3, 1)
        x2 = float(x2 / s2)
        y2 = float(y2 / s2)
        xyxy_[idx] = [x1, y1, x2, y2]

    return xyxy_


def draw_bounding_boxes(image, bounding_boxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image given a list of (x1, y1, x2, y2) coordinates.

    :param image: The input image to draw the bounding boxes on.
    :type image: numpy.ndarray
    :param bounding_boxes: A list of (x1, y1, x2, y2) coordinates for each bounding box.
    :type bounding_boxes: list[tuple(int, int, int, int)]
    :param color: The color of the bounding boxes. Default is green.
    :type color: tuple(int, int, int)
    :param thickness: The thickness of the bounding boxes. Default is 2.
    :type thickness: int
    """

    for bbox in bounding_boxes:
        x1, y1, x2, y2 = np.intp(bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_matches(img1, kpts1, img2, kpts2, matches):
    vis = np.hstack([img1, img2])
    MAX_DIST_VAL = max([match.distance for match in matches])

    WIDTH = img2.shape[1]

    for idx, (src, dst, match) in enumerate(zip(kpts1, kpts2, matches)):
        src_x, src_y = src
        dst_x, dst_y = dst
        dst_x += WIDTH

        COLOR = (0, int(255 * (match.distance / MAX_DIST_VAL)), 0)

        vis = cv2.line(vis, (src_x, src_y), (dst_x, dst_y), COLOR, 1)

    return vis


def color_from_id(id):
    np.random.seed(id)
    return np.random.randint(0, 255, size=3).tolist()


def draw_tracks(image, tracks, ids_dict, src, H_src_dst, classes=None):
    """
    Draw bounding boxes on an image and print tracking IDs for each box.

    Args:
        image: An array representing the image to draw on.
        boxes: A list of bounding boxes, where each box is a tuple of (x, y, w, h).
        ids: A list of tracking IDs, where each ID corresponds to a box in the boxes list.
    """
    # Convert the image to RGB color space
    vis = np.array(image)
    bboxes = tracks[:, :4]
    ids = tracks[:, 4]
    labels = tracks[:, 4] #5
    centroids[src] = centroids.get(src, {})

    # Loop over each bounding box and draw it on the image
    for i, box in enumerate(bboxes):
        id = ids_dict[ids[i]]
        color = color_from_id(id)

        # Get the box coordinates
        x1, y1, x2, y2 = np.intp(box)
        # Draw the box on the image
        if centroids == None:
            vis = cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=2)
        else:
            centroids[src][id] = centroids[src].get(id, [])
            centroids[src][id].append(((x1 + x2) // 2, (y1 + y2) // 2))
            vis = draw_history(vis, box, centroids[src][id], color)

        # Print the tracking ID next to the box
        if classes == None:
            text = f"{labels[i]} {id}"
        else:
            text = f"{classes[labels[i]]} {id}"
        vis = cv2.putText(
            vis, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2
        )

    return vis


def draw_label(image, x, y, label, track_id, color):
    # Convert the image to RGB color space
    vis = np.array(image)

    # Print the tracking ID next to the box
    text = f"{label} {track_id}"
    vis = cv2.putText(
        vis, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2
    )

    return vis


def draw_history(image, box, centroids, color):
    """
    Draw a bounding box and its historical centroids on an image.

    Args:
        image: An array representing the image to draw on.
        box: A tuple of (x, y, w, h) representing the bounding box to draw.
        centroids: A list of tuples representing the historical centroids of the bounding box.
    """
    # Convert the image to RGB color space
    vis = np.array(image)

    # Draw the bounding box on the image
    x1, y1, x2, y2 = np.intp(box)
    thickness = 2
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

    centroids = np.intp(centroids)
    # Draw the historical centroids on the vis
    for i, centroid in enumerate(centroids):
        if i == 0:
            # Draw the current centroid as a circle
            cv2.circle(vis, centroid, 2, color, thickness=-1)
        else:
            # Draw the historical centroids as lines connecting them
            prev_centroid = centroids[i - 1]
            cv2.line(vis, prev_centroid, centroid, color, thickness=2)

    return vis

def project_bbox_to_camera(bbox, H_src_to_dst):
    """
    Project a bounding box from source camera to destination camera using homography.
    
    Args:
        bbox: Bounding box in format (x1, y1, x2, y2) from source camera
        H_src_to_dst: Homography matrix that transforms points from source to destination camera
        
    Returns:
        Projected bounding box in format (x1, y1, x2, y2) in destination camera coordinates
    """
    x1, y1, x2, y2 = bbox
    
    # Project top-left corner
    p1 = H_src_to_dst @ np.array([x1, y1, 1.0]).reshape(3, 1)
    x1_dst = float(p1[0] / p1[2])
    y1_dst = float(p1[1] / p1[2])
    
    # Project bottom-right corner
    p2 = H_src_to_dst @ np.array([x2, y2, 1.0]).reshape(3, 1)
    x2_dst = float(p2[0] / p2[2])
    y2_dst = float(p2[1] / p2[2])
    
    # Ensure coordinates are in correct order (min, max)
    x1_final = min(x1_dst, x2_dst)
    y1_final = min(y1_dst, y2_dst)
    x2_final = max(x1_dst, x2_dst)
    y2_final = max(y1_dst, y2_dst)
    
    return np.array([x1_final, y1_final, x2_final, y2_final])


def project_bboxes_to_camera(bboxes, H_src_to_dst):
    """
    Project multiple bounding boxes from source camera to destination camera using homography.
    
    Args:
        bboxes: Array of bounding boxes in format (x1, y1, x2, y2) from source camera
        H_src_to_dst: Homography matrix that transforms points from source to destination camera
        
    Returns:
        Array of projected bounding boxes in destination camera coordinates
    """
    projected_bboxes = np.zeros_like(bboxes)
    
    for idx, bbox in enumerate(bboxes):
        projected_bboxes[idx] = project_bbox_to_camera(bbox, H_src_to_dst)
    
    return projected_bboxes

def draw_projected_tracks(image, tracks, ids_dict, H_src_to_dst, color=(255, 0, 255), classes=None):
    """
    Project and draw bounding boxes from source camera to destination camera coordinates.
    
    Args:
        image: Destination camera image to draw on
        tracks: Array of tracks from source camera with shape (N, 5+) where columns are [x1, y1, x2, y2, id, ...]
        ids_dict: Dictionary mapping track IDs to global IDs
        H_src_to_dst: Homography matrix that transforms points from source to destination camera
        color: Color to draw the projected bounding boxes (default: magenta)
        classes: Optional list of class names for labels
        
    Returns:
        Image with projected tracks drawn
    """
    vis = np.array(image)
    
    if len(tracks) == 0:
        return vis
    
    bboxes = tracks[:, :4]
    ids = tracks[:, 4]
    labels = tracks[:, 5] if tracks.shape[1] > 5 else tracks[:, 4]
    
    # Project all bounding boxes at once
    projected_bboxes = project_bboxes_to_camera(bboxes, H_src_to_dst)
    
    # Draw each projected bounding box
    for i, (proj_box, track_id, label) in enumerate(zip(projected_bboxes, ids, labels)):
        global_id = ids_dict[track_id]
        
        # Get projected coordinates
        x1, y1, x2, y2 = np.intp(proj_box)
        
        # Draw the projected bounding box with specified color
        vis = cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=2)
        
        # Draw label
        if classes is None:
            text = f"Proj {int(label)} {global_id}"
        else:
            text = f"Proj {classes[int(label)]} {global_id}"
        
        vis = cv2.putText(
            vis, text, (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2
        )
    
    return vis

def draw_all_projected_tracks(frames, all_tracks, global_ids, homography_dict):
    """
    Draw projected tracks from all cameras onto all other cameras.
    
    Args:
        frames: List of frame images
        all_tracks: List of track arrays for each camera
        global_ids: List of global ID mappings for each camera
        homography_dict: Dictionary mapping (src_cam, dst_cam) -> homography matrix
    
    Returns:
        List of frames with projected tracks drawn
    """
    colors = [
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
        (255, 255, 0),   # Yellow
        (255, 128, 0),   # Orange
        (128, 0, 255),   # Purple
    ]
    
    for dst_cam_idx in range(len(frames)):
        for src_cam_idx in range(len(all_tracks)):
            # Skip same camera
            if src_cam_idx == dst_cam_idx:
                continue
            
            # Skip empty tracks
            if len(all_tracks[src_cam_idx]) == 0:
                continue
            
            # Get homography
            H_key = (src_cam_idx, dst_cam_idx)
            if H_key not in homography_dict:
                continue
            
            # Draw projected tracks
            frames[dst_cam_idx] = draw_projected_tracks(
                frames[dst_cam_idx],
                all_tracks[src_cam_idx],
                global_ids[src_cam_idx],
                homography_dict[H_key],
                color=colors[src_cam_idx % len(colors)]
            )
    
    return frames