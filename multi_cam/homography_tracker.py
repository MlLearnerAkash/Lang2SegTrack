import numpy as np
# from sort import associate_detections_to_trackers
def linear_assignment(cost_matrix):
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def modify_bbox_source(bboxes, homography):
    """
    Modify the source of bounding boxes.

    Args:
        bboxes (np.ndarray): Bounding boxes to modify.
        H (np.ndarray): Homography matrix.
    Returns:
        np.ndarray: Modified bounding boxes.
    """

    bboxes_ = list()

    for bbox in bboxes:
        x0, y0, x1, y1, *keep = bbox

        p0 = np.dot(homography, np.array([x0, y0, 1]).reshape(3, 1)).reshape(3)
        p1 = np.dot(homography, np.array([x1, y1, 1]).reshape(3, 1)).reshape(3)

        x0 = int(p0[0] / p0[-1])
        y0 = int(p0[1] / p0[-1])
        x1 = int(p1[0] / p1[-1])
        y1 = int(p1[1] / p1[-1])
        bboxes_.append([x0, y0, x1, y1] + keep)

    return np.asarray(bboxes_)


class MultiCameraTracker:
    def __init__(self, homographies: list, iou_thres=0.2):
        """
        Multi Camera Tracking class contructor.
        """
        self.num_sources = len(homographies)
        self.homographies = homographies
        self.iou_thres = iou_thres
        self.next_id = 1

        self.ids = [{} for _ in range(self.num_sources)]
        self.age = [{} for _ in range(self.num_sources)]

    def update(self, tracks: list):
        # Project tracks to a common reference
        proj_tracks = []
        for i, trks in enumerate(tracks):
            proj_tracks.append(modify_bbox_source(trks, self.homographies[i]))

        # For each pair of sources
        for i in range(self.num_sources):
            for j in range(i + 1, self.num_sources):
                # Match tracks with IOU
                matched = {}
                matches, unmatches_i, unmatches_j = associate_detections_to_trackers(
                    proj_tracks[i], proj_tracks[j], iou_threshold=self.iou_thres
                )

                # Set global ids for the matched tracks
                for idx_i, idx_j in matches:
                    # Ids
                    id_i = proj_tracks[i][idx_i][4]
                    id_j = proj_tracks[j][idx_j][4]
                    # Current match ids
                    match_i = self.ids[i].get(id_i)
                    match_j = self.ids[j].get(id_j)

                    # If track i has a global id and is older then track j
                    if (
                        match_i != None
                        and self.age[i].get(id_i, 0) >= self.age[j].get(id_j, 0)
                        and not matched.get(match_i, False)
                    ):
                        self.ids[j][id_j] = match_i
                        matched[match_i] = True
                    # Else if track j has a global id
                    elif match_j != None and not matched.get(match_j, False):
                        self.ids[i][id_i] = match_j
                        matched[match_j] = True
                    # None of them has a global id
                    else:
                        self.ids[i][id_i] = self.next_id
                        self.ids[j][id_j] = self.next_id
                        matched[self.next_id] = True
                        self.next_id += 1

                    # Increment tracks age
                    self.age[i][id_i] = self.age[i].get(id_i, 0) + 1
                    self.age[j][id_j] = self.age[j].get(id_j, 0) + 1

                # Set global ids for unmatched tracks
                for idx_i in unmatches_i:
                    id_i = proj_tracks[i][idx_i][4]
                    match_i = self.ids[i].get(id_i)

                    if match_i == None or matched.get(match_i, False):
                        self.ids[i][id_i] = self.next_id
                        matched[self.next_id] = True
                        self.next_id += 1

                    # Increment track age
                    self.age[i][id_i] = self.age[i].get(id_i, 0) + 1

                for idx_j in unmatches_j:
                    id_j = proj_tracks[j][idx_j][4]
                    match_j = self.ids[j].get(id_j)

                    if match_j == None or matched.get(match_j, False):
                        self.ids[j][id_j] = self.next_id
                        matched[self.next_id] = True
                        self.next_id += 1

                    # Increment track age
                    self.age[j][id_j] = self.age[j].get(id_j, 0) + 1

        return self.ids
