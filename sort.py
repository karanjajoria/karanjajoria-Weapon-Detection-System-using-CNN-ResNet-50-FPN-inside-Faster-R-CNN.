# Lightweight SORT tracker adapted for clarity (uses bounding boxes [x1,y1,x2,y2]) and simple IOU matching.
# This is NOT a production-grade implementation but works for small demos.
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2-xx1)
    h = np.maximum(0., yy2-yy1)
    inter = w*h
    area1 = (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    area2 = (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1])
    return inter/(area1+area2-inter+1e-6)

class Track:
    def __init__(self, bbox, track_id):
        self.bbox = np.array(bbox, dtype=float)
        self.id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

class Sort:
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, dets):
        # dets: list of [x1,y1,x2,y2,score]
        trks = [t.bbox for t in self.tracks]
        matched, unmatched_dets, unmatched_trks = [], list(range(len(dets))), list(range(len(trks)))
        if len(trks)>0 and len(dets)>0:
            iou_matrix = np.zeros((len(trks), len(dets)), dtype=float)
            for t, trk in enumerate(trks):
                for d, det in enumerate(dets):
                    iou_matrix[t, d] = iou(trk, det[:4])
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r,c in zip(row_ind, col_ind):
                if iou_matrix[r,c] < self.iou_threshold:
                    continue
                matched.append((r,c))
                unmatched_trks.remove(r)
                unmatched_dets.remove(c)
        # Update matched
        for r,c in matched:
            self.tracks[r].bbox = np.array(dets[c][:4], dtype=float)
            self.tracks[r].hits += 1
            self.tracks[r].time_since_update = 0
        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            tr = Track(np.array(dets[i][:4], dtype=float), self.next_id)
            self.next_id += 1
            self.tracks.append(tr)
        # Age unmatched tracks and remove old ones
        for i in sorted(unmatched_trks, reverse=True):
            tr = self.tracks[i]
            tr.time_since_update += 1
            tr.age += 1
            if tr.time_since_update > self.max_age:
                self.tracks.pop(i)
        # return list of active tracks as (id, bbox)
        return [(t.id, t.bbox.tolist()) for t in self.tracks]
