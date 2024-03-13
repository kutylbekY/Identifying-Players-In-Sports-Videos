import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']

def is_close(coord1, coord2, threshold=50):
    """Check if coordinates are within a certain distance threshold."""
    dist = np.linalg.norm(np.array(coord1[:2]) - np.array(coord2[:2]))  # Simple distance check based on the top-left corner
    return dist < threshold

def dist_calc(coord1, coord2):
    """Check if coordinates are within a certain distance threshold."""
    dist = np.linalg.norm(np.array(coord1[:2]) - np.array(coord2[:2]))  # Simple distance check based on the top-left corner
    return dist

class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        self.available_ids = set(range(20))  # Initialize available IDs
        self.dead_ids = set()  # Initialize an empty set for dead IDs
        self.prev_ids = set()  # Initialize an empty set for dead IDs
        self.last_known_coords = {}  # Store the last known coordinates of each ID
        self.track_ids = {}

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img):
    # def update(self, bbox_xywh, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        # bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_xywh[i], conf, features[i]) for i, conf in enumerate(
            confidences) if conf > self.min_confidence]
        # detections = [Detection(bbox_tlwh[i], 1, features[i]) for i in range(len(bbox_tlwh))]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        
        # available_track_ids = set(range(16))  # IDs from 0 to 15
        curr_ids = set()
        new_ids = set()
        local_dead_ids = set()
        used = set()
        
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            track_id = track.track_id
            track_box = track.to_tlwh()
            track_coord = self._tlwh_to_xyxy(track_box)

            if track_id < 0 or track_id > 15:
                if self.available_ids:
                    track_id = self.available_ids.pop()
                    curr_ids.add(track_id)
                    self.last_known_coords[track_id] = track_coord
            else:
                curr_ids.add(track_id)
                self.last_known_coords[track_id] = track_coord

        if (len(self.prev_ids) != 0):
            local_dead_ids = self.prev_ids - curr_ids
            new_ids = curr_ids - self.prev_ids

        for dead_id in local_dead_ids:
            self.dead_ids.add(dead_id)

        # Pre-update loop to adjust available_track_ids based on current tracks
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            track_id = track.track_id
            track_box = track.to_tlwh()
            track_coord = self._tlwh_to_xyxy(track_box)

            # if track_id < 0 or track_id > 15:
            #     if not self.available_ids:
            #         for dead_id in list(self.dead_ids):
            #             tmp = min_dist
            #             min_dist = min(min_dist, dist_calc(self.last_known_coords[new_id], self.last_known_coords[dead_id]))

            #             if (min_dist != tmp):
            #                 min_id = dead_id
            if track_id < 0 or track_id > 15:
                # print("dead_ids: ", self.dead_ids)
                # print("curr_ids: ", curr_ids)
                # print("prev_ids: ", self.prev_ids)
                min_dist = 999999
                min_id = 999999

                for dead_id in list(self.dead_ids):
                    tmp_dist = dist_calc(track_coord, self.last_known_coords[dead_id])

                    if (min_dist > tmp_dist and (dead_id not in used)):
                        min_dist = tmp_dist
                        min_id = dead_id

                if (min_id != 999999):
                    # if is_close(self.last_known_coords[new_id], self.last_known_coords[min_id]):
                    self.dead_ids.remove(min_id)
                    curr_ids.add(min_id)
                    # track.track_id = min_id
                    self.track_ids[track_id] = min_id
                    self.last_known_coords[min_id] = track_coord
                    used.add(min_id)
            elif track_id in new_ids:
                # print("dead_ids: ", self.dead_ids)
                # print("prev_ids: ", self.prev_ids)
                # print("curr_ids: ", curr_ids)
                # print("new_ids: ", new_ids)
                # print("used: ", used)
                # print("last_known_coords: ", self.last_known_coords)
                # print("track_id: ", track_id)
                # assigned = False
                # track_coord_ = self.last_known_coords[track_id]
                for new_id in new_ids:
                    if (track_id == new_id):
                        min_dist = 999999
                        min_id = 999999

                        for dead_id in list(self.dead_ids):
                            # tmp_dist = dist_calc(self.last_known_coords[new_id], self.last_known_coords[dead_id])

                            # if (min_dist > tmp_dist and (dead_id not in used)):
                            #     min_dist = tmp_dist
                            #     min_id = dead_id

                            tmp = min_dist
                            min_dist = min(min_dist, dist_calc(self.last_known_coords[new_id], self.last_known_coords[dead_id]))

                            if (min_dist != tmp):
                                min_id = dead_id

                        if (min_id != 999999):
                            # if is_close(self.last_known_coords[new_id], self.last_known_coords[min_id]):
                            self.dead_ids.remove(min_id)
                            self.available_ids.add(new_id)
                            curr_ids.remove(new_id)
                            curr_ids.add(min_id)
                            # track.track_id = min_id
                            self.track_ids[track_id] = min_id
                            used.add(min_id)

                            del self.last_known_coords[new_id]
                            self.last_known_coords[min_id] = track_coord
                            # assigned = True
                            break
                # if not assigned and new_id in self.available_ids:
                #     self.available_ids.remove(new_id)

            # if id >= 0 and id <= 15:
            #     if id in available_track_ids:
            #         available_track_ids.remove(id)
            # self.last_known_coords[id] = self._tlwh_to_xyxy(track.to_tlwh())

        self.prev_ids = curr_ids.copy()
        used = set()

        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id

            if ((track_id in self.track_ids) and (self.track_ids[track_id] not in used)):
                track_id = self.track_ids[track_id]
            used.add(track_id)
            # x1, y1, x2, y2 = self.last_known_coords[track_id]

            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=int))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
