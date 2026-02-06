# boxmot/trackers/scsort/scsort.py (Final Reviewed & Patched Version)

from collections import deque
import numpy as np

from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils.matching import iou_distance, linear_assignment
from boxmot.utils.ops import xyxy2xysr


# ... (Helper functions and KalmanBoxTracker class are unchanged from the last version) ...
# PART 1: TOP-LEVEL HELPER FUNCTIONS (UNCHANGED)
def k_previous_obs(observations, cur_age, k, is_obb=False):
    if len(observations) == 0:
        if is_obb:
            return [-1, -1, -1, -1, -1, -1]
        else:
            return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations: return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score is None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


# PART 2: KalmanBoxTracker OBJECT (WITH NUMERICAL STABILITY FIX)
class KalmanBoxTracker(object):
    count = 0

    def __init__(self, bbox, cls, det_ind, delta_t=3, max_obs=50, Q_xy_scaling=0.01, Q_s_scaling=0.0001):
        self.det_ind, self.Q_xy_scaling, self.Q_s_scaling = det_ind, Q_xy_scaling, Q_s_scaling
        self.kf = KalmanFilterXYSR(dim_x=7, dim_z=4, max_obs=max_obs)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.;
        self.kf.P[4:, 4:] *= 1000.;
        self.kf.P *= 10.;
        self.kf.Q[4:6, 4:6] *= self.Q_xy_scaling;
        self.kf.Q[-1, -1] *= self.Q_s_scaling
        self.kf.x[:4] = xyxy2xysr(bbox)
        self.time_since_update = 0;
        self.id = KalmanBoxTracker.count;
        KalmanBoxTracker.count += 1
        self.max_obs = max_obs;
        self.history = deque([], maxlen=self.max_obs);
        self.hits = 0;
        self.hit_streak = 0;
        self.age = 0
        self.conf = bbox[-1];
        self.cls = cls;
        self.last_observation = np.array([-1, -1, -1, -1, -1]);
        self.observations = dict()
        self.history_observations = deque([], maxlen=self.max_obs);
        self.velocity = None;
        self.delta_t = delta_t
        self.home_mean = None;
        self.home_covariance = None;
        self.home_M2 = np.zeros((2, 2));
        self.home_n = 0

    def update_home_model(self, new_center):
        self.home_n += 1
        if self.home_n == 1:
            self.home_mean = new_center; self.home_covariance = np.eye(2) * 100
        else:
            delta = new_center - self.home_mean
            self.home_mean += delta / self.home_n
            delta2 = new_center - self.home_mean
            self.home_M2 += np.outer(delta, delta2)
            if self.home_n > 1:
                regularization_term = np.eye(2) * 1e-6
                self.home_covariance = self.home_M2 / (self.home_n - 1) + regularization_term

    def update(self, bbox, cls, det_ind):
        self.det_ind = det_ind
        if bbox is not None:
            self.conf = bbox[-1];
            self.cls = cls
            if self.last_observation.sum() >= 0:
                previous_box = k_previous_obs(self.observations, self.age, self.delta_t)
                if previous_box.sum() < 0: previous_box = self.last_observation
                self.velocity = speed_direction(previous_box, bbox)
            self.last_observation = bbox;
            self.observations[self.age] = bbox;
            self.history_observations.append(bbox);
            self.time_since_update = 0
            self.hits += 1;
            self.hit_streak += 1;
            self.kf.update(xyxy2xysr(bbox))
            new_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            self.update_home_model(new_center)
        else:
            self.kf.update(bbox)

    def predict(self):
        if ((self.kf.x[6] + self.kf.x[2]) <= 0): self.kf.x[6] *= 0.0
        self.kf.predict();
        self.age += 1
        if (self.time_since_update > 0): self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


class ScSort(BaseTracker):
    def __init__(self, per_class: bool = False, min_conf: float = 0.1, det_thresh: float = 0.2, max_age: int = 30,
                 min_hits: int = 3, asso_threshold: float = 0.1, delta_t: int = 3,
                 use_byte: bool = True, Q_xy_scaling: float = 0.01, Q_s_scaling: float = 0.0001,
                 rescue_dist_thresh: float = 2.0, byte_iou_thresh: float = 0.5):
        super().__init__(max_age=max_age, per_class=per_class)
        self.per_class, self.min_conf, self.max_age, self.min_hits, self.asso_threshold = per_class, min_conf, max_age, min_hits, asso_threshold
        self.frame_count, self.det_thresh, self.delta_t, self.use_byte = 0, det_thresh, delta_t, use_byte
        self.Q_xy_scaling, self.Q_s_scaling = Q_xy_scaling, Q_s_scaling
        KalmanBoxTracker.count = 0

        self.rescue_dist_thresh = rescue_dist_thresh
        self.byte_iou_thresh = byte_iou_thresh

    def _location_cost(self, tracks, detections):
        cost_matrix = np.ones((len(detections), len(tracks)))
        if len(tracks) == 0 or len(detections) == 0: return cost_matrix
        det_centers = np.array([((d[0] + d[2]) / 2, (d[1] + d[3]) / 2) for d in detections])
        for j, track in enumerate(tracks):
            if track.home_n <= 1: continue
            try:
                cov_inv = np.linalg.inv(track.home_covariance)
                delta = det_centers - track.home_mean
                squared_dist = np.sum(np.dot(delta, cov_inv) * delta, axis=1)
                squared_dist[squared_dist < 0] = 0
                mahalanobis_dist = np.sqrt(squared_dist)
                cost = mahalanobis_dist / 3.0
                cost_matrix[:, j] = np.minimum(cost, 1.0)
            except np.linalg.LinAlgError:
                continue
        return cost_matrix

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        self.check_inputs(dets, img)
        self.frame_count += 1

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets[:, 4 + self.is_obb]
        inds_low = confs > self.min_conf;
        inds_high = confs < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high);
        dets_second = dets[inds_second]
        remain_inds = confs > self.det_thresh;
        dets = dets[remain_inds]

        trks = np.zeros((len(self.active_tracks), 5 + self.is_obb))
        to_del = [];
        ret = []
        for t, trk in enumerate(trks):
            pos = self.active_tracks[t].predict()[0]
            trk[:] = [pos[i] for i in range(4 + self.is_obb)] + [0]
            if np.any(np.isnan(pos)): to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        active_trks_for_context = [t for i, t in enumerate(self.active_tracks) if i not in to_del]
        for t in reversed(to_del): self.active_tracks.pop(t)

        # STAGE 1: High-Precision IoU Association
        iou_cost = iou_distance(dets[:, 0: 5 + self.is_obb], trks)
        matched, u_detection, u_track = linear_assignment(iou_cost, thresh=self.asso_threshold)
        for m in matched:
            self.active_tracks[m[1]].update(dets[m[0], :-2], dets[m[0], -2], dets[m[0], -1])

        # STAGE 2: Track-Driven Home Model Rescue
        if len(u_track) > 0 and len(u_detection) > 0:
            rescue_trks_indices = u_track
            rescue_dets_indices = u_detection
            rescue_active_trks = [active_trks_for_context[i] for i in rescue_trks_indices]
            rescue_dets = dets[rescue_dets_indices]
            rescue_cost_matrix = self._location_cost(rescue_active_trks, rescue_dets)
            rescued_matched, _, _ = linear_assignment(rescue_cost_matrix, thresh=self.rescue_dist_thresh)
            if len(rescued_matched) > 0:
                rescued_det_indices_local = np.array([m[0] for m in rescued_matched])
                rescued_trk_indices_local = np.array([m[1] for m in rescued_matched])
                for m in rescued_matched:
                    det_global_idx = rescue_dets_indices[m[0]]
                    trk_global_idx = rescue_trks_indices[m[1]]
                    self.active_tracks[trk_global_idx].update(dets[det_global_idx, :-2], dets[det_global_idx, -2],
                                                              dets[det_global_idx, -1])
                u_detection = np.setdiff1d(u_detection, rescue_dets_indices[rescued_det_indices_local])
                u_track = np.setdiff1d(u_track, rescue_trks_indices[rescued_trk_indices_local])

        # STAGE 3: BYTE Association
        if self.use_byte and len(dets_second) > 0 and len(u_track) > 0:
            u_trks = trks[u_track]
            iou_cost_byte = iou_distance(dets_second[:, 0: 5 + self.is_obb], u_trks)
            matched_indices, _, _ = linear_assignment(iou_cost_byte, thresh=self.byte_iou_thresh)
            to_remove_trk_indices = []
            for m in matched_indices:
                det_ind, trk_ind_local = m[0], m[1]
                trk_ind_global = u_track[trk_ind_local]
                self.active_tracks[trk_ind_global].update(dets_second[det_ind, :-2], dets_second[det_ind, -2],
                                                          dets_second[det_ind, -1])
                to_remove_trk_indices.append(trk_ind_global)
            u_track = np.setdiff1d(u_track, np.array(to_remove_trk_indices))

        # STAGE 4: Re-match Association
        if len(u_detection) > 0 and len(u_track) > 0:
            left_dets = dets[u_detection]
            # --- PATCH: Safely get last_boxes for remaining tracks ---
            active_tracks_for_rematch = [active_trks_for_context[i] for i in u_track]
            last_boxes = np.array([trk.last_observation for trk in active_tracks_for_rematch])

            if len(last_boxes) > 0:
                rematch_thresh = self.asso_threshold + 0.2
                iou_cost_rematch = iou_distance(left_dets[:, 0: 5 + self.is_obb], last_boxes)
                rematched_indices, _, _ = linear_assignment(iou_cost_rematch, thresh=rematch_thresh)
                to_remove_det_indices, to_remove_trk_indices = [], []
                for m in rematched_indices:
                    det_ind_local, trk_ind_local = m[0], m[1]
                    det_ind_global = u_detection[det_ind_local]
                    trk_ind_global = u_track[trk_ind_local]
                    self.active_tracks[trk_ind_global].update(dets[det_ind_global, :-2], dets[det_ind_global, -2],
                                                              dets[det_ind_global, -1])
                    to_remove_det_indices.append(det_ind_global)
                    to_remove_trk_indices.append(trk_ind_global)
                u_detection = np.setdiff1d(u_detection, np.array(to_remove_det_indices))
                u_track = np.setdiff1d(u_track, np.array(to_remove_trk_indices))

        # Finalization
        for m in u_track: self.active_tracks[m].update(None, None, None)
        for i in u_detection:
            trk = KalmanBoxTracker(dets[i, :5], dets[i, 5], dets[i, 6], delta_t=self.delta_t,
                                   Q_xy_scaling=self.Q_xy_scaling, Q_s_scaling=self.Q_s_scaling, max_obs=self.max_obs)
            self.active_tracks.append(trk)
        i = len(self.active_tracks)
        for trk in reversed(self.active_tracks):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4 + self.is_obb]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1], [trk.conf], [trk.cls], [trk.det_ind])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age: self.active_tracks.pop(i)
        if len(ret) > 0: return np.concatenate(ret)
        return np.array([])

# The code above is the original code of SCSORT. However, we have made some modifications to the code to make it work with our
#
# # boxmot/trackers/scsort/scsort.py (Final Robust and Backward-Compatible Version)
#
# from collections import deque
# import numpy as np
# from scipy.spatial.distance import cdist
#
# from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR
# from boxmot.trackers.basetracker import BaseTracker
# from boxmot.utils.matching import iou_distance, linear_assignment
# from boxmot.utils.ops import xyxy2xysr
#
#
# # ... (Helper functions remain the same) ...
# def k_previous_obs(observations, cur_age, k, is_obb=False):
#     if len(observations) == 0:
#         if is_obb:
#             return [-1, -1, -1, -1, -1, -1]
#         else:
#             return [-1, -1, -1, -1, -1]
#     for i in range(k):
#         dt = k - i
#         if cur_age - dt in observations: return observations[cur_age - dt]
#     max_age = max(observations.keys())
#     return observations[max_age]
#
#
# def convert_x_to_bbox(x, score=None):
#     w = np.sqrt(x[2] * x[3])
#     h = x[2] / w
#     if (score is None):
#         return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
#     else:
#         return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))
#
#
# def speed_direction(bbox1, bbox2):
#     cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
#     cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm
#
#
# class KalmanBoxTracker(object):
#     count = 0
#
#     # PATCH: Made `emb` optional with a default of None
#     def __init__(self, bbox, cls, det_ind, emb=None, delta_t=3, max_obs=50, Q_xy_scaling=0.01, Q_s_scaling=0.0001):
#         self.det_ind, self.Q_xy_scaling, self.Q_s_scaling = det_ind, Q_xy_scaling, Q_s_scaling
#         self.kf = KalmanFilterXYSR(dim_x=7, dim_z=4, max_obs=max_obs)
#         self.kf.F = np.array(
#             [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
#              [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
#         self.kf.H = np.array(
#             [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
#         self.kf.R[2:, 2:] *= 10.;
#         self.kf.P[4:, 4:] *= 1000.;
#         self.kf.P *= 10.;
#         self.kf.Q[4:6, 4:6] *= self.Q_xy_scaling;
#         self.kf.Q[-1, -1] *= self.Q_s_scaling
#         self.kf.x[:4] = xyxy2xysr(bbox)
#         self.time_since_update = 0;
#         self.id = KalmanBoxTracker.count;
#         KalmanBoxTracker.count += 1
#         self.max_obs = max_obs;
#         self.history = deque([], maxlen=self.max_obs);
#         self.hits = 0;
#         self.hit_streak = 0;
#         self.age = 0
#         self.conf = bbox[-1];
#         self.cls = cls;
#         self.last_observation = np.array([-1, -1, -1, -1, -1]);
#         self.observations = dict()
#         self.history_observations = deque([], maxlen=self.max_obs);
#         self.velocity = None;
#         self.delta_t = delta_t
#         self.home_mean = None;
#         self.home_covariance = None;
#         self.home_M2 = np.zeros((2, 2));
#         self.home_n = 0
#         self.smooth_feature = None;
#         self.features = deque([], maxlen=self.max_obs)
#         if emb is not None:
#             self.smooth_feature = emb / np.linalg.norm(emb)
#             self.features.append(self.smooth_feature)
#
#     def update_home_model(self, new_center):
#         self.home_n += 1
#         if self.home_n == 1:
#             self.home_mean = new_center; self.home_covariance = np.eye(2) * 100
#         else:
#             delta = new_center - self.home_mean
#             self.home_mean += delta / self.home_n
#             delta2 = new_center - self.home_mean
#             self.home_M2 += np.outer(delta, delta2)
#             if self.home_n > 1:
#                 regularization_term = np.eye(2) * 1e-6
#                 self.home_covariance = self.home_M2 / (self.home_n - 1) + regularization_term
#
#     def update(self, bbox, cls, det_ind, emb=None):
#         self.det_ind = det_ind
#         if bbox is not None:
#             self.conf = bbox[-1];
#             self.cls = cls
#             if self.last_observation.sum() >= 0:
#                 previous_box = k_previous_obs(self.observations, self.age, self.delta_t)
#                 if previous_box.sum() < 0: previous_box = self.last_observation
#                 self.velocity = speed_direction(previous_box, bbox)
#             self.last_observation = bbox;
#             self.observations[self.age] = bbox;
#             self.history_observations.append(bbox);
#             self.time_since_update = 0
#             self.hits += 1;
#             self.hit_streak += 1;
#             self.kf.update(xyxy2xysr(bbox))
#             new_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
#             self.update_home_model(new_center)
#             if emb is not None:
#                 new_feature = emb / np.linalg.norm(emb)
#                 if self.smooth_feature is None:
#                     self.smooth_feature = new_feature
#                 else:
#                     self.smooth_feature = 0.9 * self.smooth_feature + 0.1 * new_feature
#                 self.smooth_feature /= np.linalg.norm(self.smooth_feature)
#                 self.features.append(new_feature)
#         else:
#             self.kf.update(bbox)
#
#     def predict(self):
#         if ((self.kf.x[6] + self.kf.x[2]) <= 0): self.kf.x[6] *= 0.0
#         self.kf.predict();
#         self.age += 1
#         if (self.time_since_update > 0): self.hit_streak = 0
#         self.time_since_update += 1
#         self.history.append(convert_x_to_bbox(self.kf.x))
#         return self.history[-1]
#
#     def get_state(self):
#         return convert_x_to_bbox(self.kf.x)
#
#
# class ScSort(BaseTracker):
#     def __init__(self, per_class: bool = False, min_conf: float = 0.1, det_thresh: float = 0.2, max_age: int = 30,
#                  min_hits: int = 3, asso_threshold: float = 0.1, delta_t: int = 3,
#                  use_byte: bool = True, Q_xy_scaling: float = 0.01, Q_s_scaling: float = 0.0001,
#                  rescue_dist_thresh: float = 2.0, byte_iou_thresh: float = 0.5,
#                  reid_weight: float = 0.3):
#         super().__init__(max_age=max_age, per_class=per_class)
#         self.per_class, self.min_conf, self.max_age, self.min_hits, self.asso_threshold = per_class, min_conf, max_age, min_hits, asso_threshold
#         self.frame_count, self.det_thresh, self.delta_t, self.use_byte = 0, det_thresh, delta_t, use_byte
#         self.Q_xy_scaling, self.Q_s_scaling = Q_xy_scaling, Q_s_scaling
#         KalmanBoxTracker.count = 0
#         self.rescue_dist_thresh = rescue_dist_thresh
#         self.byte_iou_thresh = byte_iou_thresh
#         self.reid_weight = reid_weight
#
#     def _location_cost(self, tracks, detections):
#         cost_matrix = np.ones((len(detections), len(tracks)))
#         if len(tracks) == 0 or len(detections) == 0: return cost_matrix
#         det_centers = np.array([((d[0] + d[2]) / 2, (d[1] + d[3]) / 2) for d in detections])
#         for j, track in enumerate(tracks):
#             if track.home_n <= 1: continue
#             try:
#                 cov_inv = np.linalg.inv(track.home_covariance)
#                 delta = det_centers - track.home_mean
#                 squared_dist = np.sum(np.dot(delta, cov_inv) * delta, axis=1)
#                 squared_dist[squared_dist < 0] = 0
#                 mahalanobis_dist = np.sqrt(squared_dist)
#                 cost = mahalanobis_dist / 3.0
#                 cost_matrix[:, j] = np.minimum(cost, 1.0)
#             except np.linalg.LinAlgError:
#                 continue
#         return cost_matrix
#
#     @BaseTracker.setup_decorator
#     @BaseTracker.per_class_decorator
#     def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
#         self.check_inputs(dets, img)
#         self.frame_count += 1
#
#         # PATCH: Check if embeddings are provided and create a flag
#         use_reid = self.reid_weight > 0 and embs is not None
#
#         if use_reid:
#             dets_embs = embs.copy()
#         else:
#             dets_embs = None
#
#         dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
#         confs = dets[:, 4 + self.is_obb]
#         inds_low = confs > self.min_conf;
#         inds_high = confs < self.det_thresh
#         inds_second = np.logical_and(inds_low, inds_high);
#         dets_second = dets[inds_second]
#         remain_inds = confs > self.det_thresh;
#         dets = dets[remain_inds]
#         if use_reid:
#             dets_embs = dets_embs[remain_inds]
#
#         trks = np.zeros((len(self.active_tracks), 5 + self.is_obb))
#         to_del = [];
#         ret = []
#         for t, trk in enumerate(trks):
#             pos = self.active_tracks[t].predict()[0]
#             trk[:] = [pos[i] for i in range(4 + self.is_obb)] + [0]
#             if np.any(np.isnan(pos)): to_del.append(t)
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         active_trks_for_context = [t for i, t in enumerate(self.active_tracks) if i not in to_del]
#         for t in reversed(to_del): self.active_tracks.pop(t)
#
#         # --- STAGE 1: High-Precision Association (with optional Re-ID) ---
#         # PATCH: Conditional Re-ID logic
#         if use_reid and len(dets) > 0 and len(active_trks_for_context) > 0:
#             trks_embs = np.array([trk.smooth_feature for trk in active_trks_for_context])
#             iou_cost = iou_distance(dets[:, 0: 5 + self.is_obb], trks)
#             reid_cost = cdist(dets_embs, trks_embs, metric='cosine')
#             final_cost = (1 - self.reid_weight) * iou_cost + self.reid_weight * reid_cost
#         else:
#             # Fallback to pure IoU if Re-ID is not used
#             final_cost = iou_distance(dets[:, 0: 5 + self.is_obb], trks)
#
#         matched, u_detection, u_track = linear_assignment(final_cost, thresh=self.asso_threshold)
#
#         for m in matched:
#             det_emb = dets_embs[m[0]] if use_reid else None
#             self.active_tracks[m[1]].update(dets[m[0], :-2], dets[m[0], -2], dets[m[0], -1], emb=det_emb)
#
#         # --- STAGE 2: Track-Driven Home Model Rescue ---
#         if len(u_track) > 0 and len(u_detection) > 0:
#             # ... (Stage 2 logic remains the same) ...
#             rescue_trks_indices = u_track
#             rescue_dets_indices = u_detection
#             rescue_active_trks = [active_trks_for_context[i] for i in rescue_trks_indices]
#             rescue_dets = dets[rescue_dets_indices]
#             rescue_cost_matrix = self._location_cost(rescue_active_trks, rescue_dets)
#             rescued_matched, _, _ = linear_assignment(rescue_cost_matrix, thresh=self.rescue_dist_thresh)
#             if len(rescued_matched) > 0:
#                 rescued_det_indices_local = np.array([m[0] for m in rescued_matched])
#                 rescued_trk_indices_local = np.array([m[1] for m in rescued_matched])
#                 for m in rescued_matched:
#                     det_global_idx = rescue_dets_indices[m[0]]
#                     trk_global_idx = rescue_trks_indices[m[1]]
#                     det_emb = dets_embs[det_global_idx] if use_reid else None
#                     self.active_tracks[trk_global_idx].update(dets[det_global_idx, :-2], dets[det_global_idx, -2],
#                                                               dets[det_global_idx, -1], emb=det_emb)
#                 u_detection = np.setdiff1d(u_detection, rescue_dets_indices[rescued_det_indices_local])
#                 u_track = np.setdiff1d(u_track, rescue_trks_indices[rescued_trk_indices_local])
#
#         # ... (Stage 3 & 4 with patches) ...
#         if self.use_byte and len(dets_second) > 0 and len(u_track) > 0:
#             u_trks = trks[u_track]
#             iou_cost_byte = iou_distance(dets_second[:, 0: 5 + self.is_obb], u_trks)
#             matched_indices, _, _ = linear_assignment(iou_cost_byte, thresh=self.byte_iou_thresh)
#             to_remove_trk_indices = []
#             for m in matched_indices:
#                 det_ind, trk_ind_local = m[0], m[1]
#                 trk_ind_global = u_track[trk_ind_local]
#                 self.active_tracks[trk_ind_global].update(dets_second[det_ind, :-2], dets_second[det_ind, -2],
#                                                           dets_second[det_ind, -1], emb=None)
#                 to_remove_trk_indices.append(trk_ind_global)
#             u_track = np.setdiff1d(u_track, np.array(to_remove_trk_indices))
#
#         if len(u_detection) > 0 and len(u_track) > 0:
#             left_dets = dets[u_detection]
#             active_tracks_for_rematch = [active_trks_for_context[i] for i in u_track]
#             last_boxes = np.array([trk.last_observation for trk in active_tracks_for_rematch])
#             if len(last_boxes) > 0:
#                 rematch_thresh = self.asso_threshold + 0.2
#                 iou_cost_rematch = iou_distance(left_dets[:, 0: 5 + self.is_obb], last_boxes)
#                 rematched_indices, _, _ = linear_assignment(iou_cost_rematch, thresh=rematch_thresh)
#                 to_remove_det_indices, to_remove_trk_indices = [], []
#                 for m in rematched_indices:
#                     det_ind_local, trk_ind_local = m[0], m[1]
#                     det_ind_global = u_detection[det_ind_local]
#                     trk_ind_global = u_track[trk_ind_local]
#                     det_emb = dets_embs[det_ind_global] if use_reid else None
#                     self.active_tracks[trk_ind_global].update(dets[det_ind_global, :-2], dets[det_ind_global, -2],
#                                                               dets[det_ind_global, -1], emb=det_emb)
#                     to_remove_det_indices.append(det_ind_global)
#                     to_remove_trk_indices.append(trk_ind_global)
#                 u_detection = np.setdiff1d(u_detection, np.array(to_remove_det_indices))
#                 u_track = np.setdiff1d(u_track, np.array(to_remove_trk_indices))
#
#         # Finalization
#         for m in u_track: self.active_tracks[m].update(None, None, None, emb=None)
#         for i in u_detection:
#             det_emb = dets_embs[i] if use_reid else None
#             trk = KalmanBoxTracker(dets[i, :5], dets[i, 5], dets[i, 6], emb=det_emb, delta_t=self.delta_t,
#                                    Q_xy_scaling=self.Q_xy_scaling, Q_s_scaling=self.Q_s_scaling, max_obs=self.max_obs)
#             self.active_tracks.append(trk)
#         i = len(self.active_tracks)
#         for trk in reversed(self.active_tracks):
#             if trk.last_observation.sum() < 0:
#                 d = trk.get_state()[0]
#             else:
#                 d = trk.last_observation[:4 + self.is_obb]
#             if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#                 ret.append(np.concatenate((d, [trk.id + 1], [trk.conf], [trk.cls], [trk.det_ind])).reshape(1, -1))
#             i -= 1
#             if trk.time_since_update > self.max_age: self.active_tracks.pop(i)
#         if len(ret) > 0: return np.concatenate(ret)
#         return np.array([])