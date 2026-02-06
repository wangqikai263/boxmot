# boxmot/trackers/deepscsort/deepscsort_duo.py

from collections import deque
from pathlib import Path
import numpy as np
import torch

# DeepScSort specific imports
from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils.matching import iou_distance, linear_assignment

# Helper functions (unchanged from ScSort)
from boxmot.utils.ops import xyxy2xysr, xyxy2xywh

def convert_x_to_bbox(x, score=None):
    # --- START OF FIX ---
    # 确保面积和长宽比的乘积不会是负数
    # 这是防止在长期预测后状态漂移导致sqrt(负数)的关键
    s_times_r = x[2] * x[3]
    if s_times_r < 0:
        s_times_r = 0  # 如果为负，则强制设为0
    # --- END OF FIX ---

    w = np.sqrt(s_times_r) # 现在这里永远是安全的
    h = x[2] / w if w > 0 else 0 # 增加一个保护，防止除以0

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


def reid_distance(dets_embs, trks_embs):
    """
    Calculates the cosine distance between detection and track embeddings.
    Assumes embeddings are normalized.
    """
    if len(dets_embs) == 0 or len(trks_embs) == 0:
        return np.empty((len(dets_embs), len(trks_embs)))
    # Cosine similarity is the dot product of normalized vectors
    sim = np.dot(dets_embs, trks_embs.T)
    # Cosine distance is 1 - similarity
    dist = 1.0 - sim
    return dist


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    It has been updated to handle appearance features for DeepScSort.
    """
    count = 0

    def __init__(self, bbox, cls, det_ind, feature=None, delta_t=3, max_obs=50, Q_xy_scaling=0.01, Q_s_scaling=0.0001):
        self.det_ind, self.Q_xy_scaling, self.Q_s_scaling = det_ind, Q_xy_scaling, Q_s_scaling
        self.kf = KalmanFilterXYSR(dim_x=7, dim_z=4, max_obs=max_obs)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 3.;
        self.kf.P[4:, 4:] *= 1000.;
        self.kf.P *= 10.
        self.kf.Q[4:6, 4:6] *= self.Q_xy_scaling;
        self.kf.Q[-1, -1] *= self.Q_s_scaling
        self.kf.x[:4] = xyxy2xysr(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count;
        KalmanBoxTracker.count += 1
        self.age = 0
        self.hits = 0;
        self.hit_streak = 0

        self.conf = bbox[-1];
        self.cls = cls
        self.delta_t = delta_t
        self.max_obs = max_obs

        # History and observations for motion/location
        self.history = deque([], maxlen=self.max_obs)
        self.observations = dict()
        self.last_observation = np.array([-1, -1, -1, -1, -1])

        # Home Model attributes
        self.home_mean = None;
        self.home_covariance = None
        self.home_M2 = np.zeros((2, 2));
        self.home_n = 0

        # Appearance feature attribute (for DeepScSort)
        self.smooth_feature = feature
        if self.smooth_feature is not None:
            self.smooth_feature /= np.linalg.norm(self.smooth_feature)

    def update_feature(self, feature, alpha=0.9):
        """
        Updates the smoothed appearance feature using Exponential Moving Average (EMA).
        """
        if self.smooth_feature is None:
            self.smooth_feature = feature
        else:
            self.smooth_feature = alpha * self.smooth_feature + (1 - alpha) * feature
        self.smooth_feature /= np.linalg.norm(self.smooth_feature)

    def update_home_model(self, new_center):
        self.home_n += 1
        if self.home_n == 1:
            self.home_mean = new_center;
            self.home_covariance = np.eye(2) * 100
        else:
            delta = new_center - self.home_mean
            self.home_mean += delta / self.home_n
            delta2 = new_center - self.home_mean
            self.home_M2 += np.outer(delta, delta2)
            if self.home_n > 1:
                self.home_covariance = self.home_M2 / (self.home_n - 1) + np.eye(2) * 1e-6

    def update(self, bbox, cls, det_ind, feature=None):
        self.det_ind = det_ind
        if bbox is not None:
            self.conf = bbox[-1];
            self.cls = cls
            self.last_observation = bbox;
            self.observations[self.age] = bbox
            self.time_since_update = 0
            self.hits += 1;
            self.hit_streak += 1
            self.kf.update(xyxy2xysr(bbox))

            new_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            self.update_home_model(new_center)

            # Update appearance feature if provided
            if feature is not None:
                self.update_feature(feature)
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


#
# ... 文件顶部的 import 语句和其他类（如 KalmanBoxTracker）保持不变 ...
#

class DeepScSort(BaseTracker):
    def __init__(
            self,
            reid_weights: Path,
            device: torch.device,
            half: bool,
            per_class: bool = False,
            min_conf: float = 0.1,
            det_thresh: float = 0.2,
            max_age: int = 30,
            min_hits: int = 3,
            reid_asso_thresh: float = 0.25,
            # --- START OF MODIFICATION 1 ---
            # 新增融合权重参数，用于平衡IoU代价和Re-ID代价
            # 0.3 代表 IoU 占 30% 权重, Re-ID 占 70% 权重
            cost_fusion_alpha: float = 0.2,
            # --- END OF MODIFICATION 1 ---
            rescue_dist_thresh: float = 3.5,
            byte_iou_thresh: float = 0.5,
            delta_t: int = 3,
            use_byte: bool = True,
            Q_xy_scaling: float = 0.01,
            Q_s_scaling: float = 0.0001,
    ):
        super().__init__(max_age=max_age, per_class=per_class)
        self.per_class, self.min_conf, self.max_age, self.min_hits = per_class, min_conf, max_age, min_hits
        self.frame_count, self.det_thresh, self.delta_t, self.use_byte = 0, det_thresh, delta_t, use_byte
        self.Q_xy_scaling, self.Q_s_scaling = Q_xy_scaling, Q_s_scaling
        KalmanBoxTracker.count = 0

        self.reid_asso_thresh = reid_asso_thresh
        # --- START OF MODIFICATION 1 ---
        # 将新增的参数赋值给类属性
        self.cost_fusion_alpha = cost_fusion_alpha
        # --- END OF MODIFICATION 1 ---
        self.rescue_dist_thresh = rescue_dist_thresh
        self.byte_iou_thresh = byte_iou_thresh

        self.reid_model = ReidAutoBackend(weights=reid_weights, device=device, half=half).model

    def _mahalanobis_cost_2d(self, tracks, detections):
        """
        Calculates a robust, 2D Mahalanobis distance that is uncertainty-aware.
        It only considers the (x, y) center position, avoiding instability from scale and aspect ratio.
        """
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((len(detections), len(tracks)))

        # Get the (x, y) centers of all detections using the imported helper function
        det_centers = xyxy2xywh(detections[:, :4])[:, :2]

        cost_matrix = np.zeros((len(detections), len(tracks)))
        for j, track in enumerate(tracks):
            # Project the track's state into measurement space using the dependency we confirmed
            mean, cov = track.kf.project()

            # Extract only the (x, y) parts of the mean and covariance
            mean_xy = mean[:2]
            cov_xy = cov[:2, :2]

            try:
                # Invert the 2x2 covariance matrix
                cov_inv_xy = np.linalg.inv(cov_xy)

                # Calculate the difference between detection centers and the projected track center
                delta_xy = det_centers - mean_xy.T

                # Calculate the squared Mahalanobis distance in 2D
                mahalanobis_dist_sq = np.sum(np.dot(delta_xy, cov_inv_xy) * delta_xy, axis=1)

                # Normalize using the Chi-squared distribution 95% confidence threshold
                # for 2 degrees of freedom (which is 5.991)
                cost = mahalanobis_dist_sq / 5.991
                # Apply gating: any distance beyond the 95% confidence is considered max cost
                cost[cost > 1.0] = 1.0
                cost_matrix[:, j] = cost

            except np.linalg.LinAlgError:
                # If the covariance is singular, it's a bad state, assign max cost
                cost_matrix[:, j] = 1.0

        return cost_matrix
    def _location_cost(self, tracks, detections):
        # ... 此方法无需修改 ...
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

        # ... Pre-processing, Feature Extraction, Predict track locations 等部分无需修改 ...
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets[:, 4 + self.is_obb]
        remain_inds = confs > self.det_thresh
        inds_low = confs > self.min_conf;
        inds_high = confs < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_high = dets[remain_inds]
        dets_second = dets[inds_second]
        if embs is not None:
            dets_embs = embs[remain_inds]
        else:
            dets_embs = self.reid_model.get_features(dets_high[:, 0:4], img) if len(dets_high) > 0 else np.array([])
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
        trks_embs = np.array([t.smooth_feature for t in active_trks_for_context if t.smooth_feature is not None])

        # --- START OF DYNAMIC COST INTEGRATION ---
        # STAGE 1 & 2: Fused Cost Association with Dynamic 2D Motion Cost

        # 1. 使用全新的、更鲁棒的2D马氏距离作为动态运动代价
        # 注意：这里传入的是 active_trks_for_context，它包含了正确的KalmanBoxTracker实例列表
        motion_cost = self._mahalanobis_cost_2d(active_trks_for_context, dets_high)

        # 2. 外观代价计算保持不变
        reid_cost = reid_distance(dets_embs, trks_embs)

        # 3. 融合“动态运动代价”和“外观代价”
        fused_cost = self.cost_fusion_alpha * motion_cost + (1 - self.cost_fusion_alpha) * reid_cost

        cost_matrix = fused_cost

        # 4. 使用匈牙利算法进行线性分配
        # 注意：由于代价函数的改变，reid_asso_thresh 将需要重新调优
        matched, u_detection, u_track = linear_assignment(cost_matrix, thresh=self.reid_asso_thresh)
        # --- END OF DYNAMIC COST INTEGRATION ---

        for m in matched:
            self.active_tracks[m[1]].update(dets_high[m[0], :-2], dets_high[m[0], -2], dets_high[m[0], -1],
                                            dets_embs[m[0]])

        # ... STAGE 3: Home Model Rescue, BYTE Association, Finalization 等后续部分无需修改 ...
        if len(u_track) > 0 and len(u_detection) > 0:
            rescue_trks_indices = u_track
            rescue_dets_indices = u_detection
            rescue_active_trks = [active_trks_for_context[i] for i in rescue_trks_indices]
            rescue_dets = dets_high[rescue_dets_indices]
            rescue_cost_matrix = self._location_cost(rescue_active_trks, rescue_dets)
            rescued_matched, _, _ = linear_assignment(rescue_cost_matrix, thresh=self.rescue_dist_thresh)
            if len(rescued_matched) > 0:
                rescued_det_indices_local = np.array([m[0] for m in rescued_matched])
                rescued_trk_indices_local = np.array([m[1] for m in rescued_matched])
                for m in rescued_matched:
                    det_global_idx = rescue_dets_indices[m[0]]
                    trk_global_idx = rescue_trks_indices[m[1]]
                    self.active_tracks[trk_global_idx].update(dets_high[det_global_idx, :-2],
                                                              dets_high[det_global_idx, -2],
                                                              dets_high[det_global_idx, -1], feature=None)
                u_detection = np.setdiff1d(u_detection, rescue_dets_indices[rescued_det_indices_local])
                u_track = np.setdiff1d(u_track, rescue_trks_indices[rescued_trk_indices_local])
        if self.use_byte and len(dets_second) > 0 and len(u_track) > 0:
            u_trks_boxes = trks[u_track]
            iou_cost_byte = iou_distance(dets_second[:, 0: 5 + self.is_obb], u_trks_boxes)
            matched_indices, _, _ = linear_assignment(iou_cost_byte, thresh=self.byte_iou_thresh)
            to_remove_trk_indices = []
            for m in matched_indices:
                trk_ind_global = u_track[m[1]]
                self.active_tracks[trk_ind_global].update(dets_second[m[0], :-2], dets_second[m[0], -2],
                                                          dets_second[m[0], -1], feature=None)
                to_remove_trk_indices.append(trk_ind_global)
            u_track = np.setdiff1d(u_track, np.array(to_remove_trk_indices))
        for m in u_track: self.active_tracks[m].update(None, None, None, None)
        for i in u_detection:
            trk = KalmanBoxTracker(
                dets_high[i, :5], dets_high[i, 5], dets_high[i, 6],
                feature=dets_embs[i],
                delta_t=self.delta_t,
                Q_xy_scaling=self.Q_xy_scaling, Q_s_scaling=self.Q_s_scaling, max_obs=self.max_obs
            )
            self.active_tracks.append(trk)
        i = len(self.active_tracks)
        for trk in reversed(self.active_tracks):
            d = trk.last_observation[:4 + self.is_obb] if trk.last_observation.sum() >= 0 else trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1], [trk.conf], [trk.cls], [trk.det_ind])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age: self.active_tracks.pop(i)
        if len(ret) > 0: return np.concatenate(ret)
        return np.array([])
