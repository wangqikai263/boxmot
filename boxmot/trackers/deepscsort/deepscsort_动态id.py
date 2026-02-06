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
from boxmot.utils.ops import xyxy2xysr


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
        # if (self.time_since_update > 0): self.hit_streak = 0 # <-- 已删除或注释掉
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
            # --- 以下为原有参数，保持不变 ---
            min_conf: float = 0.1,
            det_thresh: float = 0.2,
            max_age: int = 80,
            min_hits: int = 3,
            iou_gate_thresh: float = 0.8,  # 注意：这个在原代码中没有，但iou_distance在用，最好加上
            reid_asso_thresh: float = 0.5,
            cost_fusion_alpha: float = 0.2,
            rescue_dist_thresh: float = 2.0,
            byte_iou_thresh: float = 0.5,
            delta_t: int = 3,
            use_byte: bool = True,
            Q_xy_scaling: float = 0.01,
            Q_s_scaling: float = 0.0001,
            # --- START OF ID POOL MODIFICATION 1 ---
            # --- 以下为新增的、控制ID池和Re-entry的参数 ---
            max_students: int = 40,  # 预设的课堂最大ID容量
            id_recycle_age: int = 300,  # Lost状态的ID多久后可被回收为Available (e.g., 10s @ 30fps)
            reentry_beta: float = 0.4,  #   # Re-entry阶段，HomeModel位置代价的权重
            # --- START OF FINAL FIX ---
            reentry_reid_gate_thresh: float = 1.0  # 外观相似度的“一票否决”门槛
            # --- END OF FINAL FIX ---
            # --- END OF ID POOL MODIFICATION 1 ---
    ):
        # --- 以下为赋值语句的重构 ---
        super().__init__(max_age=max_age, per_class=per_class)
        self.per_class, self.min_conf, self.max_age, self.min_hits = per_class, min_conf, max_age, min_hits
        self.frame_count, self.det_thresh, self.delta_t, self.use_byte = 0, det_thresh, delta_t, use_byte
        self.Q_xy_scaling, self.Q_s_scaling = Q_xy_scaling, Q_s_scaling

        # 关联和代价相关的阈值
        self.iou_gate_thresh = iou_gate_thresh
        self.reid_asso_thresh = reid_asso_thresh
        self.cost_fusion_alpha = cost_fusion_alpha
        self.rescue_dist_thresh = rescue_dist_thresh
        self.byte_iou_thresh = byte_iou_thresh

        self.reid_model = ReidAutoBackend(weights=reid_weights, device=device, half=half).model

        # --- START OF ID POOL MODIFICATION 1 ---
        # --- 初始化动态ID池 ---
        self.id_recycle_age = id_recycle_age
        self.reentry_beta = reentry_beta
        # --- START OF FINAL FIX ---
        self.reentry_reid_gate_thresh = reentry_reid_gate_thresh
        # --- END OF FINAL FIX ---

        # 不再使用 KalmanBoxTracker.count 来生成ID
        KalmanBoxTracker.count = 0

        # 初始化ID池，预先创建所有可能的学生ID
        self.id_pool = []
        for i in range(1, max_students + 1):
            self.id_pool.append({
                "id": i,
                "status": "Available",  # ID的初始状态为“可用”
                "tracker": None,  # 当前没有关联的短期跟踪器
                "home_model": {  # 长期Home Model信息
                    "mean": None,
                    "covariance": None,
                    "n": 0
                },
                # --- START OF FIX ---
                "hit_streak": 0,  # 为长期身份增加功绩记录
                # --- END OF FIX ---
                "last_feature": None,  # 最后的有效外观特征
                "age_lost": 0  # 处于Lost状态的帧数
            })
        # self.active_tracks 列表将被动态生成，不再是主要存储
        # --- END OF ID POOL MODIFICATION 1 ---

    def _location_cost_for_pool(self, lost_ids, detections):
        cost_matrix = np.ones((len(detections), len(lost_ids)))
        if len(lost_ids) == 0 or len(detections) == 0:
            return cost_matrix

        det_centers = np.array([((d[0] + d[2]) / 2, (d[1] + d[3]) / 2) for d in detections])
        for j, id_data in enumerate(lost_ids):
            home_model = id_data['home_model']
            if home_model['n'] <= 1:
                continue
            try:
                cov_inv = np.linalg.inv(home_model['covariance'])
                delta = det_centers - home_model['mean']
                squared_dist = np.sum(np.dot(delta, cov_inv) * delta, axis=1)
                squared_dist[squared_dist < 0] = 0
                mahalanobis_dist = np.sqrt(squared_dist)
                cost = mahalanobis_dist / 3.0  # Normalize for gating
                cost_matrix[:, j] = np.minimum(cost, 1.0)
            except np.linalg.LinAlgError:
                continue
        return cost_matrix

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        self.check_inputs(dets, img)
        self.frame_count += 1

        # --- START OF ID POOL MODIFICATION 2 ---
        # --- 1. 数据预处理和特征提取 ---
        dets_with_idx = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets_with_idx[:, 4 + self.is_obb]

        # 筛选高置信度和低置信度的检测框
        remain_inds = confs > self.det_thresh
        dets_high = dets_with_idx[remain_inds]

        inds_low = confs > self.min_conf
        inds_high = confs < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets_with_idx[inds_second]

        # 为高置信度检测框提取Re-ID特征
        if embs is not None:
            dets_embs = embs[remain_inds]
        else:
            dets_embs = self.reid_model.get_features(dets_high[:, 0:4], img) if len(dets_high) > 0 else np.array([])

        # --- 2. 从ID池中准备轨迹数据并执行预测 ---

        # 动态筛选出活跃和失追的ID
        active_ids = [p for p in self.id_pool if p['status'] == 'Active']
        lost_ids = [p for p in self.id_pool if p['status'] == 'Lost']

        # 对活跃轨迹进行预测
        predicted_trks = []
        for id_data in active_ids:
            pos = id_data['tracker'].predict()[0]
            if not np.any(np.isnan(pos)):
                predicted_trks.append(np.concatenate((pos, [0])))  # 兼容iou_distance的格式
        predicted_trks = np.array(predicted_trks)

        # 准备活跃轨迹的Re-ID特征用于匹配
        active_trks_embs = np.array(
            [p['tracker'].smooth_feature for p in active_ids if p['tracker'].smooth_feature is not None])

        ret = []  # 用于存储最终输出结果
        # --- END OF ID POOL MODIFICATION 2 ---

        # --- START OF FINAL FIX - UNIFIED GLOBAL MATCHING ---
        # --- 3. 核心数据关联：统一匹配阶段 ---

        # a. 准备候选者：合并 Active 和 Lost 轨迹
        # ----------------------------------------------------
        active_ids = [p for p in self.id_pool if p['status'] == 'Active']
        lost_ids = [p for p in self.id_pool if p['status'] == 'Lost']

        # 所有候选者ID
        candidate_ids = active_ids + lost_ids

        unmatched_dets_high_indices = np.arange(len(dets_high))

        if len(candidate_ids) > 0 and len(unmatched_dets_high_indices) > 0:

            # b. 构建统一代价矩阵
            # --------------------------
            num_dets = len(unmatched_dets_high_indices)
            num_candidates = len(candidate_ids)
            cost_matrix = np.zeros((num_dets, num_candidates))

            dets_to_match = dets_high[unmatched_dets_high_indices]
            embs_to_match = dets_embs[unmatched_dets_high_indices]

            for j, id_data in enumerate(candidate_ids):
                # case 1: 候选者是 Active 轨迹
                if id_data['status'] == 'Active':
                    pred_box = id_data['tracker'].history[-1][0].reshape(1, -1)
                    trk_emb = id_data['tracker'].smooth_feature.reshape(1, -1)

                    iou_cost = iou_distance(dets_to_match[:, 0:5], pred_box)
                    reid_cost = reid_distance(embs_to_match, trk_emb)
                    fused_cost = self.cost_fusion_alpha * iou_cost + (1 - self.cost_fusion_alpha) * reid_cost
                    cost_matrix[:, j] = fused_cost.flatten()

                # case 2: 候选者是 Lost 轨迹
                elif id_data['status'] == 'Lost':
                    trk_emb = id_data['last_feature'].reshape(1, -1)

                    reid_cost = reid_distance(embs_to_match, trk_emb)
                    location_cost = self._location_cost_for_pool([id_data], dets_to_match)
                    fused_cost = self.reentry_beta * location_cost + (1 - self.reentry_beta) * reid_cost

                    fused_cost[reid_cost > self.reentry_reid_gate_thresh] = 1.0
                    cost_matrix[:, j] = fused_cost.flatten()

            # c. 执行全局线性分配
            # --------------------------
            matched_indices, u_det_indices, u_candidate_indices = linear_assignment(cost_matrix,
                                                                                    thresh=self.reid_asso_thresh)

            # d. 更新匹配结果
            # ------------------
            for m in matched_indices:
                det_local_idx = m[0]
                candidate_idx = m[1]

                det_global_idx = unmatched_dets_high_indices[det_local_idx]
                id_data = candidate_ids[candidate_idx]

                # case 1: 匹配上了 Active 轨迹
                if id_data['status'] == 'Active':
                    id_data['tracker'].update(
                        dets_high[det_global_idx, :-2],
                        dets_high[det_global_idx, -2],
                        dets_high[det_global_idx, -1],
                        dets_embs[det_global_idx]
                    )

                # case 2: 匹配上了 Lost 轨迹 (复活)
                elif id_data['status'] == 'Lost':
                    bbox = dets_high[det_global_idx, :5]
                    cls = dets_high[det_global_idx, 5]
                    det_ind = dets_high[det_global_idx, 6]
                    feature = dets_embs[det_global_idx]

                    new_tracker = KalmanBoxTracker(
                        bbox, cls, det_ind,
                        feature=feature,
                        delta_t=self.delta_t,
                        Q_xy_scaling=self.Q_xy_scaling,
                        Q_s_scaling=self.Q_s_scaling,
                        max_obs=self.max_age + 5
                    )

                    new_tracker.id = id_data['id']
                    new_tracker.hit_streak = id_data['hit_streak']
                    if id_data['home_model']['n'] > 0:
                        new_tracker.home_mean = id_data['home_model']['mean']
                        new_tracker.home_covariance = id_data['home_model']['covariance']
                        new_tracker.home_n = id_data['home_model']['n']

                    id_data['status'] = 'Active'
                    id_data['tracker'] = new_tracker
                    id_data['age_lost'] = 0

            # 更新未匹配的检测列表，用于创建新人
            unmatched_dets_high_indices = unmatched_dets_high_indices[u_det_indices]
        # e. BYTE 匹配 (逻辑可以保持，但需要对未匹配的Active轨迹进行)
        # ... 这里的逻辑需要微调，以适应 u_candidate_indices ...
        # 为简化，我们可以暂时跳过BYTE，先验证核心逻辑

        # --- END OF FINAL FIX - UNIFIED GLOBAL MATCHING ---

        # --- START OF ID POOL MODIFICATION 4 ---
        # --- 4. 状态更新、新人处理与结果输出 ---

        # a. 更新ID池中所有ID的状态 (修复版)
        # ------------------------------------
        active_ids_after_match = [p for p in self.id_pool if p['status'] == 'Active']

        for id_data in active_ids_after_match:
            tracker = id_data['tracker']

            # 卡尔曼滤波器内部的 time_since_update 已经自动增加了
            # 检查是否超过了 max_age
            if tracker.time_since_update > self.max_age:
                # 状态切换为 Lost
                id_data['status'] = 'Lost'
                id_data['age_lost'] = 0  # 开始从0计数
                # 保存最后的状态信息
                id_data['last_feature'] = tracker.smooth_feature
                home_model = id_data['home_model']
                home_model['mean'] = tracker.home_mean
                home_model['covariance'] = tracker.home_covariance
                id_data['hit_streak'] = tracker.hit_streak  # 保存功绩
                home_model['n'] = tracker.home_n
                # 销毁短期的tracker实例
                id_data['tracker'] = None

        # 处理 Lost 状态的ID (这部分逻辑是正确的，保持不变)
        lost_ids_after_match = [p for p in self.id_pool if p['status'] == 'Lost']
        for id_data in lost_ids_after_match:
            id_data['age_lost'] += 1
            if id_data['age_lost'] > self.id_recycle_age:
                id_data['status'] = 'Available'
                id_data['last_feature'] = None

        # b. 为真正的新人分配ID
        # ----------------------------
        # unmatched_dets_high_indices 现在包含了所有未被关联的高置信度检测
        for det_idx in unmatched_dets_high_indices:
            # 寻找一个可用的ID
            available_id_slot = None
            for id_data in self.id_pool:
                if id_data['status'] == 'Available':
                    available_id_slot = id_data
                    break

            # 如果找到了可用的ID
            if available_id_slot is not None:
                new_tracker = KalmanBoxTracker(
                    dets_high[det_idx, :5], dets_high[det_idx, 5], dets_high[det_idx, 6],
                    feature=dets_embs[det_idx],
                    delta_t=self.delta_t, Q_xy_scaling=self.Q_xy_scaling, Q_s_scaling=self.Q_s_scaling
                )
                new_tracker.id = available_id_slot['id']

                # 更新ID池
                available_id_slot['status'] = 'Active'
                available_id_slot['tracker'] = new_tracker
                # --- START OF FIX ---
                available_id_slot['hit_streak'] = new_tracker.hit_streak  # 初始化为0
                # --- END OF FIX ---
                available_id_slot['age_lost'] = 0
                available_id_slot['last_feature'] = new_tracker.smooth_feature
            # else:
            # 如果没有可用的ID（教室满员），则忽略这个新检测。
            # print(f"Warning: ID Pool is full. Cannot assign new ID for detection.")

        # c. 整理并输出当前帧的跟踪结果 (优化版)
        # --------------------------------------
        for id_data in self.id_pool:
            if id_data['status'] == 'Active':
                tracker = id_data['tracker']
                if tracker.time_since_update < 1 and (
                        tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                    # 使用 get_state() 获取卡尔曼滤波平滑后的位置
                    d = tracker.get_state()[0]
                    ret.append(np.concatenate((
                        d,
                        [tracker.id],
                        [tracker.conf],
                        [tracker.cls],
                        [tracker.det_ind]
                    )).reshape(1, -1))

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.array([])
        # --- END OF ID POOL MODIFICATION 4 ---

