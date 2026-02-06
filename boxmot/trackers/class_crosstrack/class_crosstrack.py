# boxmot/boxmot/trackers/Class_crosstrack/class_crosstrack.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np

from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils.matching import iou_distance, linear_assignment
from boxmot.utils.ops import xyxy2xysr, xyxy2xywh


# -----------------------
# helpers
# -----------------------
def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x is None:
        return x
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def _cosine_distance(dets_embs: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """Cosine distance = 1 - dot(normalized). dets[M,D], protos[N,D] -> [M,N]."""
    if dets_embs.size == 0 or protos.size == 0:
        return np.empty((len(dets_embs), len(protos)), dtype=np.float32)
    sim = np.dot(dets_embs, protos.T)
    return (1.0 - sim).astype(np.float32)


def _convert_x_to_bbox(x: np.ndarray, score: Optional[float] = None) -> np.ndarray:
    # safety for sqrt(negative)
    s_times_r = float(x[2] * x[3])
    if s_times_r < 0:
        s_times_r = 0.0
    w = np.sqrt(s_times_r)
    h = float(x[2] / w) if w > 0 else 0.0

    if score is None:
        out = np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.], dtype=np.float32)
        return out.reshape((1, 4))
    out = np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score], dtype=np.float32)
    return out.reshape((1, 5))


class _KalmanBox:
    """Minimal KF wrapper (XYSR) for per-slot short-term motion."""
    def __init__(
        self,
        delta_t: int = 3,
        max_obs: int = 50,
        Q_xy_scaling: float = 0.01,
        Q_s_scaling: float = 0.0001,
    ):
        self.kf = KalmanFilterXYSR(dim_x=7, dim_z=4, max_obs=max_obs)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1]], dtype=np.float32
        )
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0]], dtype=np.float32
        )

        self.kf.R[2:, 2:] *= 3.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[4:6, 4:6] *= Q_xy_scaling
        self.kf.Q[-1, -1] *= Q_s_scaling

        self.delta_t = int(delta_t)

        self.last_observation = np.array([-1, -1, -1, -1, -1], dtype=np.float32)  # xyxyc
        self.conf = 0.0
        self.cls = -1
        self.det_ind = -1

        self.home_mean = None
        self.home_covariance = None
        self.home_M2 = np.zeros((2, 2), dtype=np.float32)
        self.home_n = 0

    def reset_with_det(self, bbox_xyxyc: np.ndarray, cls: int, det_ind: int):
        self.kf.x[:] = 0
        self.kf.x[:4] = xyxy2xysr(bbox_xyxyc)

        self.last_observation = bbox_xyxyc.astype(np.float32)
        self.conf = float(bbox_xyxyc[-1])
        self.cls = int(cls)
        self.det_ind = int(det_ind)

        # reset home model for this lesson
        self.home_mean = None
        self.home_covariance = None
        self.home_M2[:] = 0
        self.home_n = 0

        cx = (bbox_xyxyc[0] + bbox_xyxyc[2]) / 2
        cy = (bbox_xyxyc[1] + bbox_xyxyc[3]) / 2
        self._update_home(np.array([cx, cy], dtype=np.float32))

    def _update_home(self, center_xy: np.ndarray):
        self.home_n += 1
        if self.home_n == 1:
            self.home_mean = center_xy.astype(np.float32)
            self.home_covariance = np.eye(2, dtype=np.float32) * 100
            return
        delta = center_xy - self.home_mean
        self.home_mean = self.home_mean + delta / self.home_n
        delta2 = center_xy - self.home_mean
        self.home_M2 += np.outer(delta, delta2).astype(np.float32)
        if self.home_n > 1:
            self.home_covariance = self.home_M2 / (self.home_n - 1) + np.eye(2, dtype=np.float32) * 1e-6

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        return _convert_x_to_bbox(self.kf.x)

    def update(self, bbox_xyxyc: Optional[np.ndarray], cls: Optional[int], det_ind: Optional[int]):
        if bbox_xyxyc is not None:
            self.last_observation = bbox_xyxyc.astype(np.float32)
            self.conf = float(bbox_xyxyc[-1])
            if cls is not None:
                self.cls = int(cls)
            if det_ind is not None:
                self.det_ind = int(det_ind)
            self.kf.update(xyxy2xysr(bbox_xyxyc))

            cx = (bbox_xyxyc[0] + bbox_xyxyc[2]) / 2
            cy = (bbox_xyxyc[1] + bbox_xyxyc[3]) / 2
            self._update_home(np.array([cx, cy], dtype=np.float32))
        else:
            self.kf.update(bbox_xyxyc)

    def get_state(self) -> np.ndarray:
        return _convert_x_to_bbox(self.kf.x)


@dataclass
class _Slot:
    global_id: int
    face_proto: Optional[np.ndarray] = None  # [D] L2 normalized
    kf: Optional[_KalmanBox] = None

    state: str = "ABSENT"          # VISIBLE / LOST / ABSENT
    initialized: bool = False      # whether this lesson has been activated
    miss_count: int = 0


class ClassCrossTrack(BaseTracker):
    """
    Cross-lesson fixed-N tracker (tracking layer only):
      - N slots, never create/delete.
      - Cross-lesson identity anchored by face_proto gallery.
      - Within-lesson motion assisted by robust 2D Mahalanobis cost.
      - start_lesson() resets motion/state, keeps face_proto.
    """

    def __init__(
        self,
        num_slots: int = 40,
        gallery_path: Union[str, Path, None] = None,

        # detection thresholds
        min_conf: float = 0.1,
        det_thresh: float = 0.2,

        # association
        face_gate_thresh: float = 0.35,   # cosine distance gate
        asso_thresh: float = 0.6,         # fused cost thresh for Hungarian
        cost_fusion_alpha: float = 0.2,   # motion weight; face weight = 1-alpha
        cold_start_frames: int = 10,      # first K frames use face-only (alpha=0)

        # state machine
        lost_to_absent: int = 60,

        # gallery update
        face_ema: float = 0.9,
        require_embs: bool = True,
        open_set_init: bool = False,      # allow filling empty protos by arbitrary assignment (lesson1 quick boot)

        # optional rescue/byte (never creates tracks)
        use_byte: bool = True,
        byte_iou_thresh: float = 0.5,
        rescue_cost_thresh: float = 3.5,

        # KF params
        delta_t: int = 3,
        max_obs: int = 50,
        Q_xy_scaling: float = 0.01,
        Q_s_scaling: float = 0.0001,

        # BaseTracker args
        max_age: int = 999999,
        per_class: bool = False,
        **kwargs,
    ):
        # ⚠️ per_class 会触发 BaseTracker.per_class_decorator 分类调用，embs 对齐会被破坏
        if per_class:
            raise ValueError("ClassCrossTrack 暂不支持 per_class=True（会导致 embs 与 dets 对齐问题）。请置 False。")

        super().__init__(max_age=max_age, per_class=per_class)

        self.num_slots = int(num_slots)
        self.min_conf = float(min_conf)
        self.det_thresh = float(det_thresh)

        self.face_gate_thresh = float(face_gate_thresh)
        self.asso_thresh = float(asso_thresh)
        self.cost_fusion_alpha = float(cost_fusion_alpha)
        self.cold_start_frames = int(cold_start_frames)

        self.lost_to_absent = int(lost_to_absent)
        self.face_ema = float(face_ema)
        self.require_embs = bool(require_embs)
        self.open_set_init = bool(open_set_init)

        self.use_byte = bool(use_byte)
        self.byte_iou_thresh = float(byte_iou_thresh)
        self.rescue_cost_thresh = float(rescue_cost_thresh)

        self.delta_t = int(delta_t)
        self.max_obs = int(max_obs)
        self.Q_xy_scaling = float(Q_xy_scaling)
        self.Q_s_scaling = float(Q_s_scaling)

        self.gallery_path = Path(gallery_path) if gallery_path else None

        self.frame_count = 0
        self.frame_in_lesson = 0

        # fixed slots
        self.slots: List[_Slot] = []
        for i in range(self.num_slots):
            self.slots.append(
                _Slot(
                    global_id=i + 1,
                    face_proto=None,
                    kf=_KalmanBox(
                        delta_t=self.delta_t,
                        max_obs=self.max_obs,
                        Q_xy_scaling=self.Q_xy_scaling,
                        Q_s_scaling=self.Q_s_scaling,
                    ),
                    state="ABSENT",
                    initialized=False,
                    miss_count=0,
                )
            )

        # load gallery if exists
        if self.gallery_path is not None and self.gallery_path.exists():
            self.load_gallery(self.gallery_path)

    # -----------------------
    # lesson control
    # -----------------------
    def start_lesson(self, lesson_id=None):
        """Call at the start of each lesson/clip. Keep face_proto; reset motion/state."""
        self.frame_in_lesson = 0
        for s in self.slots:
            s.state = "ABSENT"
            s.initialized = False
            s.miss_count = 0
            s.kf = _KalmanBox(
                delta_t=self.delta_t,
                max_obs=self.max_obs,
                Q_xy_scaling=self.Q_xy_scaling,
                Q_s_scaling=self.Q_s_scaling,
            )

    # -----------------------
    # gallery IO
    # -----------------------
    def load_gallery(self, path: Union[str, Path]):
        p = Path(path)
        data = np.load(str(p), allow_pickle=True)
        protos = data["face_protos"]  # [N, D]
        if protos.shape[0] != self.num_slots:
            raise ValueError(f"Gallery num_slots mismatch: {protos.shape[0]} vs {self.num_slots}")
        protos = _l2norm(protos.astype(np.float32))
        for i in range(self.num_slots):
            self.slots[i].face_proto = protos[i]

    def save_gallery(self, path: Union[str, Path, None] = None):
        p = Path(path) if path else self.gallery_path
        if p is None:
            raise ValueError("No gallery_path provided.")
        # infer D
        valid = [s.face_proto for s in self.slots if s.face_proto is not None and s.face_proto.size > 1]
        if len(valid) == 0:
            raise ValueError("No valid face_proto to save.")
        D = valid[0].shape[0]
        arr = np.zeros((self.num_slots, D), dtype=np.float32)
        for i, s in enumerate(self.slots):
            if s.face_proto is not None and s.face_proto.shape[0] == D:
                arr[i] = s.face_proto.astype(np.float32)
        np.savez(str(p), face_protos=_l2norm(arr))

    # -----------------------
    # costs
    # -----------------------
    def _mahalanobis_cost_2d(self, tracks: List[_KalmanBox], dets_xyxyc_cls_ind: np.ndarray) -> np.ndarray:
        """[M,T] cost in [0,1] with gating."""
        M = len(dets_xyxyc_cls_ind)
        T = len(tracks)
        if M == 0 or T == 0:
            return np.empty((M, T), dtype=np.float32)

        det_centers = xyxy2xywh(dets_xyxyc_cls_ind[:, :4])[:, :2]  # [M,2]
        cost = np.ones((M, T), dtype=np.float32)

        for j, trk in enumerate(tracks):
            mean, cov = trk.kf.project()
            mean_xy = mean[:2]
            cov_xy = cov[:2, :2]
            try:
                cov_inv = np.linalg.inv(cov_xy)
                delta = det_centers - mean_xy.T
                dist_sq = np.sum((delta @ cov_inv) * delta, axis=1)
                c = (dist_sq / 5.991).astype(np.float32)  # 95% Chi-square dof=2
                c[c > 1.0] = 1.0
                cost[:, j] = c
            except np.linalg.LinAlgError:
                cost[:, j] = 1.0

        return cost

    def _location_cost(self, tracks: List[_KalmanBox], dets_xyxyc_cls_ind: np.ndarray) -> np.ndarray:
        """Home-model rescue cost. [M,T]."""
        M = len(dets_xyxyc_cls_ind)
        T = len(tracks)
        out = np.ones((M, T), dtype=np.float32)
        if M == 0 or T == 0:
            return out

        det_centers = np.array(
            [((d[0] + d[2]) / 2, (d[1] + d[3]) / 2) for d in dets_xyxyc_cls_ind],
            dtype=np.float32
        )
        for j, trk in enumerate(tracks):
            if trk.home_n <= 1 or trk.home_covariance is None:
                continue
            try:
                cov_inv = np.linalg.inv(trk.home_covariance)
                delta = det_centers - trk.home_mean
                sq = np.sum((delta @ cov_inv) * delta, axis=1)
                sq[sq < 0] = 0
                maha = np.sqrt(sq)
                c = (maha / 3.0).astype(np.float32)
                out[:, j] = np.minimum(c, 1.0)
            except np.linalg.LinAlgError:
                continue
        return out

    def _proto_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (proto_mat[N,D], valid_mask[N])."""
        protos = [s.face_proto for s in self.slots]
        valid = np.array([p is not None and p.ndim == 1 and p.size > 1 for p in protos], dtype=bool)

        if not valid.any():
            return np.zeros((self.num_slots, 1), dtype=np.float32), valid

        D = protos[int(np.where(valid)[0][0])].shape[0]
        mat = np.zeros((self.num_slots, D), dtype=np.float32)
        for i, p in enumerate(protos):
            if p is not None and p.shape[0] == D:
                mat[i] = p.astype(np.float32)
        return _l2norm(mat), valid

    # -----------------------
    # main update
    # -----------------------
    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        dets: [x1,y1,x2,y2,conf,cls]
        embs: face embeddings aligned with dets rows (num_dets, D), ideally L2-normalized.
        returns: [x1,y1,x2,y2,global_id,conf,cls,det_ind]
        """
        self.check_inputs(dets, img)
        self.frame_count += 1
        self.frame_in_lesson += 1

        if (embs is None) and self.require_embs:
            raise ValueError("ClassCrossTrack 需要 embs (face embeddings)；请在感知层生成并传入 update(..., embs=...).")

        # append det_ind
        dets = np.asarray(dets)
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])  # det_ind at last col

        confs = dets[:, 4 + self.is_obb]
        remain_inds = confs > self.det_thresh
        inds_low = confs > self.min_conf
        inds_high = confs < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_high = dets[remain_inds]     # [M, 7]
        dets_second = dets[inds_second]   # [M2, 7]

        if len(dets_high) == 0:
            # no high conf dets: mark miss
            for s in self.slots:
                s.kf.predict()
                s.miss_count += 1
                s.state = "ABSENT" if s.miss_count > self.lost_to_absent else "LOST"
            return np.array([])

        # predict all
        for s in self.slots:
            s.kf.predict()

        # embeddings for high dets
        if embs is None:
            dets_embs = None
        else:
            dets_embs = _l2norm(np.asarray(embs)[remain_inds].astype(np.float32))

        M = len(dets_high)
        N = self.num_slots

        # face cost
        proto_mat, proto_valid = self._proto_matrix()  # [N,D]
        if proto_mat.shape[1] == 1:
            # no gallery at all
            if not self.open_set_init:
                # can still do motion-only for initialized slots, but cold-start will be weak
                proto_valid[:] = False

        if dets_embs is None:
            face_cost = np.ones((M, N), dtype=np.float32)
        else:
            if proto_mat.shape[1] != dets_embs.shape[1] and proto_mat.shape[1] != 1:
                raise ValueError(f"Embedding dim mismatch: dets_embs D={dets_embs.shape[1]} vs proto D={proto_mat.shape[1]}")
            if proto_mat.shape[1] == 1:
                face_cost = np.ones((M, N), dtype=np.float32)
            else:
                face_cost = _cosine_distance(dets_embs, proto_mat)  # [M,N]
                face_cost[:, ~proto_valid] = 1.0

        # open-set init: allow empty protos to be filled (quick bootstrap)
        if self.open_set_init:
            empty_mask = np.array([s.face_proto is None for s in self.slots], dtype=bool)
            if empty_mask.any():
                # let Hungarian assign something to empty slots by giving them low face cost
                face_cost[:, empty_mask] = 0.0

        # motion cost only for initialized
        motion_cost = np.ones((M, N), dtype=np.float32)
        init_mask = np.array([s.initialized for s in self.slots], dtype=bool)
        if init_mask.any():
            tracks = [self.slots[i].kf for i in np.where(init_mask)[0]]
            mc = self._mahalanobis_cost_2d(tracks, dets_high)  # [M, sum(init)]
            motion_cost[:, init_mask] = mc

        # fusion
        alpha = 0.0 if self.frame_in_lesson <= self.cold_start_frames else self.cost_fusion_alpha
        fused = alpha * motion_cost + (1.0 - alpha) * face_cost

        # face gating
        fused[face_cost > self.face_gate_thresh] = 1.0

        # hungarian (returns matched det-slot pairs)
        matched, u_det, u_slot = linear_assignment(fused, thresh=self.asso_thresh)

        # apply matches
        for det_i, slot_i in matched:
            s = self.slots[int(slot_i)]
            det_row = dets_high[int(det_i)]

            bbox_xyxyc = det_row[:5].astype(np.float32)
            cls = int(det_row[5])
            det_ind = int(det_row[6])

            if not s.initialized:
                s.kf.reset_with_det(bbox_xyxyc, cls, det_ind)
                s.initialized = True
            else:
                s.kf.update(bbox_xyxyc, cls, det_ind)

            # update face proto if available
            if dets_embs is not None:
                f = dets_embs[int(det_i)]
                if s.face_proto is None:
                    s.face_proto = f.copy()
                else:
                    s.face_proto = _l2norm(self.face_ema * s.face_proto + (1.0 - self.face_ema) * f)

            s.miss_count = 0
            s.state = "VISIBLE"

        # rescue by home model (only initialized unmatched slots)
        if len(u_slot) > 0 and len(u_det) > 0:
            u_slot = np.asarray(u_slot, dtype=int)
            u_det = np.asarray(u_det, dtype=int)
            cand_slots = [i for i in u_slot.tolist() if self.slots[i].initialized]
            if len(cand_slots) > 0:
                tracks = [self.slots[i].kf for i in cand_slots]
                rescue_dets = dets_high[u_det]
                rc = self._location_cost(tracks, rescue_dets)
                rescued, _, _ = linear_assignment(rc, thresh=self.rescue_cost_thresh)
                used_slots = set()
                used_dets = set()
                for dloc, tloc in rescued:
                    det_global = int(u_det[int(dloc)])
                    slot_global = int(cand_slots[int(tloc)])
                    det_row = dets_high[det_global]
                    bbox_xyxyc = det_row[:5].astype(np.float32)
                    cls = int(det_row[5])
                    det_ind = int(det_row[6])
                    self.slots[slot_global].kf.update(bbox_xyxyc, cls, det_ind)
                    self.slots[slot_global].miss_count = 0
                    self.slots[slot_global].state = "VISIBLE"
                    used_slots.add(slot_global)
                    used_dets.add(det_global)

                u_slot = np.array([x for x in u_slot if int(x) not in used_slots], dtype=int)
                u_det = np.array([x for x in u_det if int(x) not in used_dets], dtype=int)

        # BYTE on second-stage (never create tracks)
        if self.use_byte and len(dets_second) > 0 and len(u_slot) > 0:
            cand_slots = [i for i in u_slot.tolist() if self.slots[i].initialized]
            if len(cand_slots) > 0:
                trk_boxes = []
                for i in cand_slots:
                    box = self.slots[i].kf.get_state()[0]  # [x1,y1,x2,y2]
                    trk_boxes.append(np.concatenate([box, [0.0]], axis=0))
                trk_boxes = np.stack(trk_boxes, axis=0)
                iou_cost = iou_distance(dets_second[:, 0:5 + self.is_obb], trk_boxes)
                byte_matched, _, _ = linear_assignment(iou_cost, thresh=self.byte_iou_thresh)
                used_slots = set()
                for dloc, tloc in byte_matched:
                    slot_global = int(cand_slots[int(tloc)])
                    det_row = dets_second[int(dloc)]
                    bbox_xyxyc = det_row[:5].astype(np.float32)
                    cls = int(det_row[5])
                    det_ind = int(det_row[6])
                    self.slots[slot_global].kf.update(bbox_xyxyc, cls, det_ind)
                    self.slots[slot_global].miss_count = 0
                    self.slots[slot_global].state = "VISIBLE"
                    used_slots.add(slot_global)
                u_slot = np.array([x for x in u_slot if int(x) not in used_slots], dtype=int)

        # unmatched slots -> miss
        for slot_i in u_slot:
            s = self.slots[int(slot_i)]
            s.kf.update(None, None, None)
            s.miss_count += 1
            s.state = "ABSENT" if s.miss_count > self.lost_to_absent else "LOST"

        # return visible
        ret = []
        for s in self.slots:
            if s.state != "VISIBLE":
                continue

            if s.kf.last_observation.sum() >= 0:
                d = s.kf.last_observation[:4]
                conf = s.kf.conf
                cls = s.kf.cls
                det_ind = s.kf.det_ind
            else:
                d = s.kf.get_state()[0]
                conf = s.kf.conf
                cls = s.kf.cls
                det_ind = s.kf.det_ind

            ret.append(
                np.array([d[0], d[1], d[2], d[3], s.global_id, conf, cls, det_ind], dtype=np.float32).reshape(1, -1)
            )

        if len(ret) > 0:
            return np.concatenate(ret, axis=0)
        return np.array([])
