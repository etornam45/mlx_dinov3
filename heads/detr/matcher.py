import mlx.core as mx
import mlx.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import List, Tuple, Dict


def box_cxcywh_to_xyxy(boxes: mx.array) -> mx.array:
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return mx.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)


def box_area(boxes: mx.array) -> mx.array:
    """Area of boxes in [x1, y1, x2, y2] format."""
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def generalized_iou(boxes_a: mx.array, boxes_b: mx.array) -> mx.array:
    """
    Pairwise Generalised IoU between two sets of boxes.

    Args:
        boxes_a: (N, 4) in xyxy format
        boxes_b: (M, 4) in xyxy format
    Returns:
        giou: (N, M) matrix
    """
    # Intersection
    inter_x1 = mx.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = mx.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = mx.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = mx.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = mx.maximum(inter_x2 - inter_x1, mx.zeros_like(inter_x2))
    inter_h = mx.maximum(inter_y2 - inter_y1, mx.zeros_like(inter_y2))
    inter_area = inter_w * inter_h

    area_a = box_area(boxes_a)[:, None]  # (N, 1)
    area_b = box_area(boxes_b)[None, :]  # (1, M)
    union_area = area_a + area_b - inter_area

    iou = inter_area / mx.maximum(union_area, mx.array(1e-6))

    # Enclosing box
    enc_x1 = mx.minimum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    enc_y1 = mx.minimum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    enc_x2 = mx.maximum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    enc_y2 = mx.maximum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    giou = iou - (enc_area - union_area) / mx.maximum(enc_area, mx.array(1e-6))
    return giou


def build_cost_matrix(
    pred_logits: mx.array,  # (num_queries, num_classes)
    pred_boxes: mx.array,  # (num_queries, 4)  cx cy w h, normalised
    gt_labels: mx.array,  # (num_gt,)
    gt_boxes: mx.array,  # (num_gt, 4)  cx cy w h, normalised
    cost_class: float = 1.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
) -> np.ndarray:
    """
    Build the (num_queries × num_gt) cost matrix used by the Hungarian solver.

    Classification cost   – negative softmax probability of the GT class.
    L1 box cost           – mean absolute error between normalised boxes.
    GIoU cost             – negative GIoU (higher overlap = lower cost).
    """

    # --- classification cost -------------------------------------------
    probs = mx.softmax(pred_logits, axis=-1)  # (Q, C)
    # Select the GT-class probability for every (query, gt) pair
    gt_idx = np.array(gt_labels.tolist(), dtype=np.int32)
    cost_cls = -np.array(probs.tolist())[:, gt_idx]  # (Q, num_gt)

    # --- L1 box cost ---------------------------------------------------
    pb = np.array(pred_boxes.tolist())  # (Q, 4)
    gb = np.array(gt_boxes.tolist())  # (num_gt, 4)
    cost_l1 = np.sum(np.abs(pb[:, None, :] - gb[None, :, :]), axis=-1)  # (Q, G)

    # --- GIoU cost -----------------------------------------------------
    pb_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    gb_xyxy = box_cxcywh_to_xyxy(gt_boxes)
    cost_giou_mat = -np.array(generalized_iou(pb_xyxy, gb_xyxy).tolist())  # (Q, G)

    # --- combined cost -------------------------------------------------
    C = cost_class * cost_cls + cost_bbox * cost_l1 + cost_giou * cost_giou_mat
    return C.astype(np.float32)


def hungarian_match(
    pred_logits: mx.array,
    pred_boxes: mx.array,
    gt_labels: mx.array,
    gt_boxes: mx.array,
    cost_class: float = 1.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the optimal query → gt assignment for a single image.

    Returns:
        query_indices  (K,)  – which queries were matched
        gt_indices     (K,)  – the GT they were matched to
    """
    if gt_labels.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    C = build_cost_matrix(
        pred_logits,
        pred_boxes,
        gt_labels,
        gt_boxes,
        cost_class,
        cost_bbox,
        cost_giou,
    )
    query_idx, gt_idx = linear_sum_assignment(C)
    return query_idx.astype(np.int64), gt_idx.astype(np.int64)


def one_hot(labels: mx.array, num_classes: int) -> mx.array:
    # Create an identity matrix and index into it using your labels
    return mx.eye(num_classes)[labels]


@dataclass
class LossStats:
    total: float
    cls: float
    bbox_l1: float
    bbox_giou: float
    num_matched: int


class HungarianLoss(nn.Module):
    """
    End-to-end set-prediction loss using optimal bipartite matching.

    Args:
        num_classes:   number of foreground classes (background is implicit)
        cost_class:    weight of the classification term in the matching cost
        cost_bbox:     weight of the L1 box term in the matching cost
        cost_giou:     weight of the GIoU term in the matching cost
        loss_class:    weight of the classification term in the training loss
        loss_bbox:     weight of the L1 box term in the training loss
        loss_giou:     weight of the GIoU term in the training loss
        no_obj_coef:   down-weighting for the background class in cross-entropy
    """

    def __init__(
        self,
        num_classes: int,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        loss_class: float = 1.0,
        loss_bbox: float = 5.0,
        loss_giou: float = 2.0,
        no_obj_coef: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.loss_class = loss_class
        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou
        self.no_obj_coef = no_obj_coef

    def _cls_loss(
        self,
        pred_logits: mx.array,  # (Q, C+1)
        gt_labels: mx.array,  # (num_gt,)
        query_idx: np.ndarray,  # matched query positions
        gt_idx: np.ndarray,  # matched GT positions
    ) -> mx.array:
        """
        Cross-entropy loss over all queries.
        Unmatched queries are assigned the background class (index num_classes).
        """
        num_queries = pred_logits.shape[0]
        # Default target: background class
        targets = np.full((num_queries,), self.num_classes, dtype=np.int32)
        if len(query_idx) > 0:
            targets[query_idx] = np.array(gt_labels.tolist(), dtype=np.int32)[gt_idx]

        targets_mx = mx.array(targets)

        # Class weights: background gets no_obj_coef, foreground gets 1.0
        weights = np.ones(self.num_classes + 1, dtype=np.float32)
        weights[self.num_classes] = self.no_obj_coef
        w = mx.array(weights)

        log_probs = mx.log(mx.softmax(pred_logits, axis=-1) + 1e-8)  # (Q, C+1)
        # Gather the log-prob of the target class for each query
        _one_hot = one_hot(targets_mx, self.num_classes + 1)  # (Q, C+1)
        nll = -mx.sum(log_probs * _one_hot, axis=-1)  # (Q,)
        sample_weights = w[targets_mx]  # (Q,)
        return mx.mean(nll * sample_weights)

    def _box_losses(
        self,
        pred_boxes: mx.array,  # (Q, 4)
        gt_boxes: mx.array,  # (num_gt, 4)
        query_idx: np.ndarray,
        gt_idx: np.ndarray,
    ) -> Tuple[mx.array, mx.array]:
        """L1 and GIoU losses computed only on matched pairs."""
        if len(query_idx) == 0:
            zero = mx.array(0.0)
            return zero, zero

        matched_pred = pred_boxes[mx.array(query_idx)]
        matched_gt = gt_boxes[mx.array(gt_idx)]

        l1 = mx.mean(mx.abs(matched_pred - matched_gt))

        pb_xyxy = box_cxcywh_to_xyxy(matched_pred)
        gb_xyxy = box_cxcywh_to_xyxy(matched_gt)
        # Diagonal of the pairwise GIoU matrix = per-pair GIoU
        giou_diag = mx.diagonal(generalized_iou(pb_xyxy, gb_xyxy))
        giou = mx.mean(1.0 - giou_diag)

        return l1, giou

    def _single_pass(
        self,
        predictions: Dict[str, mx.array],
        targets,
    ) -> Tuple[mx.array, List[LossStats]]:
        """Compute loss for ONE set of predictions (final or aux)."""
        pred_logits = predictions["logits"]   # (B, Q, C+1)
        pred_boxes  = predictions["boxes"]    # (B, Q, 4)
        batch_size  = pred_logits.shape[0]

        total_cls  = mx.array(0.0)
        total_l1   = mx.array(0.0)
        total_giou = mx.array(0.0)
        stats: List[LossStats] = []

        for i in range(batch_size):
            if isinstance(targets, dict):
                labels_np = np.array(targets["labels"][i])
                valid_idx = [j for j, v in enumerate(labels_np.tolist()) if v > 0]
                if valid_idx:
                    idx_mx   = mx.array(valid_idx)
                    gt_labels = mx.array(targets["labels"][i])[idx_mx]
                    gt_boxes  = mx.array(targets["boxes"][i])[idx_mx]
                else:
                    gt_labels = mx.array([], dtype=mx.int32)
                    gt_boxes  = mx.array([], dtype=mx.float32).reshape(0, 4)
            else:
                gt_labels = mx.array(targets[i]["labels"])
                gt_boxes  = mx.array(targets[i]["boxes"])

            q_idx, g_idx = hungarian_match(
                pred_logits[i], pred_boxes[i],
                gt_labels, gt_boxes,
                self.cost_class, self.cost_bbox, self.cost_giou,
            )

            cls_l       = self._cls_loss(pred_logits[i], gt_labels, q_idx, g_idx)
            l1_l, giou_l = self._box_losses(pred_boxes[i], gt_boxes, q_idx, g_idx)

            total_cls  = total_cls  + cls_l
            total_l1   = total_l1   + l1_l
            total_giou = total_giou + giou_l

            stats.append(LossStats(
                total=0.0,
                cls=float(cls_l.tolist()),
                bbox_l1=float(l1_l.tolist()),
                bbox_giou=float(giou_l.tolist()),
                num_matched=len(q_idx),
            ))

        loss = (
            self.loss_class * total_cls / batch_size
            + self.loss_bbox * total_l1  / batch_size
            + self.loss_giou * total_giou / batch_size
        )
        return loss, stats

    def __call__(
        self,
        predictions: Dict[str, mx.array],
        targets,
        aux_weight: float = 0.5,      # ← weight for intermediate layers
    ) -> Tuple[mx.array, List[LossStats]]:

        # Final layer loss
        final_loss, stats = self._single_pass(predictions, targets)

        # Aux losses — same matching, down-weighted
        aux_loss = mx.array(0.0)
        for aux_pred in predictions.get("aux", []):
            layer_loss, _ = self._single_pass(aux_pred, targets)
            aux_loss = aux_loss + layer_loss

        total_loss = final_loss + aux_weight * aux_loss

        for s in stats:
            s.total = float(total_loss.tolist())

        return total_loss, stats


if __name__ == "__main__":
    # Example usage
    loss_fn = HungarianLoss(num_classes=80)
    predictions = {
        "logits": mx.random.normal((2, 300, 81)),
        "boxes": mx.random.uniform(shape=(2, 300, 4)),
    }
    targets = [
        {
            "labels": mx.array([1, 2, 3]),
            "boxes": mx.random.uniform(shape=(3, 4)),
        },
        {
            "labels": mx.array([4, 5]),
            "boxes": mx.random.uniform(shape=(2, 4)),
        },
    ]
    loss, stats = loss_fn(predictions, targets)
    print(loss)
    print(stats)
