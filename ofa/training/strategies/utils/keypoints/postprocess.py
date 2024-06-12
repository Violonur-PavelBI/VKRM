from typing import Tuple

import torch


def calculate_argmax_keypoints_and_visibility(
    heatmaps: torch.Tensor, threshold: float = 0.1, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, n_keypoints, height, width = heatmaps.size()

    device = device if torch.cuda.is_available() else "cpu"
    heatmaps = heatmaps.detach().to(device)

    # [batch_size, n_keypoints, height * width]
    heatmaps_reshaped = heatmaps.reshape(batch_size, n_keypoints, -1)

    # [batch_size, n_keypoints]
    max_values, max_positions = torch.max(heatmaps_reshaped, dim=-1)

    y_positions = max_positions // width
    x_positions = max_positions % width

    # [batch_size, n_keypoints, 2]
    keypoints = torch.stack([x_positions, y_positions], dim=-1)

    # NOTE: coco format visibility \in {0, 1, 2}
    visibility = torch.where(max_values > threshold, 2, 0)

    return keypoints, visibility


def rescale_keypoints(
    keypoints: torch.Tensor,
    bboxes_xy: torch.Tensor = None,
    bboxes_wh: torch.Tensor = None,
    heatmap_size: Tuple[int, int] = None,
    from_heatmap: bool = True,
    to_image: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """`heatmap_size` and `bboxes_wh` must be specified if `from_heatmap` is `True`.

    `bboxes_xy` must be specified if `to_image` is `True`.

    Transforms keypoints from heatmap or bounding box (specified by `from_heatmap`)
    to bounding box or image (specified by `to_image`).
    NOTE: must be modified if space augmentations besides Resize are added.
    NOTE: keypoint visibility may be affected by some space augmentations.

    - keypoints: [batch_size, n_keypoints, 2]
    - bboxes_xy: [batch_size, 2]
    - bboxes_wh: [batch_size, 2]
    """
    device = device if torch.cuda.is_available() else "cpu"
    keypoints = keypoints.to(device)

    if from_heatmap:
        h, w = heatmap_size
        inv_heatmap_size = torch.tensor([w, h]).to(device)
        # rescale [0, heatmap_size] -> [0, 1]
        keypoints = keypoints / inv_heatmap_size

        bboxes_wh = bboxes_wh.to(device)
        # rescale [0, 1] -> [0, (padded / not padded) bboxes sizes]
        keypoints = keypoints * bboxes_wh.unsqueeze(1)

    if to_image:
        bboxes_xy = bboxes_xy.to(device)
        # shift [0, 0] -> [bbox_xmin, bbox_ymin]
        keypoints = keypoints + bboxes_xy.unsqueeze(1)

    return keypoints
