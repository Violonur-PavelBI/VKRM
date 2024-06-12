from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import albumentations as albu
import cv2
import matplotlib.pyplot as plt
import torch
from matplotlib import axes, patches
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable

if TYPE_CHECKING:
    from ...keypoints import Context


def add_image_to_axis(
    axis: axes.Axes,
    image_path: str,
    bbox: None,
    pad_size: None,
    crop: bool = True,
    pad: bool = False,
):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if crop:
        x, y, w, h = bbox
        image = image[
            math.floor(y) : math.ceil(y + h), math.floor(x) : math.ceil(x + w)
        ]

    if pad:
        w_pad, h_pad = pad_size
        transform = albu.augmentations.geometric.PadIfNeeded(
            min_height=int(h_pad),
            min_width=int(w_pad),
            always_apply=True,
            border_mode=cv2.BORDER_CONSTANT,
        )
        image = transform(image=image)["image"]

    axis.imshow(image)


def add_keypoints_to_axis(
    axis: axes.Axes, keypoints, visibility, classes_idxs, colour, size=10
):
    if keypoints is not None:
        if visibility is None:
            visibility = [2] * len(keypoints)
        for (x, y), v, class_id in zip(keypoints, visibility, classes_idxs):
            if v == 2:
                axis.scatter(x, y, c=colour, s=size)
                axis.annotate(
                    class_id.item() if isinstance(class_id, torch.Tensor) else class_id,
                    (x + 2, y + 2),
                    c=colour,
                    fontsize=size,
                )


def add_bbox_to_axis(axis, bbox, colour, line_width=2):
    """Draw bounding box on an image."""
    if bbox is not None:
        x, y, w, h = bbox
        box = patches.Rectangle(
            (x, y), w, h, edgecolor=colour, facecolor="none", lw=line_width
        )
        axis.add_patch(box)


def plot_image(
    axis: axes.Axes,
    image_path: str,
    index: int = None,
    keypoints_gt=None,
    visibility_gt=None,
    keypoints_pred=None,
    visibility_pred=None,
    classes_idx=None,
    bbox=None,
    pad_size=None,
    metrics_dict: Dict[str, float] = None,
    crop=False,
    pad=True,
    title=False,
):
    add_image_to_axis(axis, image_path, bbox, pad_size, crop, pad)

    if classes_idx is None:
        if keypoints_gt is not None:
            classes_idx = range(len(keypoints_gt))
        elif keypoints_pred is not None:
            classes_idx = range(len(keypoints_pred))

    add_keypoints_to_axis(axis, keypoints_gt, visibility_gt, classes_idx, "blue")
    add_keypoints_to_axis(axis, keypoints_pred, visibility_gt, classes_idx, "red")

    # add_bbox_to_axis(axis, bbox, "white")

    if title:
        if metrics_dict is not None:
            metrics = [f"{k}: {v:.4f}" for k, v in metrics_dict.items()]
            for i in range(1, len(metrics)):
                if i % 3 == 0:
                    metrics[i] = f"\n{metrics[i]}"
                else:
                    metrics[i] = f"  {metrics[i]}"
            title = "".join(metrics)
        else:
            title = os.path.split(image_path)[1]
            if index is not None:
                title = f"{index}: {title}"
        axis.set_title(title)


def visualise_keypoints(
    batch: Dict[str, Any],
    indices: List[int] = None,
    figsize: Tuple[int, int] = None,
    metrics: List[Dict[str, float]] = None,
    crop: bool = False,
    pad: bool = True,
):
    """
    Plot keypoints on images cropped to bounding boxes.

    - `keypoints`' coordinates in `batch` must be already rescaled from heatmaps back to bounding boxes.
    - `batch`: `Context`
    """
    batch_size = len(batch["image_path"])
    if indices is None:
        indices = range(batch_size)

    n = len(indices)
    ncols = min(n, 4)
    nrows = 1 + (n - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 5 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)

    for i, index in enumerate(indices):
        if index >= batch_size:
            continue

        row_i = i // ncols
        col_i = i % ncols
        axis = axes[row_i, col_i]

        # TODO: rename keys in batch
        image_path = batch["image_path"][index]
        keypoints_gt = batch["keypoints"][index] if "keypoints" in batch else None
        visibility_gt = batch["visibility"][index] if "visibility" in batch else None
        classes_idx = batch["classes_idx"][index] if "classes_idx" in batch else None
        bbox = batch["bbox"][index] if "bbox" in batch else None
        pad_size = batch["pad_size"][index] if "pad_size" in batch else None
        keypoints_pred = (
            batch["keypoints_pred"][index] if "keypoints_pred" in batch else None
        )
        visibility_pred = (
            batch["visibility_pred"][index] if "visibility_pred" in batch else None
        )
        metrics_dict = metrics[index] if metrics is not None else None

        plot_image(
            axis=axis,
            image_path=image_path,
            index=index,
            keypoints_gt=keypoints_gt,
            visibility_gt=visibility_gt,
            keypoints_pred=keypoints_pred,
            visibility_pred=visibility_pred,
            classes_idx=classes_idx,
            bbox=bbox,
            pad_size=pad_size,
            metrics_dict=metrics_dict,
            crop=crop,
            pad=pad,
            title=True,
        )

    legend = [
        patches.Patch(facecolor="blue", label="gt"),
        patches.Patch(facecolor="red", label="pred"),
    ]
    fig.legend(handles=legend, loc="right")

    plt.close()
    return fig


def visualise_heatmaps(heatmaps, figsize=None):
    """
    - heatmaps: [N, C, H, W]
    """
    heatmaps_sum = torch.sum(heatmaps, dim=1).detach().cpu().numpy()

    n = len(heatmaps)
    ncols = min(n, 4)
    nrows = 1 + (n - 1) // ncols
    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)
    fig = plt.figure(figsize=figsize)
    axes = ImageGrid(
        fig,
        111,
        nrows_ncols=(nrows, ncols),
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.05,
    )
    for i in range(n):
        axis = axes[i]
        imc = axis.imshow(heatmaps_sum[i], cmap="hot", interpolation="nearest")
    fig.colorbar(imc, cax=axes.cbar_axes[0])
    plt.close()
    return fig


def visualise_keypoints_and_heatmaps(
    context: Context, pad: bool = True, indices: List = None
):
    batch_size = len(context["image_path"])
    if indices is None:
        indices = list(range(batch_size))
    else:
        indices = list(set(indices) & set(range(batch_size)))
    n = len(indices)

    f, ax = plt.subplots(nrows=n, ncols=2, figsize=(8, 4 * n), squeeze=False)

    heatmaps_sum = torch.sum(context["output"][indices], dim=1).detach().cpu().numpy()
    for i, index in enumerate(indices):
        plot_image(
            axis=ax[i, 0],
            image_path=context["image_path"][index],
            index=index,
            keypoints_gt=context["keypoints_rescaled"][index],
            visibility_gt=context["visibility"][index],
            keypoints_pred=context["keypoints_pred_rescaled"][index],
            visibility_pred=context["visibility_pred"][index],
            classes_idx=context["classes_idx"][index],
            bbox=context["bbox"][index],
            pad_size=context["pad_size"][index],
            metrics_dict=None,
            crop=True,
            pad=pad,
            title=False,
        )

        imc = ax[i, 1].imshow(heatmaps_sum[i], cmap="hot", interpolation="nearest")
        divider = make_axes_locatable(ax[i, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        f.colorbar(imc, cax=cax)

    legend = [
        patches.Patch(facecolor="blue", label="gt"),
        patches.Patch(facecolor="red", label="pred"),
    ]
    f.legend(handles=legend, loc="upper center")

    plt.close()
    return f
