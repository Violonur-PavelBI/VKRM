from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

import albumentations as albu
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches

if TYPE_CHECKING:
    from ...detection_yolov4 import Context


def denorm(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    return albu.Normalize(
        mean=-np.array(mean) / np.array(std),
        std=1 / (255 * np.array(std)),
        max_pixel_value=1,
    )


def visualise_detection(
    context: Context,
    class_names: List[str],
    indices: List[int] = None,
    max_cols: int = 3,
    close_fig: bool = True,
    metrics: Tuple[str, List[List[float]]] = None,
    fig_size: float = 4,
    dpi: int = 100,
    visualise_preds: bool = True,
    visualise_targets: bool = False,
    show_name: bool = False,
):
    batch_size = len(context["image_path"])
    if indices is None:
        indices = list(range(batch_size))
    else:
        indices = list(set(indices) & set(range(batch_size)))
    n = len(indices)
    denormalise = denorm()

    ncols = min(n, max_cols)
    nrows = 1 + (n - 1) // ncols
    f, axes = plt.subplots(
        nrows,
        ncols,
        squeeze=False,
        figsize=(fig_size * ncols, fig_size * nrows),
        dpi=dpi,
    )
    for i, index in enumerate(indices):
        col_i = i % ncols
        row_i = i // ncols
        axis: plt.Axes = axes[row_i][col_i]

        img = context["image"][index]
        img: np.ndarray = img.permute(1, 2, 0).detach().cpu().numpy()
        img = denormalise(image=img)["image"]
        img = img.round().astype("uint8")
        axis.imshow(img)

        if visualise_targets:
            height, width = img.shape[:2]
            for t in context["target"]:
                img_idx, label, xc, yc, w, h = t.cpu()
                label = int(label.item())
                if img_idx != index:
                    continue

                xc = xc * width
                yc = yc * height
                w = w * width
                h = h * height

                xmin = xc - w / 2
                ymin = yc - h / 2
                box = patches.Rectangle(
                    (xmin, ymin),
                    w,
                    h,
                    edgecolor="green",
                    facecolor="none",
                    lw=2,
                )
                axis.add_patch(box)

                text = f"{class_names[label]}"
                axis.text(xmin, ymin + h + 12, text, c="green", fontsize=8)

        if visualise_preds:
            for t in context["target_pred"][index]:
                if t.size(0) == 6:
                    xmin, ymin, xmax, ymax, conf, label = t.cpu()
                    label = int(label.item())
                else:
                    xmin, ymin, xmax, ymax, conf = t.cpu()
                    label = 0
                if conf == -1:
                    break

                w = xmax - xmin
                h = ymax - ymin
                box = patches.Rectangle(
                    (xmin, ymin),
                    w,
                    h,
                    edgecolor="red",
                    facecolor="none",
                    lw=2,
                )
                axis.add_patch(box)

                text = f"{class_names[label]} {conf:.2f}"
                axis.text(xmin, ymin - 5, text, c="red", fontsize=8)

        axis.axis("off")
        if metrics is not None:
            name, values = metrics[0], metrics[1][index]
            values = " ".join([f"{v:.2f}" for v in values])
            axis.set_title(f"{name}=({values})")
        if show_name:
            file_name = context["image_path"][index]
            file_name = file_name.split("/")[-2:]
            file_name = "/".join(file_name)
            axis.set_title(file_name)

    if close_fig:
        plt.close()

    return f
