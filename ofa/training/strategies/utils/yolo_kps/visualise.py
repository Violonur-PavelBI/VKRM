from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

import albumentations as albu
import matplotlib.pyplot as plt
from matplotlib import patches
import torch

if TYPE_CHECKING:
    from ...yolo_kps import Context


def denorm(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    return albu.Normalize(
        mean=-np.array(mean) / np.array(std),
        std=1 / (255 * np.array(std)),
        max_pixel_value=1,
    )


def generate_colours(count: int, seed=42) -> List[int]:
    """Возвращает список состоящий из случайных числе длинной count * 3"""
    g = torch.Generator()
    g.manual_seed(seed)
    colours = list(map(float, torch.rand(size=(count * 3,), generator=g)))
    return colours


def _put_targets_on_axis(
    context: Context, index: int, axis: plt.Axes, img: np.ndarray
) -> None:
    axis.imshow(img)
    axis.axis("off")

    height, width = img.shape[:2]
    for t in context["target"]:
        img_idx, label, xc, yc, w, h = t[:6].cpu().clone()
        kps = t[6:].cpu().clone()
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

        if kps.shape[0] > 0:
            kps = kps.view(-1, 3)
            kps[:, :2] *= torch.tensor([width, height])
            n_kpt = kps.size(0)
            colours = generate_colours(n_kpt)
            colours = torch.tensor(colours).view(n_kpt, 3)

            for colour, kpt in zip(colours, kps):
                x, y, v = kpt
                if v > 0:
                    point = patches.Circle(
                        (x, y), radius=7, facecolor=colour.tolist(), edgecolor="black"
                    )
                    axis.add_patch(point)


def _put_preds_on_axis(
    context: Context, index: int, axis: plt.Axes, img: np.ndarray
) -> None:
    axis.imshow(img)
    axis.axis("off")
    for t in context["target_pred"][index]:
        xmin, ymin, xmax, ymax, conf, label = t[:6].cpu().clone()
        kps = t[6:].cpu().clone()
        label = int(label.item())
        if label == -1:
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

        if kps.shape[0] > 0:
            n_kpt = kps.shape[0] // 3
            colors_arr = generate_colours(n_kpt)

            for i in range(0, kps.shape[0], 3):  # x,y,vis
                x, y, v = kps[i : i + 3]
                color = colors_arr[i : i + 3]

                if v > 0.5:  # conf check
                    point = patches.Circle(
                        (x, y), radius=6, facecolor=color, edgecolor="black"
                    )
                    axis.add_patch(point)

        text = f"conf: {conf:.2f}"
        axis.text(xmin, ymin - 5, text, c="red", fontsize=8)


def visualise_keypoints(
    context: Context,
    class_names: List[str],
    indices: List[int] = None,
    close_fig: bool = True,
    metrics: Tuple[str, List[List[float]]] = None,
    fig_size: float = 4,
    dpi: int = 100,
    visualise_preds: bool = True,
    visualise_targets: bool = True,
):
    batch_size = len(context["image_path"])
    if indices is None:
        indices = list(range(batch_size))
    else:
        indices = list(set(indices) & set(range(batch_size)))
    denormalise = denorm()

    ncols = 3
    nrows = len(indices)
    f, axes = plt.subplots(
        nrows,
        ncols,
        squeeze=False,
        figsize=(fig_size * ncols, fig_size * nrows),
        dpi=dpi,
    )
    for i, index in enumerate(indices):
        row_i = i

        imgage: torch.Tensor = context["image"][index]
        plot_img: np.ndarray = imgage.permute(1, 2, 0).detach().cpu().numpy()
        plot_img = denormalise(image=plot_img)["image"]
        plot_img = plot_img.round().astype("uint8")

        if visualise_targets:
            axis: plt.Axes = axes[row_i][0]
            axis.set_title("Разметка")
            _put_targets_on_axis(context, index, axis, plot_img)

        if visualise_preds:
            axis: plt.Axes = axes[row_i][1]
            axis.set_title("Прогноз")
            _put_preds_on_axis(context, index, axis, plot_img)

        if visualise_preds and visualise_targets:
            axis: plt.Axes = axes[row_i][2]
            axis.set_title("Наложение прогноза на разметку")
            _put_targets_on_axis(context, index, axis, plot_img)
            _put_preds_on_axis(context, index, axis, plot_img)

    if close_fig:
        plt.close()

    return f
