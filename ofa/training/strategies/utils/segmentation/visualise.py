from __future__ import annotations

from typing import TYPE_CHECKING, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from ofa.utils import attach_palette_to_mask

if TYPE_CHECKING:
    from ...segmentation import Context

from ..detection_yolov4.visualise import denorm


def visualise_segmentation(
    context: Context,
    dataset_classes: int,
    background: bool = False,
    indices: List = None,
    close_figure=True,
    dpi=100,
) -> plt.Figure:
    batch_size = context["target"].size(0)
    if indices is None:
        indices = list(range(batch_size))
    else:
        indices = list(set(indices) & set(range(batch_size)))
    n = len(indices)
    denormalise = denorm()

    argmax_target = torch.argmax(context["target"][indices], dim=1).cpu().numpy()

    detached_out = context["output"][indices].detach().cpu()
    argmax_pred = torch.argmax(detached_out, dim=1).numpy()

    f, ax = plt.subplots(n, 3, sharey=True, figsize=(12, 4 * n), squeeze=False, dpi=dpi)
    for i, index in enumerate(indices):
        img = context["image"][index]
        img: np.ndarray = img.permute(1, 2, 0).detach().cpu().numpy()
        img = denormalise(image=img)["image"]
        img = img.round().astype("uint8")

        plot_target = attach_palette_to_mask(
            argmax_target[i], dataset_classes, background
        )
        plot_pred = attach_palette_to_mask(argmax_pred[i], dataset_classes, background)

        ax[i, 0].imshow(img)
        ax[i, 0].set_title("Изображение")
        ax[i, 1].imshow(plot_target)
        ax[i, 1].set_title("Целевая маска")
        ax[i, 2].imshow(plot_pred)
        ax[i, 2].set_title("Прогноз сети")

    if close_figure:
        plt.close()
    return f
