import os
import warnings

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from dltools.configs.dataset import DetectionYoloV4DatasetConfig
from matplotlib.cm import ScalarMappable
from dltools.data_providers.detection_yolov4.utils.annotations import read_coco_ann
import copy


def build_auto_anchors(
    args: DetectionYoloV4DatasetConfig,
    num_strides: int = 3,
    scatter: bool = False,
    treshold: float = 0.001,
    min_box_size: int = 4,
    init_num_clust: int = 3,
    weight_change_epoch: int = 5,
    weight_transmission: int = 50,
):
    """Найти оптимальные анкеры для датасета


    Args:
        args (DetectionYoloV4DatasetConfig): Конфигурация датасета  с указанием пути до аннотации и целевого размера изображений
        num_strides (int, optional): Количество выходов сети. Defaults to 3.
        scatter (bool, optional): Включение визуализацию данных и анкеров с передачей их на выход ф-ии. Defaults to False.
        treshold (float, optional): Допустимое относительное количество данных не входящих в область действий анкеров. Defaults to 0.001.
        min_box_size (int, optional):  Минимальный размер бокса для допуска к расчётам анкеров в пикселях. Defaults to 4.
        init_num_clust (int, optional): Начальное количество анкеров в каждом выходе сети. Defaults to 3.
        weight_change_epoch (int, optional): Вес добавляемый точкам не входящим в область покрытия анкеро. Defaults to 5.
        weight_transmission (int, optional): Граница веса точки после которой увеличивается количество анкеров. Defaults to 50.

    Returns:
        anchors: Рассчитанные анкеры
        figure: Фигура содержащая графики данных в обычном и логарифмическом масштабе с соответствующей легендой.

        - If scatter is False, returns a single object anchors.
        - If scatter is True, returns two objects anchors and figure.
    """

    annot_path = os.path.join(args.dataset_path, args.ann_file_train)
    data = read_coco_ann(annot_path)
    image_size = args.image_size
    box_sizes_list = []
    for imge in data:
        boxs = imge["bboxes"]
        or_im_size = imge["image_size"]
        scale = min((image_size[0] / or_im_size[0]), (image_size[1] / or_im_size[1]))
        for box in boxs:
            size_images = [box[2] * scale, box[3] * scale]
            if min(size_images) > min_box_size:
                box_sizes_list.append(size_images)

    set_size = len(box_sizes_list)

    logarithmic_dataset = np.log(np.array(box_sizes_list))
    num_clust = num_strides * init_num_clust
    weigh = np.ones(set_size)
    permit = False
    while True:

        if num_clust > 50:
            warnings.warn(f"When searching, the number of anchors exceeded 50\n")
            break

        box_clusterer = KMeans(n_clusters=num_clust, n_init="auto")
        labels = box_clusterer.fit_predict(logarithmic_dataset, sample_weight=weigh)

        garbage_items = np.ones(set_size)
        for i, center in enumerate(box_clusterer.cluster_centers_):
            diff = np.abs(logarithmic_dataset - center)
            garbage_items *= np.max(diff, axis=1) >= np.log(4)
        index_ub_weigh = np.where(garbage_items == True)[0].tolist()

        if len(index_ub_weigh) / set_size > treshold:
            weight_change = np.zeros(set_size)
            weight_change[index_ub_weigh] = weight_change_epoch
            weigh = weigh + weight_change
            if sum((weigh > weight_transmission) * weight_change) > 0:
                num_clust += 1
                permit = True
            continue

        if num_clust - 1 < num_strides or permit:
            break
        num_clust -= 1

    sorted_indices = np.argsort(box_clusterer.cluster_centers_[:, 0])

    cluster_centers = box_clusterer.cluster_centers_[sorted_indices]
    cluster_exp = np.exp(cluster_centers)

    anchor_clusterer = KMeans(n_clusters=num_strides, n_init="auto")
    index = anchor_clusterer.fit_predict(cluster_exp)

    anchors = [[] for _ in range(num_strides)]
    cluster_desc_sorted_exp = np.array(cluster_exp, int).tolist()
    for i in range(num_clust):
        anchors[index[i]].append(cluster_desc_sorted_exp[i])

    if not scatter:
        return anchors

    figure, axes = plt.subplots(1, 2, layout="constrained", figsize=(15, 5))
    figure.suptitle("Кластеризация данных с анкерами в качестве центров", fontsize=12)
    scatter_dataset = logarithmic_dataset
    color = copy.copy(labels)
    max_color = np.max(color)
    scatter_dataset = np.append(
        scatter_dataset, np.log(cluster_desc_sorted_exp), axis=0
    )

    for i in range(len(cluster_centers)):
        color = np.append(color, max_color + index[i] + 1)
    color[index_ub_weigh] = max_color + num_strides + 1
    exp_scatter_dataset = np.exp(scatter_dataset)
    sm = ScalarMappable(cmap="viridis")
    sm.set_array(color)

    axes[0].scatter(
        x=scatter_dataset[:, 0], y=scatter_dataset[:, 1], c=sm.to_rgba(color)
    )

    for class_label in np.sort(np.unique(color)):
        mask = color == class_label
        if class_label <= max_color:
            axes[0].scatter(
                scatter_dataset[mask, 0],
                scatter_dataset[mask, 1],
                c=sm.to_rgba(color[mask]),
                label=f"Кластер {class_label}",
            )
            axes[1].scatter(
                exp_scatter_dataset[mask, 0],
                exp_scatter_dataset[mask, 1],
                c=sm.to_rgba(color[mask]),
                label=f"Кластер {class_label}",
            )
        elif class_label == max_color + num_strides + 1:
            axes[0].scatter(
                scatter_dataset[mask, 0],
                scatter_dataset[mask, 1],
                c="red",
                label=f"Неудовлетворяющие условию точки",
            )
            axes[1].scatter(
                exp_scatter_dataset[mask, 0],
                exp_scatter_dataset[mask, 1],
                c="red",
                label=f"Неудовлетворяющие условию точки",
            )
        else:
            axes[0].scatter(
                scatter_dataset[mask, 0],
                scatter_dataset[mask, 1],
                c=sm.to_rgba(color[mask]),
                label=f"Анкеры в stride {class_label-max_color}",
            )
            axes[1].scatter(
                exp_scatter_dataset[mask, 0],
                exp_scatter_dataset[mask, 1],
                c=sm.to_rgba(color[mask]),
                label=f"Анкеры в stride {class_label-max_color}",
            )
    axes[0].set_xlabel("log height")
    axes[0].set_ylabel("log width")
    axes[0].set_title("В логарифмическом масштабе")
    axes[1].set_xlabel("height")
    axes[1].set_ylabel("width")
    axes[1].set_title("В реальном масштабе")

    axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    return anchors, figure
