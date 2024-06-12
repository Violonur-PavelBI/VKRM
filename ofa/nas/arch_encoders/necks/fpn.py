import random

import numpy as np

from ..abstract import ArchEncoder
from ofa.utils.configs_dataclasses import NeckConfig
from models.ofa.necks import neck_name2class


class FPNEncoder(ArchEncoder):
    def __init__(self, neck_config: NeckConfig):
        self.levels = neck_config.levels
        self.n_blocks = max(neck_config.depth_list)

        self.mid_channels = neck_config.mid_channels
        self.out_channels = neck_config.out_channel_list

        self.depth_list = neck_config.depth_list
        self.kernel_list = neck_config.kernel_list
        self.width_list = neck_config.width_list

        self.upsample_mode = neck_config.upsample_mode
        self.upsample_modes_list = neck_name2class[neck_config.type].upsample_modes_list

        self.n_dim = 0

        self.mid_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="mid")
        self.up_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="up")

        self.d_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="d")
        self.k_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="k")
        self.w_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="w")

    def _build_info_dict(self, target):
        if target in ["mid", "up"]:
            if target == "mid":
                target_dict = self.mid_info
                choices = self.width_list
            else:
                target_dict = self.up_info
                choices = self.upsample_modes_list
            self._fill_info_dict(target_dict, choices, 0)

        elif target == "d":
            for i in range(self.levels):
                self._fill_info_dict(self.d_info, self.depth_list, i)

        elif target in ["k", "w"]:
            if target == "k":
                target_dict = self.k_info
                choices = self.kernel_list
            else:
                target_dict = self.w_info
                choices = self.width_list
            for i in range(self.levels):
                for j in range(self.n_blocks):
                    index = i * self.n_blocks + j
                    self._fill_info_dict(target_dict, choices, index)

        else:
            raise NotImplementedError

    def _fill_info_dict(self, target_dict, choices, index):
        target_dict["val2id"].append({})
        target_dict["id2val"].append({})
        target_dict["L"].append(self.n_dim)
        for val in choices:
            target_dict["val2id"][index][val] = self.n_dim
            target_dict["id2val"][index][self.n_dim] = val
            self.n_dim += 1
        target_dict["R"].append(self.n_dim)

    def arch2feature(self, arch_dict):
        mid = arch_dict["mid"]
        up = arch_dict["up"]

        depth = arch_dict["out"]["d"]
        kernel = arch_dict["out"]["k"]
        width = arch_dict["out"]["w"]

        feature = np.zeros(self.n_dim)

        feature[self.mid_info["val2id"][0][mid]] = 1
        feature[self.up_info["val2id"][0][up]] = 1

        for i in range(self.levels):
            d = depth[i]
            feature[self.d_info["val2id"][i][d]] = 1
            for j in range(d):
                index = i * self.n_blocks + j
                k = kernel[i][j]
                feature[self.k_info["val2id"][index][k]] = 1
                w = width[i][j]
                feature[self.w_info["val2id"][index][w]] = 1

        return feature

    def feature2arch(self, feature):
        for i in range(self.mid_info["L"][0], self.mid_info["R"][0]):
            if feature[i] == 1:
                mid = self.mid_info["id2val"][0][i]
        for i in range(self.up_info["L"][0], self.up_info["R"][0]):
            if feature[i] == 1:
                up = self.up_info["id2val"][0][i]

        depth, kernel, width = [], [], []
        for level in range(self.levels):
            for i in range(self.d_info["L"][level], self.d_info["R"][level]):
                if feature[i] == 1:
                    d = self.d_info["id2val"][level][i]
                    depth.append(d)
            kernel.append([])
            width.append([])
            for block in range(self.n_blocks):
                index = level * self.n_blocks + block
                skip = True
                for i in range(self.k_info["L"][index], self.k_info["R"][index]):
                    if feature[i] == 1:
                        skip = False
                        k = self.k_info["id2val"][index][i]
                        kernel[level].append(k)
                for i in range(self.w_info["L"][index], self.w_info["R"][index]):
                    if feature[i] == 1:
                        assert not skip
                        w = self.w_info["id2val"][index][i]
                        width[level].append(w)
                if skip:
                    kernel[level].append(0)
                    width[level].append(0)

        out = {"d": depth, "k": kernel, "w": width}
        arch_dict = {"mid": mid, "up": up, "out": out}
        return arch_dict

    def random_sample_arch(self):
        out = {
            "d": random.choices(self.depth_list, k=self.levels),
            "k": [
                random.choices(self.kernel_list, k=self.n_blocks)
                for _ in range(self.levels)
            ],
            "w": [
                random.choices(self.width_list, k=self.n_blocks)
                for _ in range(self.levels)
            ],
        }
        return {
            "mid": random.choice(self.width_list),
            "up": self.upsample_mode,
            "out": out,
        }

    def random_resample(self, arch_dict, mutate_prob):
        if random.random() < mutate_prob:
            arch_dict["mid"] = random.choice(self.width_list)
        for i in range(self.levels):
            if random.random() < mutate_prob:
                arch_dict["out"]["d"][i] = random.choice(self.depth_list)
            for j in range(self.n_blocks):
                if random.random() < mutate_prob:
                    arch_dict["out"]["k"][i][j] = random.choice(self.kernel_list)
                if random.random() < mutate_prob:
                    arch_dict["out"]["w"][i][j] = random.choice(self.width_list)
