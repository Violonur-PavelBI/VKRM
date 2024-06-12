import random
from typing import List

import numpy as np

from ..abstract import ArchEncoder
from ofa.utils.configs_dataclasses import BackboneConfig


class ResNetArchEncoder(ArchEncoder):
    def __init__(self, backbone_config: BackboneConfig):
        self.expand_ratio_list = backbone_config.expand_ratio_list
        self.depth_list = backbone_config.depth_list
        self.width_mult_list = backbone_config.width_mult_list
        self.width_idx_list = list(range(len(self.width_mult_list)))
        self.base_depth_list = backbone_config.base_depth_list

        self.n_dim = 0
        self.input_stem_d_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="input_stem_d")
        self.width_mult_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="width_mult")
        self.e_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="e")

    @property
    def num_stages(self):
        return len(self.base_depth_list)

    @property
    def num_blocks(self):
        return sum(self.base_depth_list) + self.num_stages * max(self.depth_list)

    def _build_info_dict(self, target):
        if target == "input_stem_d":
            target_dict = self.input_stem_d_info
            target_dict["L"].append(self.n_dim)
            for skip in [0, 1]:
                target_dict["val2id"][skip] = self.n_dim
                target_dict["id2val"][self.n_dim] = skip
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)
        elif target == "e":
            target_dict = self.e_info
            choices = self.expand_ratio_list
            for i in range(self.num_blocks):
                target_dict["val2id"].append({})
                target_dict["id2val"].append({})
                target_dict["L"].append(self.n_dim)
                for e in choices:
                    target_dict["val2id"][i][e] = self.n_dim
                    target_dict["id2val"][i][self.n_dim] = e
                    self.n_dim += 1
                target_dict["R"].append(self.n_dim)
        elif target == "width_mult":
            target_dict = self.width_mult_info
            choices = self.width_idx_list
            for i in range(self.num_stages + 2):
                target_dict["val2id"].append({})
                target_dict["id2val"].append({})
                target_dict["L"].append(self.n_dim)
                for w in choices:
                    target_dict["val2id"][i][w] = self.n_dim
                    target_dict["id2val"][i][self.n_dim] = w
                    self.n_dim += 1
                target_dict["R"].append(self.n_dim)
        else:
            raise NotImplementedError

    def arch2feature(self, arch_dict):
        d_list = arch_dict["d"]
        e_list = arch_dict["e"]
        w_list = arch_dict["w"]

        input_stem_skip = 1 if d_list[0] > 0 else 0
        d_list = d_list[1:]

        feature = np.zeros(self.n_dim)
        feature[self.input_stem_d_info["val2id"][input_stem_skip]] = 1
        for i in range(self.num_stages + 2):
            w = w_list[i]
            feature[self.width_mult_info["val2id"][i][w]] = 1

        start_pt = 0
        for i, base_depth in enumerate(self.base_depth_list):
            depth = base_depth + d_list[i]
            for j in range(start_pt, start_pt + depth):
                e = e_list[j]
                feature[self.e_info["val2id"][j][e]] = 1
            start_pt += max(self.depth_list) + base_depth

        return feature

    def feature2arch(self, feature):
        for j in range(self.input_stem_d_info["L"][0], self.input_stem_d_info["R"][0]):
            if feature[j] == 1:
                input_stem_skip = 2 * self.input_stem_d_info["id2val"][j]

        d_list, e_list, w_list = [input_stem_skip], [], []
        for i in range(self.num_stages + 2):
            for j in range(self.width_mult_info["L"][i], self.width_mult_info["R"][i]):
                if feature[j] == 1:
                    w_list.append(self.width_mult_info["id2val"][i][j])

        d, skipped, stage_id = 0, 0, 0
        for i in range(self.num_blocks):
            skip = True
            for j in range(self.e_info["L"][i], self.e_info["R"][i]):
                if feature[j] == 1:
                    e_list.append(self.e_info["id2val"][i][j])
                    skip = False
            if skip:
                e_list.append(0)
                skipped += 1
            else:
                d += 1

            max_stage_depth = max(self.depth_list) + self.base_depth_list[stage_id]
            if i + 1 == self.num_blocks or (skipped + d) % max_stage_depth == 0:
                d_list.append(d - self.base_depth_list[stage_id])
                d, skipped = 0, 0
                stage_id += 1

        arch_dict = {"d": d_list, "e": w_list, "w": w_list}
        return arch_dict

    def random_sample_arch(self):
        return {
            "ks": random.choices([3], k=self.num_blocks),  # TODO: убрать этот костыль
            "d": [random.choice([0, 2])]
            + random.choices(self.depth_list, k=self.num_stages),
            "e": random.choices(self.expand_ratio_list, k=self.num_blocks),
            "w": random.choices(self.width_idx_list, k=self.num_stages + 2),
        }

    def random_resample(self, arch_dict, mutate_prob):
        for i in range(self.num_blocks):
            if random.random() < mutate_prob:
                arch_dict["e"][i] = random.choice(self.expand_ratio_list)

        if random.random() < mutate_prob:
            arch_dict["d"][0] = random.choice([0, 2])
        for i in range(self.num_stages):
            if random.random() < mutate_prob:
                arch_dict["d"][1 + i] = random.choice(self.depth_list)

        for i in range(self.num_stages + 2):
            if random.random() < mutate_prob:
                arch_dict["w"][i] = random.choice(self.width_idx_list)
