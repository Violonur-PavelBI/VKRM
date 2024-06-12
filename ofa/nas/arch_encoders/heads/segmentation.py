# TODO: подумать.

import random

import numpy as np

from ..abstract import ArchEncoder
from ofa.utils.configs_dataclasses import SegmentationHeadConfig


class SegmentationHeadEncoder(ArchEncoder):
    def __init__(self, head_config: SegmentationHeadConfig):
        self.mid_channels = head_config.mid_channels
        if not isinstance(self.mid_channels, list):
            self.mid_channels = [self.mid_channels]

        self.n_dim = 0
        self.mid_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="mid")

    def _build_info_dict(self, target):
        if target == "mid":
            target_dict = self.mid_info
            target_dict["L"].append(self.n_dim)
            for mid_ch in self.mid_channels:
                target_dict["val2id"][mid_ch] = self.n_dim
                target_dict["id2val"][self.n_dim] = mid_ch
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)
        else:
            raise NotImplementedError

    def arch2feature(self, arch_dict):
        mid = arch_dict["mid_channels"]
        feature = np.zeros(self.n_dim)
        feature[self.mid_info["val2id"][mid]] = 1
        return feature

    def feature2arch(self, feature):
        for j in range(self.mid_info["L"][0], self.mid_info["R"][0]):
            if feature[j] == 1:
                mid = self.mid_info["id2val"][j]
        arch_dict = {"mid_channels": mid}
        return arch_dict

    def random_sample_arch(self):
        return {"mid_channels": random.choice(self.mid_channels)}

    def random_resample(self, arch_dict, mutate_prob):
        if random.random() < mutate_prob:
            arch_dict["mid_channels"] = random.choice(self.mid_channels)
