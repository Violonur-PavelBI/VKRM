import random

import numpy as np

from ..abstract import ArchEncoder
from ofa.utils.configs_dataclasses import BackboneConfig


class MobileNetArchEncoder(ArchEncoder):
    SPACE_TYPE = "mbv3"

    def __init__(self, backbone_config: BackboneConfig):
        self.ks_list = backbone_config.ks_list
        self.expand_ratio_list = [int(e) for e in backbone_config.expand_ratio_list]
        self.depth_list = backbone_config.depth_list
        self.num_stages = backbone_config.n_stages

        self.n_dim = 0
        self.k_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="k")
        self.e_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="e")

    @property
    def num_blocks(self):
        if self.SPACE_TYPE == "mbv3":
            return self.num_stages * max(self.depth_list)
        elif self.SPACE_TYPE == "mbv2":
            return (self.num_stages - 1) * max(self.depth_list) + 1
        else:
            raise NotImplementedError

    def _build_info_dict(self, target):
        if target == "k":
            target_dict = self.k_info
            choices = self.ks_list
        elif target == "e":
            target_dict = self.e_info
            choices = self.expand_ratio_list
        else:
            raise NotImplementedError

        for i in range(self.num_blocks):
            target_dict["val2id"].append({})
            target_dict["id2val"].append({})
            target_dict["L"].append(self.n_dim)
            for val in choices:
                target_dict["val2id"][i][val] = self.n_dim
                target_dict["id2val"][i][self.n_dim] = val
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)

    def arch2feature(self, arch_dict):
        ks_list = arch_dict["ks"]
        e_list = arch_dict["e"]
        d_list = arch_dict["d"]

        feature = np.zeros(self.n_dim)
        for i in range(self.num_blocks):
            nowd = i % max(self.depth_list)
            stg = i // max(self.depth_list)
            ks, e, d = ks_list[i], e_list[i], d_list[stg]
            if nowd < d:
                feature[self.k_info["val2id"][i][ks]] = 1
                feature[self.e_info["val2id"][i][e]] = 1
        return feature

    def feature2arch(self, feature):
        d = 0
        ks_list, e_list, d_list = [], [], []
        for i in range(self.num_blocks):
            skip = True
            for j in range(self.k_info["L"][i], self.k_info["R"][i]):
                if feature[j] == 1:
                    ks_list.append(self.k_info["id2val"][i][j])
                    skip = False

            for j in range(self.e_info["L"][i], self.e_info["R"][i]):
                if feature[j] == 1:
                    e_list.append(self.e_info["id2val"][i][j])
                    assert not skip

            if skip:
                e_list.append(0)
                ks_list.append(0)
            else:
                d += 1

            if (i + 1) % max(self.depth_list) == 0 or (i + 1) == self.num_blocks:
                d_list.append(d)
                d = 0

        arch_dict = {"ks": ks_list, "e": e_list, "d": d_list}
        return arch_dict

    def random_sample_arch(self):
        return {
            "ks": random.choices(self.ks_list, k=self.num_blocks),
            "e": random.choices(self.expand_ratio_list, k=self.num_blocks),
            "d": random.choices(self.depth_list, k=self.num_stages),
        }

    def random_resample(self, arch_dict, mutate_prob):
        for i in range(self.num_blocks):
            if random.random() < mutate_prob:
                arch_dict["ks"][i] = random.choice(self.ks_list)
            if random.random() < mutate_prob:
                arch_dict["e"][i] = random.choice(self.expand_ratio_list)
        for i in range(self.num_stages):
            if random.random() < mutate_prob:
                arch_dict["d"][i] = random.choice(self.depth_list)


class MobileNetArchEncoderCatboost(MobileNetArchEncoder):
    """
    Энкодер архитектуры MobileNet, используемый в Catboost.
    Отличается от базового энкодера MobileNet тем, что кодирование
    почти полностью происходит внутри алгоритма Catboost.

    Метод feature2arch пока не реализован, из-за чего
    распределённый эволюционный поиск работать не будет.
    """

    def __init__(self, backbone_config: BackboneConfig):
        self.ks_list = backbone_config.ks_list
        self.max_ks_list = max(self.ks_list)
        self.expand_ratio_list = [int(e) for e in backbone_config.expand_ratio_list]
        self.max_expand_ratio_list = max(self.expand_ratio_list)
        self.depth_list = backbone_config.depth_list
        self.num_stages = backbone_config.n_stages

        self.n_dim = self.num_blocks * 2

        self.k_info = None
        self.e_info = None

    def _build_info_dict(self, target):
        pass

    def arch2feature(self, arch_dict):
        ks_list = arch_dict["ks"]
        e_list = arch_dict["e"]
        d_list = arch_dict["d"]

        feature = np.zeros(self.n_dim)
        for i in range(self.num_blocks):
            nowd = i % max(self.depth_list)
            stg = i // max(self.depth_list)
            d = d_list[stg]
            if nowd < d:
                feature[2 * i] = ks_list[i] / self.max_ks_list
                feature[2 * i + 1] = e_list[i] / self.max_expand_ratio_list
        return feature

    def feature2arch(self, feature):
        # TODO: implement for distributed evolution
        raise NotImplementedError
