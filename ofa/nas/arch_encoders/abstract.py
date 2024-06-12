from abc import ABC, abstractmethod


class ArchEncoder(ABC):
    def __init__(self):
        # self.num_stages = None
        # self.ks_list = None
        # self.expand_ratio_list = None
        # self.depth_list = None
        self.image_size_list = None
        self.n_dim = 0

    # @property
    # @abstractmethod
    # def num_blocks(self):
    #     pass

    @abstractmethod
    def _build_info_dict(self, target):
        pass

    @abstractmethod
    def arch2feature(self, arch_dict):
        pass

    @abstractmethod
    def feature2arch(self, feature):
        pass

    @abstractmethod
    def random_sample_arch(self) -> dict:
        pass

    @abstractmethod
    def random_resample(self, arch_dict: dict, mutate_prob: int) -> None:
        pass
