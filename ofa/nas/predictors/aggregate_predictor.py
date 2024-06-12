from typing import Dict, List, Union

from torch import nn, Tensor

from .predictors import Predictor


class AggregatePredictor(nn.Module):
    aggregation_policies = ["sum"]

    def __init__(
        self,
        predictor_backbone: Predictor,
        predictor_neck: Predictor,
        aggregation="sum",
    ) -> None:
        super().__init__()
        self.backbone = predictor_backbone
        self.neck = predictor_neck
        self.aggregation = aggregation
        if self.aggregation not in self.aggregation_policies:
            raise ValueError(f"{aggregation = } not in {self.aggregation_policies}")

    def predict(self, arch_dict_list: Union[Dict, List[Dict]]) -> Tensor:
        if isinstance(arch_dict_list, dict):
            arch_dict_list = [arch_dict_list]

        backbone_dict_list = [arch_dict["backbone"] for arch_dict in arch_dict_list]
        neck_dict_list = [arch_dict["neck"] for arch_dict in arch_dict_list]

        backbone_predict = self.backbone.predict(backbone_dict_list)
        neck_predict = self.neck.predict(neck_dict_list)

        if self.aggregation == "sum":
            predict = backbone_predict + neck_predict
        return predict
