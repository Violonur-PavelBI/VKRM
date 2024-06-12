from .acc_dataset import AccuracyDataset
from .predictors import Predictor, PredictorCatboost


class ArchToNumberDataset:
    """
    Временное решение.
    Данный вид датасета отличается тем, что у него нет функционала для сборки.
    """

    train_val_split = AccuracyDataset.train_val_split
    build_data_loaders = AccuracyDataset.build_data_loaders

    def __init__(self, path) -> None:
        self.dict_path = path
