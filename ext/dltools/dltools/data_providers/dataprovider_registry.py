from .base_provider import DataProvider
from loguru import logger

__all__ = ["DataProvidersRegistry"]


class DataProvidersRegistry:
    REGISTRY = {}

    @staticmethod
    def registry(data_provider_cls: DataProvider):
        DataProvidersRegistry.REGISTRY[data_provider_cls.NAME] = data_provider_cls
        return data_provider_cls

    @staticmethod
    def get_provider_by_name(name: str):
        try:
            return DataProvidersRegistry.REGISTRY[name]
        except KeyError as e:
            logger.info(f"{name} not found or not implemeted!")
            raise e
