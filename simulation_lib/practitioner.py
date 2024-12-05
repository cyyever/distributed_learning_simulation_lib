from cyy_torch_toolbox import Config, DatasetCollection, MachineLearningPhase, Trainer
from cyy_torch_toolbox.dataset import SamplerBase, SplitBase


class Practitioner:
    def __init__(self, practitioner_id: int) -> None:
        self.__id: int = practitioner_id
        self.__worker_id = practitioner_id
        self._dataset_sampler: dict[str, SamplerBase | SplitBase] = {}

    @property
    def id(self):
        return self.__id

    @property
    def worker_id(self):
        return self.__worker_id

    def set_worker_id(self, worker_id: int) -> None:
        self.__worker_id = worker_id

    def set_sampler(self, sampler: SamplerBase | SplitBase) -> None:
        collection_name = sampler.dataset_collection.name
        assert collection_name not in self._dataset_sampler
        self._dataset_sampler[collection_name] = sampler

    def has_dataset(self, name: str) -> bool:
        return name in self._dataset_sampler

    def create_dataset_collection(self, config: Config) -> DatasetCollection:
        sampler = self._dataset_sampler[config.dc_config.dataset_name]
        assert sampler.dataset_collection is not None
        if isinstance(sampler, SplitBase):
            return sampler.sample(part_id=self.__worker_id)
        return sampler.sample()

    def create_trainer(self, config: Config) -> Trainer:
        trainer = config.create_trainer(
            dc=self.create_dataset_collection(config=config)
        )
        sampler = self._dataset_sampler[config.dc_config.dataset_name]
        trainer.dataset_collection.remove_dataset(phase=MachineLearningPhase.Test)
        assert sampler.dataset_collection.has_dataset(MachineLearningPhase.Test)
        return trainer
