import random
from typing import Any

import torch
from cyy_naive_lib.algorithm.mapping_op import (
    get_mapping_items_by_key_order,
    get_mapping_values_by_key_order,
)
from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import (
    ClassificationDatasetCollection,
    DatasetCollection,
    DatasetCollectionSplit,
    MachineLearningPhase,
    RandomSplit,
    SplitBase,
)
from cyy_torch_toolbox.dataset import (  # noqa: F401
    SampleInfo,
    get_dataset_collection_sampler,
    get_dataset_collection_split,
    global_sampler_factory,
)


class RandomLabelIIDSplit(SplitBase):
    def __init__(
        self,
        dataset_collection: DatasetCollection,
        part_number: int,
        sampled_class_number: int,
    ) -> None:
        super().__init__(dataset_collection=dataset_collection, part_number=part_number)
        assert isinstance(dataset_collection, ClassificationDatasetCollection)
        assert not dataset_collection.is_mutilabel()
        labels = dataset_collection.get_labels()
        assert sampled_class_number < len(labels)
        assigned_labels = [
            random.sample(list(labels), sampled_class_number)
            for _ in range(part_number)
        ]

        # Assure that all labels are allocated
        assert len(labels) == len(set(sum(assigned_labels, start=[])))

        for phase in self.get_phases():
            if phase in self._dataset_indices:
                continue
            self._dataset_indices[phase] = {}
            for worker_id, indices in enumerate(
                self._samplers[phase].split_indices(
                    part_proportions=[
                        {label: 1 for label in labels} for labels in assigned_labels
                    ]
                )
            ):
                self._dataset_indices[phase][worker_id] = SampleInfo(indices=indices)
        for worker_id, worker_labels in enumerate(assigned_labels):
            log_info(
                "worker %s has assigned worker_labels %s", worker_id, worker_labels
            )
            worker_indices = self._dataset_indices[MachineLearningPhase.Training][
                worker_id
            ].indices
            assert worker_indices is not None
            training_set_size = len(worker_indices)
            log_info("worker %s has training set size %s", worker_id, training_set_size)


class DirichletSplit(DatasetCollectionSplit):
    def __init__(
        self,
        dataset_collection: DatasetCollection,
        concentration: float | list[dict[Any, float]],
        part_number: int,
    ) -> None:
        if not isinstance(concentration, list):
            assert isinstance(dataset_collection, ClassificationDatasetCollection)
            all_labels = dataset_collection.get_labels()
            concentration = [
                {label: float(concentration) for label in all_labels}
            ] * part_number
        assert isinstance(concentration, list)
        assert len(concentration) == part_number
        part_proportions: list[dict] = []
        for worker_concentration in concentration:
            concentration_tensor = torch.tensor(
                list(get_mapping_values_by_key_order(worker_concentration))
            )
            prob = torch.distributions.dirichlet.Dirichlet(
                concentration_tensor
            ).sample()
            part_proportions.append({})
            for (k, _), label_prob in zip(
                get_mapping_items_by_key_order(worker_concentration), prob, strict=False
            ):
                part_proportions[-1][k] = label_prob

        super().__init__(
            dataset_collection=dataset_collection, part_proportions=part_proportions
        )


global_sampler_factory.register("random_label_iid", RandomLabelIIDSplit)
global_sampler_factory.register("random_label_iid_split", RandomLabelIIDSplit)
global_sampler_factory.register("dirichlet_split", DirichletSplit)
global_sampler_factory.register("random_split", RandomSplit)

__all__ = ["get_dataset_collection_sampler", "get_dataset_collection_split"]
