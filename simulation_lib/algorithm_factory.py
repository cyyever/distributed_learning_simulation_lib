import functools
import itertools
from collections.abc import Callable
from typing import Self

from cyy_naive_lib.log import log_warning
from cyy_naive_lib.topology import ClientEndpoint, ServerEndpoint

from .config import DistributedTrainingConfig
from .context import FederatedLearningContext


class CentralizedAlgorithmFactory:
    config: dict[str, dict] = {}

    @classmethod
    def register_algorithm(
        cls: type[Self],
        algorithm_name: str,
        client_cls: Callable,
        server_cls: Callable,
        client_endpoint_cls: None | Callable = None,
        server_endpoint_cls: None | Callable = None,
        algorithm_cls: None | Callable = None,
    ) -> None:
        assert algorithm_name not in cls.config
        if client_endpoint_cls is None:
            client_endpoint_cls = ClientEndpoint
        if server_endpoint_cls is None:
            server_endpoint_cls = ServerEndpoint
        cls.config[algorithm_name] = {
            "client_cls": client_cls,
            "server_cls": server_cls,
            "client_endpoint_cls": client_endpoint_cls,
            "server_endpoint_cls": server_endpoint_cls,
        }
        if algorithm_cls is not None:
            cls.config[algorithm_name]["algorithm_cls"] = algorithm_cls

    @classmethod
    def has_algorithm(cls, algorithm_name: str) -> bool:
        return algorithm_name in cls.config

    @classmethod
    def create_client(
        cls: type[Self],
        context: FederatedLearningContext,
        algorithm_name: str,
        kwargs: dict,
        endpoint_kwargs: dict,
        extra_kwargs: dict | None = None,
    ) -> None:
        config = cls.config[algorithm_name]
        if extra_kwargs is None:
            extra_kwargs = {}
        endpoint = context.create_client_endpoint(
            config["client_endpoint_cls"], **endpoint_kwargs
        )
        return config["client_cls"](endpoint=endpoint, **(kwargs | extra_kwargs))

    @classmethod
    def create_server(
        cls,
        context: FederatedLearningContext,
        algorithm_name: str,
        kwargs: dict,
        endpoint_kwargs: dict,
        extra_kwargs: dict | None = None,
    ) -> None:
        config = cls.config[algorithm_name]
        if extra_kwargs is None:
            extra_kwargs = {}
        endpoint = context.create_server_endpoint(
            config["server_endpoint_cls"], **endpoint_kwargs
        )
        algorithm = None
        if "algorithm_cls" in config:
            algorithm = config["algorithm_cls"]()
            assert "algorithm" not in extra_kwargs
            extra_kwargs["algorithm"] = algorithm

        return config["server_cls"](endpoint=endpoint, **(kwargs | extra_kwargs))


def get_worker_config(
    config: DistributedTrainingConfig, practitioners: None | set = None
) -> dict:
    if practitioners is None:
        practitioners = config.create_practitioners()
    else:
        config.worker_number = len(practitioners)
        for worker_id, practitioner in enumerate(
            sorted(practitioners, key=lambda p: p.id)
        ):
            assert practitioner.has_dataset(config.dc_config.dataset_name)
            practitioner.set_worker_id(worker_id)
    assert practitioners
    assert CentralizedAlgorithmFactory.has_algorithm(config.distributed_algorithm)

    context = FederatedLearningContext(worker_num=config.worker_number)
    context.create_semaphore("log_lock")
    result: dict = {
        "context": context,
    }
    result["server"] = {}
    result["server"]["constructor"] = functools.partial(
        CentralizedAlgorithmFactory.create_server,
        context=context,
        algorithm_name=config.distributed_algorithm,
        endpoint_kwargs=config.endpoint_kwargs.get("server", {}),
        kwargs={"config": config},
    )
    worker_number_per_process = config.get_worker_number_per_process()
    log_warning(
        "There are %s workers in total, and %s workers form a group",
        len(practitioners),
        worker_number_per_process,
    )
    client_config: list[list[dict]] = []
    for batch in itertools.batched(list(practitioners), n=worker_number_per_process):
        client_config.append(
            [
                {
                    "constructor": functools.partial(
                        CentralizedAlgorithmFactory.create_client,
                        context=context,
                        algorithm_name=config.distributed_algorithm,
                        endpoint_kwargs=config.endpoint_kwargs.get("worker", {})
                        | {
                            "worker_id": practitioner.worker_id,
                        },
                        kwargs={
                            "config": config,
                            "practitioner": practitioner,
                        },
                    ),
                }
                for practitioner in batch
            ]
        )
    assert client_config
    result["worker"] = client_config
    return result
