from collections.abc import Callable
from typing import Any, Self


class AlgorithmRepository:
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
        cls.config[algorithm_name] = {
            "client_cls": client_cls,
            "server_cls": server_cls,
        }
        if client_endpoint_cls is not None:
            cls.config[algorithm_name]["client_endpoint_cls"] = client_endpoint_cls
        if server_endpoint_cls is not None:
            cls.config[algorithm_name]["server_endpoint_cls"] = server_endpoint_cls
        if algorithm_cls is not None:
            cls.config[algorithm_name]["algorithm_cls"] = algorithm_cls

    @classmethod
    def has_algorithm(cls, algorithm_name: str) -> bool:
        return algorithm_name in cls.config

    @classmethod
    def create_client(
        cls: type[Self],
        algorithm_name: str,
        kwargs: dict,
        endpoint_kwargs: dict,
        **extra_kwargs: Any,
    ) -> None:
        config = cls.config[algorithm_name]
        if "client_endpoint_cls" in config:
            endpoint_kwargs["endpoint_cls"] = config["client_endpoint_cls"]
        context = extra_kwargs["context"]
        endpoint = context.create_client_endpoint(**endpoint_kwargs)
        return config["client_cls"](endpoint=endpoint, **kwargs, **extra_kwargs)

    @classmethod
    def create_server(
        cls,
        algorithm_name: str,
        kwargs: dict,
        endpoint_kwargs: dict,
        **extra_kwargs: Any,
    ) -> None:
        config = cls.config[algorithm_name]
        context = extra_kwargs["context"]
        if "server_endpoint_cls" in config:
            endpoint_kwargs["endpoint_cls"] = config["server_endpoint_cls"]
        endpoint = context.create_server_endpoint(**endpoint_kwargs)
        algorithm = None
        if "algorithm_cls" in config:
            algorithm = config["algorithm_cls"]()
            assert "algorithm" not in extra_kwargs
            extra_kwargs["algorithm"] = algorithm

        return config["server_cls"](endpoint=endpoint, **kwargs, **extra_kwargs)
