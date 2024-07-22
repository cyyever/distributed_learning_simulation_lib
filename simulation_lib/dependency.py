import importlib
import os


def import_dependencies() -> dict:
    result = {}
    libs = ["cyy_torch_graph", "cyy_torch_text", "cyy_torch_vision"]
    if "dataset_type" in os.environ:
        match os.environ["dataset_type"].lower():
            case "graph":
                libs = ["cyy_torch_graph"]
            case "vision":
                libs = ["cyy_torch_vision"]
            case "text":
                libs = ["cyy_torch_text"]
            case _:
                raise NotImplementedError(os.environ["dataset_type"])

    for dependency in libs:
        try:
            importlib.import_module(dependency)
            result[dependency] = True
        except BaseException:
            pass
    return result


import_results = import_dependencies()
