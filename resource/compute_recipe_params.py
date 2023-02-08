import dataiku
import torch


def compute_device(payload, config, plugin_config, inputs):
    """Compute a list of PyTorch devices for the "device" param"""
    choices = [
        {"value": "auto", "label": "Auto"},
        {"value": "cpu", "label": "CPU (disable CUDA)"},
    ]

    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        value = f"cuda:{i}"
        label = f"{value} ({device_name})"
        choices.append({"value": value, "label": label})

    return {"choices": choices}


def compute_base_image_path(payload, config, plugin_config, inputs):
    """Compute a list of files for the "base_image_path" param"""
    base_folder_input_ = next(
        input_ for input_ in inputs if input_["role"] == "base_image_folder"
    )
    base_folder_name = base_folder_input_["fullName"]
    base_folder = dataiku.Folder(base_folder_name)

    paths = sorted(base_folder.list_paths_in_partition())
    choices = [{"value": path, "label": path} for path in paths]

    return {"choices": choices}


_PARAMETER_COMPUTE_FUNCTIONS = {
    "device": compute_device,
    "base_image_path": compute_base_image_path,
}


def do(payload, config, plugin_config, inputs):
    """Compute the param for the given payload"""
    parameter_name = payload["parameterName"]
    compute_func = _PARAMETER_COMPUTE_FUNCTIONS[parameter_name]
    return compute_func(payload, config, plugin_config, inputs)
