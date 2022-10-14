import torch


def do(payload, config, plugin_config, inputs):
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
