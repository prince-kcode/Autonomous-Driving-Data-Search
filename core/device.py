import torch

def get_device() -> torch.device:
    """
    Returns optimal available device: CUDA GPU > Apple MPS > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_device_info() -> dict:
    dev = get_device()
    info = {
        "device": str(dev),
        "type": dev.type
    }
    if dev.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["memory_allocated_mb"] = round(torch.cuda.memory_allocated(0) / 1024**2, 2)
        info["cuda_version"] = torch.version.cuda
    return info

