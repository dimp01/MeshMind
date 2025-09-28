import hashlib
import os
from functools import lru_cache
from typing import Dict, Optional

import requests
import torch
import yaml
from filelock import FileLock
from tqdm.auto import tqdm

MODEL_PATHS = {
    "transmitter": "https://drive.google.com/uc?export=download&id=1dTMLDq-ktjoBDJaSKLMRbR-HVVck932q",
    "decoder": "https://openaipublic.azureedge.net/main/shap-e/vector_decoder.pt",
    "text300M": "https://drive.google.com/uc?export=download&id=1CIpHfXVaHRm9TYfbSNUJfx-sS_QVLnwC",
    "image300M": "https://drive.google.com/uc?export=download&id=17a3do_emwJZp-E2DWMybQbvJa1rLM7Jh",
}

CONFIG_PATHS = {
    "transmitter": "https://drive.google.com/uc?export=download&id=1wH1nGRSsmg8U72RaPXAtS4Fl-IVbnYcL",
    "decoder": "https://openaipublic.azureedge.net/main/shap-e/vector_decoder_config.yaml",
    "text300M": "https://drive.google.com/uc?export=download&id=1i3lTsjazz9IkE9Jb7NH8WsiIMFCpzKyU",
    "image300M": "https://drive.google.com/uc?export=download&id=15alU5pN1OYEVL59s8SYT0TXVaZvlzpVR",
    "diffusion": "https://drive.google.com/uc?export=download&id=1d8MU3eTM9MHTrbabJ3l-qzJyZzwR5ztL",
}


@lru_cache()
def default_cache_dir() -> str:
    return os.path.join(os.path.abspath(os.getcwd()), "model_cache")


def fetch_file_cached(
    url: str, progress: bool = True, cache_dir: Optional[str] = None, chunk_size: int = 4096
) -> str:
    """
    Download the file at the given URL into a local file and return the path.
    If cache_dir is specified, it will be used to download the files.
    Otherwise, default_cache_dir() is used.
    """
    file_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()

    if cache_dir is None:
        cache_dir = default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, file_hash)
    if os.path.exists(local_path):
        return local_path

    response = requests.get(url, stream=True)
    response.raise_for_status()
    size = int(response.headers.get("content-length", "0"))
    with FileLock(local_path + ".lock"):
        if progress:
            pbar = tqdm(total=size, unit="iB", unit_scale=True)
        tmp_path = local_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if progress:
                    pbar.update(len(chunk))
                f.write(chunk)
        os.rename(tmp_path, local_path)
        if progress:
            pbar.close()
        return local_path


def load_config(
    config_name: str,
    progress: bool = False,
    cache_dir: Optional[str] = None,
    chunk_size: int = 4096,
):
    if config_name not in CONFIG_PATHS:
        raise ValueError(
            f"Unknown config name {config_name}. Known names are: {CONFIG_PATHS.keys()}."
        )
    path = fetch_file_cached(
        CONFIG_PATHS[config_name], progress=progress, cache_dir=cache_dir, chunk_size=chunk_size
    )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(
    checkpoint_name: str,
    device: torch.device,
    progress: bool = True,
    cache_dir: Optional[str] = None,
    chunk_size: int = 4096,
) -> Dict[str, torch.Tensor]:
    if checkpoint_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown checkpoint name {checkpoint_name}. Known names are: {MODEL_PATHS.keys()}."
        )
    path = fetch_file_cached(
        MODEL_PATHS[checkpoint_name], progress=progress, cache_dir=cache_dir, chunk_size=chunk_size
    )
    return torch.load(path, map_location=device)


def load_model(
    model_name: str,
    device: torch.device,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    from .configs import model_from_config

    model = model_from_config(load_config(model_name, **kwargs), device=device)
    model.load_state_dict(load_checkpoint(model_name, device=device, **kwargs))
    model.eval()
    return model
