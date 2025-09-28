import hashlib
import gdown
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

URL_HASHES = {
    "https://drive.google.com/uc?export=download&id=1dTMLDq-ktjoBDJaSKLMRbR-HVVck932q": "af02a0b85a8abdfb3919584b63c540ba175f6ad4790f574a7fef4617e5acdc3b",
    "https://openaipublic.azureedge.net/main/shap-e/vector_decoder.pt": "d7e7ebbfe3780499ae89b2da5e7c1354012dba5a6abfe295bed42f25c3be1b98",
    "https://drive.google.com/uc?export=download&id=1CIpHfXVaHRm9TYfbSNUJfx-sS_QVLnwC": "e6b4fa599a7b3c3b16c222d5f5fe56f9db9289ff0b6575fbe5c11bc97106aad4",
    "https://drive.google.com/uc?export=download&id=17a3do_emwJZp-E2DWMybQbvJa1rLM7Jh": "cb8072c64bbbcf6910488814d212227de5db291780d4ea99c6152f9346cf12aa",
    "https://drive.google.com/uc?export=download&id=1wH1nGRSsmg8U72RaPXAtS4Fl-IVbnYcL": "ffe1bcb405104a37d9408391182ab118a4ef313c391e07689684f1f62071605e",
    "https://openaipublic.azureedge.net/main/shap-e/vector_decoder_config.yaml": "e6d373649f8e24d85925f4674b9ac41c57aba5f60e42cde6d10f87381326365c",
    "https://drive.google.com/uc?export=download&id=1i3lTsjazz9IkE9Jb7NH8WsiIMFCpzKyU": "f290beeea3d3e9ff15db01bde5382b6e549e463060c0744f89c049505be246c1",
    "https://drive.google.com/uc?export=download&id=15alU5pN1OYEVL59s8SYT0TXVaZvlzpVR": "4e0745605a533c543c72add803a78d233e2a6401e0abfa0cad58afb4d74ad0b0",
    "https://drive.google.com/uc?export=download&id=1d8MU3eTM9MHTrbabJ3l-qzJyZzwR5ztL": "efcb2cd7ee545b2d27223979d41857802448143990572a42645cd09c2942ed57",
}


@lru_cache()
def default_cache_dir() -> str:
    return os.path.join(os.path.abspath(os.getcwd()), "model_cache")


def fetch_file_cached(
    url: str, progress: bool = True, cache_dir: Optional[str] = None, chunk_size: int = 8192
) -> str:
    expected_hash = URL_HASHES.get(url)
    if expected_hash is None:
        raise ValueError(f"No expected hash found for URL: {url}")

    file_info = gdown.get_url_info(url)
    filename = file_info['name']
    size = int(file_info.get('size', 0))

    if cache_dir is None:
        cache_dir = default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)

    local_path = os.path.join(cache_dir, filename)
    if os.path.exists(local_path):
        check_hash(local_path, expected_hash)
        return local_path

    tmp_path = local_path + ".tmp"
    with FileLock(local_path + ".lock"):
        if "drive.google.com" in url:
            gdown.download(url, output=tmp_path, quiet=not progress)
        else:
            pbar = tqdm(total=size, unit="iB", unit_scale=True, unit_divisor=1024) if progress else None
            try:
                with requests.get(url, stream=True) as r, open(tmp_path, "wb") as f:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            if progress:
                                pbar.update(len(chunk))
            finally:
                if progress and pbar:
                    pbar.close()

        os.rename(tmp_path, local_path)

    check_hash(local_path, expected_hash)
    return local_path


def check_hash(path: str, expected_hash: str):
    actual_hash = hash_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"The file {path} should have hash {expected_hash} but has {actual_hash}. "
            "Try deleting it and running this call again."
        )


def hash_file(path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            data = file.read(4096)
            if not len(data):
                break
            sha256_hash.update(data)
    return sha256_hash.hexdigest()


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
