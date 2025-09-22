import torch
import gc
import streamlit as st

def clear_memory(verbose=True):
    """
    Safely clears GPU VRAM and CPU RAM used by PyTorch tensors.
    Does not close the Streamlit server.
    """
    for obj in dir():
        if isinstance(obj, torch.Tensor):
            del obj

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # frees inter-process memory

    if verbose:
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024**2
            st.toast(f"GPU VRAM allocated: {vram:.2f} MB", icon="📈")
        else:
            st.warning("⚠ Cleared memory. GPU not available.")
