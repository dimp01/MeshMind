import os

def ensure_output_dir(path="3d_models_output"):
    """
    Ensures the output directory exists.
    """
    os.makedirs(path, exist_ok=True)
    return path

def safe_join(output_dir, filename):
    """
    Combines directory + filename into a safe file path.
    """
    return os.path.join(output_dir, filename)
