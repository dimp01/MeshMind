import numpy as np
import trimesh
from diffusers.utils import export_to_obj

def build_trimesh(decoder_output):
    """
    Converts Shap-E MeshDecoderOutput into a trimesh.Trimesh object.
    Handles tensor → numpy conversions safely.
    """
    verts = getattr(decoder_output, "verts", None)
    faces = getattr(decoder_output, "faces", None)

    if verts is None or faces is None:
        raise ValueError("Decoder output missing verts/faces attributes.")

    # Convert to numpy
    verts_np = verts.cpu().numpy() if hasattr(verts, "cpu") else np.asarray(verts)
    faces_np = faces.cpu().numpy() if hasattr(faces, "cpu") else np.asarray(faces)

    # Handle Shap-E [N,4] face format
    if faces_np.shape[1] == 4:
        faces_np = faces_np[:, 1:]

    return trimesh.Trimesh(vertices=verts_np, faces=faces_np.astype(np.int64), process=False)

def save_mesh_obj(decoder_output, file_path):
    """
    Saves a Shap-E MeshDecoderOutput to .obj file.
    """
    export_to_obj(decoder_output, file_path)
    return file_path
