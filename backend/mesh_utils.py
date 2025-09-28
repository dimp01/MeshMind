import numpy as np
import trimesh
import pymeshfix


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

def save_mesh_as(decoder_output, file_path):
    """
    Saves a Shap-E MeshDecoderOutput to .obj file.
    """
    mesh = build_trimesh(decoder_output)
    repair_mesh(mesh, file_path)
    return file_path


def repair_mesh(mesh, output_path, verbose=True):
    """
    Repair a mesh using PyMeshFix.
    - Fills holes
    - Fixes non-manifold edges
    - Outputs repaired mesh as OBJ (default) or same as input extension
    
    Args:
        input_path (str): Path to input mesh (.obj, .stl, etc.)
        output_path (str): Path to save repaired mesh. Defaults to input_path + '_repaired.obj'
        verbose (bool): Print stats before and after
    """
    if mesh.is_empty:
        raise ValueError("Mesh is empty or invalid.")

    if verbose:
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Extract vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces

    # Repair using PyMeshFix
    meshfix = pymeshfix.MeshFix(vertices, faces)
    meshfix.repair(verbose=verbose)  # fixes holes, non-manifold edges

    # Reconstruct repaired mesh
    repaired_mesh = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f, process=False)

    if verbose:
        print(f"Repaired mesh: {len(repaired_mesh.vertices)} vertices, {len(repaired_mesh.faces)} faces")

    # Save repaired mesh
    repaired_mesh.export(output_path)
    if verbose:
        print(f"Saved repaired mesh to: {output_path}")
