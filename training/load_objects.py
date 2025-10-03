from objaverse import load_uids, load_objects
from pathlib import Path
import random, os, shutil

OUT = Path.home() / ".objaverse/hf-objaverse-v1/glbs/"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

def load_object(n_samples=10):
    uids = load_uids()
    if len(uids) == 0:
        print("No UIDs available from Objaverse")
        return OUT

    sample_ids = random.sample(uids, min(n_samples, len(uids)))
    objects = load_objects(uids=sample_ids, download_processes=os.cpu_count())

    if len(objects) == 0:
        print("No objects downloaded — falling back to Shap-E example assets")
        try:
            sample_dir = Path(__file__).parent.parent / "sample_data"
            if sample_dir.exists():
                for f in sample_dir.rglob("*"):
                    if f.suffix in (".obj", ".ply", ".glb", ".gltf"):
                        shutil.copy(f, OUT / f.name)
                print("Copied sample meshes to", OUT)
            else:
                print("No sample dir; please supply OBJ/GLB files")
        except Exception as e:
            print("Sample fallback failed:", e)

    print("Objects available:", len(list(OUT.glob("*"))))
    return OUT
