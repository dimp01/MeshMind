import os
import torch
import trimesh
from pathlib import Path
from PIL import Image
from .load_objects import load_object
from transformers import BlipProcessor, BlipForConditionalGeneration
from pyvirtualdisplay import Display
import json

os.environ["PYOPENGL_PLATFORM"] = "egl"
display = Display(visible=0, size=(800, 600), backend="xvfb")
display.start()

OUT_CAPS = Path(__file__).parent.parent / "captions"
OUT_CAPS.mkdir(exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def render_views(mesh_path, image_size=224, num_views=5):
    mesh = trimesh.load(mesh_path, force='mesh')
    scene = mesh.scene()
    images = []

    for i in range(num_views):
        img = scene.save_image(resolution=(800,600), visible=True)
        if img is not None:
            images.append(Image.open(trimesh.util.wrap_as_stream(img)).convert("RGB"))
    return images

def caption_images(images):
    captions = []
    for im in images:
        inputs = processor(images=im, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_length=40)
        txt = processor.decode(out[0], skip_special_tokens=True)
        captions.append(txt)
    return captions

def load_captions(limit=20):
    DATA_DIR = load_object(limit)
    ENC_OUT = Path(__file__).parent.parent / "encoded"
    ENC_OUT.mkdir(exist_ok=True)

    # Build object dict: {name -> path}
    objects = {f.stem: f for f in DATA_DIR.iterdir() if f.suffix in (".obj", ".ply", ".glb", ".gltf")}

    for mesh_name, mesh_path in objects.items():
        try:
            views = render_views(mesh_path, num_views=10)
            if len(views) == 0:
                print("No renders for", mesh_name, "— skipping")
                continue

            captions = caption_images(views)
            agg = " ".join(dict.fromkeys(captions))  # remove duplicates

            (OUT_CAPS / (mesh_name + ".txt")).write_text(agg)
            print("Captured captions for", mesh_name, "->", agg[:140])

            # save metadata
            meta = {"mesh": str(mesh_path), "caption": agg}
            (ENC_OUT / (mesh_name + ".json")).write_text(json.dumps(meta))
        except Exception as e:
            print("Caption failed for", mesh_name, e)

    print("Created small metadata training set in", ENC_OUT)
    return ENC_OUT
