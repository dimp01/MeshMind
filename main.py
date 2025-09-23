import os
import streamlit as st
from datetime import datetime
from backend.config import device
from backend.gemini_prompt import (
    text_model_prompt,
    diffusion_model_prompt,
    gen_file_name,
)
from backend.mesh_utils import build_trimesh, save_mesh_obj
from backend.file_utils import ensure_output_dir, safe_join
from backend.generate import GenerateModel
from backend.cleaner import clear_memory

from frontend.ui import sidebar_controls
from frontend.viewer import show_viewer, show_download_button
from frontend.history import show_history

# --- Clean Startup --- 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
clear_memory(verbose=True)

# --- Streamlit Page Setup ---
st.set_page_config(page_title="AI Product Designer Pro", page_icon="🤖", layout="wide")
st.title("🤖 AI Product Designer Pro")

# --- Initialize session state ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar controls ---
controls = sidebar_controls()

# --- Tabs for UI ---
viewer_tab, history_tab = st.tabs(["🖼️ 3D Viewer", "📜 Generation History"])

# ---------------------------
# Viewer Tab
# ---------------------------
with viewer_tab:
    st.subheader("Live 3D Model Viewer")
    viewer_panel = st.empty()
    download_panel = st.empty()

    if controls["generate_button"]:
        # Build prompt using Gemini
        with st.status("✨ Evaluating your request....", expanded=True) as status:
            if controls["is_diffusion"]:
                generate_prompt = diffusion_model_prompt
            else:
                generate_prompt = text_model_prompt
            st.write("🔄 Refining words… turning chaos into clarity")
            prompt = generate_prompt(
                product_type=controls["product_name"],
                dimensions=controls["dimensions"],
                features=controls["features"],
                materials=controls["material"],
                style=controls["style"],
                intended_use=controls["intended_use"],
            )
            st.write(" - A prompt was generated.")

            if not prompt:
                st.warning("⚠️ Gemini did not return a valid prompt. Try again.")
            else:
                st.write("🧠 Generating 3D model... This may take a few minutes.")
                try:
                    generate = GenerateModel(
                        prompt,
                        guidance_scale=controls["guidance_scale"],
                        num_inference_steps=controls["steps"],
                        frame_size=controls["frame_size"],
                        output_type="mesh",
                        return_dict=True,
                    )
                    if controls["is_diffusion"]:
                        decoder_output = generate.diffusion()
                    else:
                        decoder_output = generate.text()
    
                    # Build mesh
                    trimesh_obj = build_trimesh(decoder_output)
    
                    # File management
                    output_dir = ensure_output_dir()
                    file_name = gen_file_name(prompt)
                    file_path = safe_join(output_dir, file_name)
                    save_mesh_obj(decoder_output, file_path)
                    st.write(" - 3D model was generated.")
    
                    # Update session history
                    st.session_state.history.append(
                        {
                            "prompt": prompt,
                            "file_path": file_path,
                            "timestamp": datetime.now().strftime("%I:%M:%S %p"),
                        }
                    )
    
                    status.update(
                        label="✅ Generation complete!", state="complete", expanded=False
                    )
    
                    # Display in viewer
                    show_viewer(trimesh_obj, viewer_panel)
    
                    # Download button
                    show_download_button(file_path, download_panel)
                    clear_memory()

                except Exception as e:
                    clear_memory()
                    st.error(f"❌ An error occurred while generating the model: {e}")

# ---------------------------
# History Tab
# ---------------------------
with history_tab:
    show_history(viewer_panel, download_panel)
