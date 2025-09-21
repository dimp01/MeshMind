import streamlit as st
from backend.config import device
from backend.shap_e_pipeline import load_shap_e_pipeline
from backend.gemini_prompt import generate_shap_e_prompt, gen_file_name
from backend.mesh_utils import build_trimesh, save_mesh_obj
from backend.file_utils import ensure_output_dir, safe_join

from frontend.ui import sidebar_controls
from frontend.viewer import show_viewer, show_download_button
from frontend.history import show_history

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="AI Product Designer Pro",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 AI Product Designer Pro")

# --- Initialize session state ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar controls ---
controls = sidebar_controls()

# --- Load Shap-E pipeline (cached) ---
pipe = load_shap_e_pipeline(device)

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
        prompt = generate_shap_e_prompt(
            product_type=controls["product_name"],
            dimensions=controls["dimensions"],
            features=controls["features"],
            materials=controls["material"],
            style=controls["style"],
            intended_use=controls["intended_use"]
        )

        if not prompt:
            st.warning("⚠️ Gemini did not return a valid prompt. Try again.")
        else:
            with st.spinner("🧠 Generating 3D model... This may take a few minutes."):
                try:
                    # Run Shap-E
                    outputs = pipe(
                        prompt,
                        guidance_scale=controls["guidance_scale"],
                        num_inference_steps=controls["steps"],
                        frame_size=controls["frame_size"],
                        output_type="mesh",
                        return_dict=True,
                    )
                    decoder_output = outputs.images[0]

                    # Build mesh
                    trimesh_obj = build_trimesh(decoder_output)

                    # File management
                    output_dir = ensure_output_dir()
                    file_name = gen_file_name(prompt)
                    file_path = safe_join(output_dir, file_name)
                    save_mesh_obj(decoder_output, file_path)

                    # Update session history
                    st.session_state.history.append({
                        "prompt": prompt,
                        "file_path": file_path,
                        "timestamp": datetime.now().strftime("%I:%M:%S %p")
                    })

                    # Display in viewer
                    show_viewer(trimesh_obj, viewer_panel)

                    # Download button
                    show_download_button(file_path, download_panel)

                except Exception as e:
                    st.error(f"❌ An error occurred while generating the model: {e}")

# ---------------------------
# History Tab
# ---------------------------
with history_tab:
    show_history(viewer_panel, download_panel)
