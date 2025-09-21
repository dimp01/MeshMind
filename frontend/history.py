import os
import streamlit as st
import numpy as np
import trimesh
import pyvista as pv
from stpyvista import stpyvista

def show_history(viewer_panel, download_panel):
    """
    Displays session history of generated 3D models.
    Allows reloading old models into the viewer.
    """

    if not st.session_state.history:
        st.info("Your generated models will appear here (session only).")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            st.markdown("---")
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**File:** `{item['file_path']}`")
                st.caption(f"Generated at: {item['timestamp']}")

                # Download button for saved .obj
                with open(item['file_path'], "rb") as f:
                    hist_file_bytes = f.read()

                st.download_button(
                    label="Download .obj",
                    data=hist_file_bytes,
                    file_name=os.path.basename(item['file_path']),
                    key=f"hist_dl_{i}"
                )

            with col2:
                if st.button("Load in Viewer", key=f"hist_view_{i}"):
                    # Reconstruct meshes from stored verts/faces
                    verts_np = np.asarray(item["verts"])
                    faces_np = np.asarray(item["faces"]).astype(np.int64)

                    reloaded_trimesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
                    pv_mesh = pv.wrap(reloaded_trimesh)

                    with viewer_panel.container():
                        st.info(f"`{item['prompt']}`")
                        pv_mesh = pv.wrap(reloaded_trimesh)
                        plotter = pv.Plotter(window_size=[600, 600], border=False)
                        plotter.add_mesh(pv_mesh, show_edges=True)
                        plotter.view_isometric()
                        plotter.background_color = "black"
                        stpyvista(plotter, key="main_viewer")
                        # render_mesh_to_png_and_show(pv_mesh, key=f"reloaded_viewer_{i}")

                    # update download button for the reloaded model
                    with open(item['file_path'], "rb") as f:
                        reloaded_bytes = f.read()

                    with download_panel.container():
                        st.download_button(
                            label="⬇️ Download .obj file",
                            data=reloaded_bytes,
                            file_name=os.path.basename(item['file_path']),
                            mime="application/octet-stream"
                        )
                    st.success("Model loaded into the viewer!")
