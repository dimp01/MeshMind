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
    st.caption("Session History")

    if not st.session_state.history:
        st.info("Your generated models will appear here (session only).")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            st.markdown("---") if i!=0 else 0
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                pv_mesh = pv.wrap(reloaded_trimesh)
                plotter = pv.Plotter(off_screen=True, window_size=[150, 120])
                plotter.add_mesh(pv_mesh, show_edges=True)
                plotter.view_isometric()
                plotter.background_color = "black"
                img_array = plotter.screenshot(return_img=True)
                plotter.close()
                st.image(img_array)

            with col2:
                st.write(f"**File:** `{item['file_path']}`")
                st.caption(f"Generated at: {item['timestamp']}")

                # Download button for saved .obj
                with open(item['file_path'], "rb") as f:
                    hist_file_bytes = f.read()

                st.download_button(
                    label=f"Download .{item["format"]}",
                    data=hist_file_bytes,
                    file_name=os.path.basename(item['file_path']),
                    key=f"hist_dl_{i}"
                )

            with col3:
                if st.button("Load in Viewer", key=f"hist_view_{i}"):
                    reloaded_trimesh = trimesh.load(item["file_path"])
                    pv_mesh = pv.wrap(reloaded_trimesh)

                    with viewer_panel.container():
                        st.info(f"`{item['prompt']}`")
                        pv_mesh = pv.wrap(reloaded_trimesh)
                        plotter = pv.Plotter(window_size=[600, 600], border=False)
                        plotter.add_mesh(pv_mesh, show_edges=True)
                        plotter.view_isometric()
                        plotter.background_color = "black"
                        stpyvista(plotter, key="main_viewer")

                    # update download button for the reloaded model
                    with open(item['file_path'], "rb") as f:
                        reloaded_bytes = f.read()

                    with download_panel.container():
                        st.download_button(
                            label=f"⬇️ Download .{item["format"]} file",
                            data=reloaded_bytes,
                            file_name=os.path.basename(item['file_path']),
                            mime="application/octet-stream"
                        )
                    st.success("Model loaded into the viewer!")
