import streamlit as st
import pyvista as pv
from stpyvista import stpyvista

pv.start_xvfb()

def show_viewer(trimesh_obj, container):
    """
    Displays a 3D trimesh object in the Streamlit PyVista viewer.
    """
    with container.container():
        pv_mesh = pv.wrap(trimesh_obj)
        plotter = pv.Plotter(window_size=[600, 600], border=False)
        plotter.add_mesh(
            pv_mesh,
            scalars=trimesh_obj.visual.vertex_colors[:, :3],
            rgb=True,
            show_edges=False
        )
        # plotter.add_mesh(pv_mesh, show_edges=True)
        plotter.view_isometric()
        plotter.background_color = "black"
        stpyvista(plotter, key="main_viewer")

def show_download_button(file_path, container, format):
    """
    Displays a download button for the generated OBJ file.
    """
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    with container.container():
        st.download_button(
            label=f"⬇️ Download .{format} file",
            data=file_bytes,
            file_name=file_path.split("/")[-1],
            mime="application/octet-stream"
        )
