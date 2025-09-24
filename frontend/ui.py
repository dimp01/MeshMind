import streamlit as st

def sidebar_controls():
    """
    Sidebar UI for user inputs and generation settings.
    Returns: dict of all control values
    """
    st.sidebar.header("⚙️ Generation Controls")

    product_name = st.sidebar.text_input("Product Name:", value="A flower vase")
    colors = st.sidebar.text_input("Colors:", value="purple and white")

    features = st.sidebar.text_area(
        "Key Features (comma separated):",
        value="realistic, detailed"
    )

    diffusion = st.sidebar.checkbox("Use Diffusion", value=False, help="Improves model quality (slower the process of generation)")
    features_list = [f.strip() for f in features.replace("\n", ",").split(",") if f.strip()]

    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings", expanded=False):
        length = st.number_input("Length (cm):", min_value=1.0, max_value=200.0, value=30.0, step=0.1)
        width = st.number_input("Width (cm):", min_value=1.0, max_value=100.0, value=12.0, step=0.1)
        height = st.number_input("Height (cm):", min_value=1.0, max_value=100.0, value=15.0, step=0.1)

        form_factor = st.radio(
            "Form Factor:",
            options=["Cylindrical", "Rectangular", "Ergonomic", "Custom Shape"],
            index=0
        )

        material = st.selectbox(
            "Material / Finish:",
            options=["Plastic", "Metal", "Wood", "Glass", "Composite", "Mixed"]
        )

        style_keywords = st.multiselect(
            "Style / Design Keywords:",
            options=["Sleek", "Minimalist", "Futuristic", "Industrial", "Vintage", "Compact", "Ergonomic"],
            default=["Sleek", "Compact"]
        )

        intended_use = st.text_input("Intended Use / Context: (Optional)", value="", placeholder="eg: for luxury house door")

    # Deep features
    with st.sidebar.expander("⚙️ Deep Features"):
        guidance_scale = st.slider(
            "Guidance Scale (Prompt Adherence)",
            min_value=1.0, max_value=30.0, value=15.5, step=0.5,
            help="Higher values = stricter adherence to prompt"
        )

        num_inference_steps = st.slider(
            "Inference Steps (Detail vs. Speed)",
            min_value=10, max_value=100, value=64, step=1,
            help="More steps = more detail but slower"
        )

        render_frame_size = st.slider(
            "Render Frame Size",
            min_value=64, max_value=256, value=160, step=32,
            help="Higher values = more detail but more GPU memory"
        )

    generate_button = st.sidebar.button("Generate 3D Model", type="primary")

    # Return all control values as a dict
    return {
        "product_name": product_name,
        "features": features_list,
        "dimensions": f"{length} x {width} x {height} cm",
        "form_factor": form_factor,
        "material": material,
        "style": ", ".join(style_keywords),
        "intended_use": intended_use,
        "guidance_scale": guidance_scale,
        "steps": num_inference_steps,
        "frame_size": render_frame_size,
        "is_diffusion": diffusion,
        "colors": colors,
        "generate_button": generate_button
    }
