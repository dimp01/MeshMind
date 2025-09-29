import streamlit as st
import random

def sidebar_controls():
    """
    Sidebar UI form for user inputs and generation settings.
    Includes reset-to-defaults and random seed option.
    """
    # --- CSS for responsive sidebar ---
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            min-width: 20% !important;
            max-width: 50% !important;
            width: 25% !important;
        }
        [data-testid="stSidebarUserContent"] {
            overflow-y: hidden;
        }
        [data-testid="stForm"] {
            border: none;
            padding: .5rem .5rem 0 .5rem;
        }
        @media (max-width: 768px) {
            section[data-testid="stSidebar"] { width: 60% !important; min-width: 60% !important; max-width: 80% !important; }
            [data-testid="stForm"] { padding: 0.5rem; }
        }
        @media (max-width: 480px) {
            section[data-testid="stSidebar"] { width: 80% !important; min-width: 80% !important; max-width: 95% !important; }
        }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.header("⚙️ Generation Controls")

    # --- Default values ---
    DEFAULTS = {
        "product_name": "A flower vase",
        "colors": "purple and white",
        "features": "realistic, detailed",
        "diffusion": False,
        "length": 30.0,
        "width": 12.0,
        "height": 15.0,
        "form_factor": "Ergonomic",
        "material": "Plastic",
        "style_keywords": ["Sleek", "Compact"],
        "intended_use": "",
        "randomize_seed": True,
        "seed": 1758743251,
        "guidance_scale": 15.0,
        "num_inference_steps": 65,
        "render_frame_size": 160,
        "format": "obj"
    }

    # Initialize session state for reset functionality
    for key, val in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Reset button
    if st.sidebar.button("Reset to Defaults"):
        for key, val in DEFAULTS.items():
            st.session_state[key] = val

    with st.sidebar.form(key='generation_form'):
        # Basic Inputs
        product_name = st.text_input("Product Name:", value=st.session_state.product_name)
        colors = st.text_input("Colors:", value=st.session_state.colors)
        features = st.text_input("Key Features (comma separated):", value=st.session_state.features)
        diffusion = st.checkbox("Use Diffusion", value=st.session_state.diffusion)

        # Advanced settings
        with st.expander("🔧 Advanced Settings", expanded=False):
            length = st.number_input("Length (cm):", min_value=1.0, max_value=200.0, value=st.session_state.length)
            width = st.number_input("Width (cm):", min_value=1.0, max_value=100.0, value=st.session_state.width)
            height = st.number_input("Height (cm):", min_value=1.0, max_value=100.0, value=st.session_state.height)

            form_factor = st.radio(
                "Form Factor:",
                options=["Cylindrical", "Rectangular", "Ergonomic", "Custom Shape"],
                index=["Cylindrical", "Rectangular", "Ergonomic", "Custom Shape"].index(st.session_state.form_factor)
            )

            material = st.selectbox(
                "Material / Finish:",
                options=["Plastic", "Metal", "Wood", "Glass", "Composite", "Mixed"],
                index=["Plastic", "Metal", "Wood", "Glass", "Composite", "Mixed"].index(st.session_state.material)
            )

            style_keywords = st.multiselect(
                "Style / Design Keywords:",
                options=["Sleek", "Minimalist", "Futuristic", "Industrial", "Vintage", "Compact", "Ergonomic"],
                default=st.session_state.style_keywords
            )

            intended_use = st.text_input("Intended Use / Context:", value=st.session_state.intended_use)

        # Deep Features with randomize option
        with st.expander("⚙️ Deep Features"):
            col1, col2 = st.columns([1, 2], gap="small")
            with col1:
                randomize_seed = st.checkbox("Randomize Seed", value=st.session_state.randomize_seed)
            with col2:
                if randomize_seed:
                    seeding = random.randint(1, 2**32 - 1)
                    st.markdown(f"Seed: {seeding}")
                else:
                    seeding = st.number_input(
                        "Seed",
                        min_value=1,
                        max_value=2**32 - 1,
                        value=st.session_state.seed,
                        step=1
                    )

            guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=30.0, value=st.session_state.guidance_scale)
            num_inference_steps = st.slider("Inference Steps", min_value=10, max_value=100, value=st.session_state.num_inference_steps)
            render_frame_size = st.slider("Render Frame Size", min_value=64, max_value=256, value=st.session_state.render_frame_size)

        # Output format
        format_col1, format_col2 = st.columns([1, 1], gap="small")
        with format_col1:
            st.markdown("Output Format:")
        with format_col2:
            chosen_format = st.selectbox("Choose a file format", options=['obj', 'ply', 'stl', 'glb'],
                                         index=['obj', 'ply', 'stl', 'glb'].index(st.session_state.format),
                                         label_visibility="collapsed")

        generate_button = st.form_submit_button("Generate 3D Model", type="primary")

    # Parse features list
    features_list = [f.strip() for f in features.replace("\n", ",").split(",") if f.strip()]

    # Return all values
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
        "seed": seeding,
        "randomize_seed": randomize_seed,
        "colors": colors,
        "format": chosen_format,
        "generate_button": generate_button
    }
