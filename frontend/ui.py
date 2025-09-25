import streamlit as st


def sidebar_controls():
    """
    Sidebar UI form for user inputs and generation settings.
    The form submission triggers the generation process.
    """
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            min-width: 20% !important;
            max-width: 50% !important;
            width: 25% !important
        }
    
        [data-testid="stSidebarUserContent"] {
            overflow-y: hidden;
        }
    
        [data-testid="stForm"] {
            border: none;
            padding: .5rem .5rem 0 .5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.header("⚙️ Generation Controls")

    with st.sidebar.form(key='generation_form'):
        product_name = st.text_input("Product Name:", value="A flower vase")
        colors = st.text_input("Colors:", value="purple and white")

        features = st.text_input(
            "Key Features (comma separated):",
            value="realistic, detailed"
        )

        diffusion = st.checkbox("Use Diffusion", value=False, help="Improves model quality (slower generation)")

        # Advanced settings
        with st.expander("🔧 Advanced Settings", expanded=False):
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

            intended_use = st.text_input("Intended Use / Context: (Optional)", value="", placeholder="e.g., for a luxury house door")

        # Deep features
        with st.expander("⚙️ Deep Features"):
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

        format_col1, format_col2 = st.columns([1, 2], gap="small")
        
        with format_col1:
            # Display the label in the first column
            st.markdown("Format:")
            
        with format_col2:
            # Display the selectbox in the second column, with its own label hidden
            chosen_format_inline = st.selectbox(
                "Choose a file format inline",
                label_visibility="collapsed",
                options=['obj', 'ply', 'stl', 'glb'],
                index=0
            )
        # The submit button for the form
        generate_button = st.form_submit_button("Generate 3D Model", type="primary")

    # Process features outside the form but within the function
    features_list = [f.strip() for f in features.replace("\n", ",").split(",") if f.strip()]

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
        "format": chosen_format,
        "generate_button": generate_button
    }
