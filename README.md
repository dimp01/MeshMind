# MeshMind 🤖 AI-Powered 3D Product Designer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dimp01/MeshMind/blob/main/run_code.ipynb)

MeshMind is an academic prototype for **AI-assisted 3D product design**.  
It allows users to create **basic but customizable 3D prototypes** from text and image prompts.

Designed for rapid prototyping and academic learning, MeshMind enables users to:

- Specify product dimensions, materials, and colors  
- Generate lightweight 3D meshes for experimentation  
- Preview models in a live 3D viewer  
- Download `.obj` or `.stl` files for CAD/3D printing  
- Maintain a session history to revisit and reload designs

---

## 🚩 Problem Statements

Existing AI-based 3D design systems face several issues:

1. **Low or inconsistent mesh quality** – Outputs often need heavy cleanup.  
2. **Hidden costs or subscriptions** – Credit systems and unclear pricing frustrate users.  
3. **Limited user control** – Lack of precision in geometry, dimensions, or textures.  
4. **Time wasted on regenerations** – Models must be regenerated even if already created.  
5. **Usability challenges** – Complex UI, unclear workflows, limited free testing.  
6. **Overpromising results** – Generated meshes often don’t match expectations.  
7. **Post-processing requirement** – Users need to fix normals, holes, or UVs.  
8. **Transparency issues** – Lack of clarity around outputs, terms, and limitations.

---

## ✔️ MeshMind Solutions

MeshMind was built to **address these challenges in an academic context**:

1. **Improved Usability**  
   - Minimal Streamlit UI with clear input boxes, dropdowns, and previews to make the tool approachable for beginners.

2. **Free & Transparent**  
   - 100% free for academic use — no subscriptions, no credits, and no hidden costs.

3. **Structured Prompting**  
   - Provide templates and fields for product type, dimensions, material, color, and reference images to reduce ambiguous prompts.

4. **Model Reuse (Session History)**  
   - Session history lets users re-preview and re-download past models without re-running generation, saving time and compute.

5. **Preview Before Download**  
   - Interactive 3D viewer (rotate/zoom/wireframe/material toggle) so users inspect models before exporting.

6. **Lightweight Post-Processing**  
   - Automatic cleanup (fix normals, remove non-manifold geometry, fill small holes, smoothing) via `trimesh` / `pymeshlab` to make outputs immediately more usable.

7. **Educational Focus & Clear Expectations**  
   - Explicit disclaimers and guidance: results are **prototypes** and final refinement should be done in Blender/FreeCAD. Provide quick links/tutorials for common refinement steps.

8. **Standard Exports**  
   - Exports in `.obj` and `.stl` (and `.glb` where applicable) for compatibility with common tools and 3D printing workflows.

---

## 🚀 Features

- **Text & Image Input:** Generate models from product descriptions or sketches.  
- **Material & Color Options:** Plastic, metal, wood, etc. with customizable colors.  
- **Live 3D Viewer:** Rotate, zoom, inspect geometry, toggle wireframe and material previews.  
- **Downloadable Formats:** Export `.obj` or `.stl` for prototyping and refinement.  
- **Session History:** Save, revisit, and re-download past designs.  
- **GPU-Accelerated:** Optimized for Google Colab GPU runtimes (recommended for faster inference).

---

## 🖥️ Running in Google Colab (GPU)

1. Open [Google Colab](https://colab.research.google.com/).  
2. Launch the notebook: `run_code.ipynb` (example path: `github.com/dimp01/MeshMind/run_code.ipynb`).  
3. Enable GPU:  
   - **Runtime → Change runtime type → Hardware accelerator → GPU (T4)**  
4. Run all cells — the notebook will:  
   - Install dependencies  
   - Load the text-to-3D model (Shap-E / Point-E or similar)  
   - Start a Streamlit UI via ngrok (or local forwarding)

5. Copy the ngrok/local URL displayed in the output and open it in your browser.

---

## 📌 Notes & Limitations

- This is an **academic demo**, not a production tool.  
- Models are **prototypes only** — refine in Blender/FreeCAD for final use.  
- Mesh quality is intentionally lightweight due to **dataset and GPU limits**.  
- Focus is on **usability, transparency, and educational value**.

---

## 🔧 Suggested Minimal Requirements (for local dev / Colab)

- Python 3.10+  
- GPU runtime recommended (Colab T4 / similar)  
- Typical Python packages (example list — add versions in `requirements.txt`):
  - `torch`
  - `trimesh`
  - `pymeshlab`
  - `streamlit`
  - `pyngrok` (if exposing via ngrok)
  - `imageio`
  - `numpy`
  - `scipy`

---

## 🧭 How MeshMind Solves Common Pain Points (Concise)

- **Quality:** Cleanup pipeline + preview reduces the need for immediate manual retopology.  
- **Cost:** Free academic access removes user friction around credits/subscriptions.  
- **Control:** Structured inputs (dimensions/materials/colors) give predictable outputs.  
- **Time:** Session history prevents unnecessary regenerations.  
- **Trust:** Clear messaging about limits and educational guidance builds user confidence.

---

## 📚 License

This project is open-source and free to use for academic and research purposes. Please see `LICENSE` for details.

---

## 🛠️ Contributors / Contact

- Project lead: *Your Name / Team*  
- Repo: `https://github.com/dimp01/MeshMind` (update as needed)  
- For academic queries, issues, or suggestions: *your-email@example.edu*

---
