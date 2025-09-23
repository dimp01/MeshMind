# MeshMind 🤖 AI-Powered 3D Product Designer

<a target="_blank" href="https://colab.research.google.com/github/dimp01/MeshMind/blob/main/run_code.ipynb)">![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)</a>

MeshMind is an advanced AI system for **automated 3D product design**.  
It leverages custom-trained diffusion and generative pipelines to generate realistic, manufacturable 3D models from structured product prompts.  

Designed for rapid prototyping and aesthetic optimization, MeshMind allows users to:

- Specify product dimensions, materials, style, and intended use  
- Generate highly detailed, symmetrical, and clean 3D meshes  
- Preview models live in a 3D viewer  
- Download `.obj` files for manufacturing, rendering, or simulation  
- Maintain a session history to revisit and reload previous designs  

---

## 🚀 Features

- **Custom Generative Pipelines:** Fine-tuned diffusion-based 3D model generation  
- **Structured Prompting:** Input features, dimensions, materials, and style for precise control  
- **Live 3D Viewer:** PyVista-powered interactive visualization  
- **Session History:** Save and reload previously generated models  
- **GPU Accelerated:** Fully compatible with Colab GPU runtimes for fast inference  
- **Modular Architecture:** Backend handles model inference and mesh processing; frontend handles UI and visualization  

---

## 🖥️ Running in Google Colab (GPU)

1. Open [Google Colab](https://colab.research.google.com/){:target="_blank"}.  
2. Open the notebook [`run_code.ipynb`](https://github.com/dimp01/MeshMind/blob/main/run_code.ipynb){:target="_blank"}.  
3. **Enable GPU:**  
   `Runtime → Change runtime type → Hardware accelerator → GPU (T4)`  
4. Run all cells. The notebook handles:
   - Installing dependencies  
   - Setting up `.env` credentials  
   - Off-screen rendering for PyVista  
   - Launching Streamlit with ngrok  

5. Open the ngrok URL displayed to access the app.

---

## 🛠️ Stopping the App

```python
from pyngrok import ngrok
import os

ngrok.kill()          # Close ngrok tunnels
!pkill -f streamlit   # Terminate Streamlit server
print("✅ Streamlit + ngrok terminated.")
