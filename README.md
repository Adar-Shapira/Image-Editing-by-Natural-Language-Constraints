# Adaptive AI Image Editing Framework
### A Context-Aware Meta-Controller for Autonomous Text-Guided Image Manipulation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Diffusers](https://img.shields.io/badge/Diffusers-SDXL-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Abstract

This project presents an **Adaptive AI Image Editing Framework** designed to overcome the inherent rigidity and manual complexity of standard text-guided diffusion pipelines. Conventional editing systems often fall into a **"Safety Trap,"** prioritizing structural preservation over the user’s requested changes, resulting in "stubborn" behavior.

To address this, we developed a novel architecture featuring a **Random Forest Meta-Controller** ("The Brain") that analyzes natural language instructions to dynamically predict optimal execution parameters. By training on a synthetic dataset of **15,000 expert examples** filtered through automated metrics like **CLIP** and **LPIPS**, our system transforms Stable Diffusion XL from a static tool into an intelligent, context-aware editing agent.

---

## 🚀 Key Features

* **🧠 Meta-Controller "Brain":** A lightweight Random Forest model that predicts interpretable editing parameters (Denoising Strength, ControlNet Scale) based on linguistic features.
* **👁️ Autonomous Segmentation:** Integrates **Grounded-SAM** (GroundingDINO + Segment Anything) for autonomous, pixel-perfect object detection and masking.
* **✋ Adaptive Generation:** Uses **SDXL Inpainting** with dynamic **ControlNet** guidance. The system intelligently disables structural constraints (ControlNet = 0.0) for generative tasks (e.g., "fill with cakes") while enforcing them for texture changes.
* **🛡️ Latent Barrier:** Implements a mathematical constraint in latent space to freeze background pixels, ensuring zero degradation of non-target areas.

---

## 🏗️ Architecture

The pipeline is designed as a modular, **closed-loop system** divided into three functional layers:

### 1. The Brain (Logic & Prediction)
* **NLP Parsing:** Uses dependency parsing (via Spacy) to decouple the 'Target Noun' from 'Spatial Adjectives' (e.g., "left cat").
* **Meta-Controller:** Analyzes the prompt to predict critical hyperparameters, such as Denoising Strength and ControlNet Scale. Unlike black-box neural networks, this layer offers interpretability.

### 2. The Eyes (Segmentation)
* **GroundingDINO:** Detects the object bounding box based on the textual target.
* **SAM (Segment Anything):** Converts the bounding box into a pixel-perfect mask.
* **Alignment:** Resizes the mask to 128x128 to align with SDXL latent dimensions.

### 3. The Hands (Execution)
* **SDXL Inpainting:** Performs iterative denoising to generate the new image content.
* **ControlNet:** Acts as a "structural skeleton" to guide the generation based on depth.
* **Latent Barrier:** Applies the formula `z_final = m * z_edited + (1-m) * z_original` to merge the edit seamlessly.

---

## 📊 Performance & Benchmarks

We evaluated the framework against standard SDXL baselines using a custom benchmark of object swaps, texture changes, and background replacements.

### The "Zone of Stubbornness"
Standard models often cluster in a high-SSIM/low-LPIPS quadrant (the "Zone of Stubbornness"), meaning they preserve structure but fail to execute radical changes. Our framework successfully pushes into the **high-LPIPS** quadrant, proving it can execute significant structural transformations.

### Quantitative Results
* **Effectiveness (LPIPS):** **65% more effective** at executing perceptual changes compared to standard baselines.
* **Accuracy (CLIP):** Maintained equivalent semantic alignment, proving the system breaks the trade-off between structural preservation and editing capability.

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone [https://github.com/your-username/adaptive-ai-editing.git](https://github.com/your-username/adaptive-ai-editing.git)
cd adaptive-ai-editing

# 2. Install PyTorch (Adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. Install Core Dependencies
pip install diffusers transformers accelerate scipy safetensors spacy scikit-image scikit-learn

# 4. Install Segment Anything & Spacy Model
pip install git+[https://github.com/facebookresearch/segment-anything.git](https://github.com/facebookresearch/segment-anything.git)
python -m spacy download en_core_web_sm
```
## 💻 Usage

### Inference (Running the Editor)

```python
from pipeline import ControllableEditPipeline
from PIL import Image

# Initialize the pipeline (Loads SDXL, SAM, and the Random Forest Brain)
# Ensure you have a GPU available
editor = ControllableEditPipeline(device="cuda")

# Load your image
image = Image.open("assets/input_image.jpg")

# Run the edit
# The 'Brain' automatically detects the target, creates a mask,
# and predicts the optimal parameters (e.g., Strength, ControlNet Scale).
result, mask, params = editor.edit(image, prompt="Turn the cat into a robot")

# Save result
result.save("output_robot.png")
print(f"Executed with adaptive params: {params}")
```

## 🧠 Training the Meta-Controller

We utilized a **Synthetic Expert Dataset** approach to train the Random Forest, as human labeling for floating-point parameters is impossible.

1.  **Simulation:** Generated **15,000** editing scenarios with randomized hyperparameters.
2.  **Gatekeeping:** Filtered results using **CLIP Score** (Semantic) and **LPIPS** (Perceptual) to retain only "Expert" examples.
3.  **Optimization:** Minimized **MSE** for continuous parameters and **Gini Impurity** for categorical choices (Mask Strategy).

## 📚 References

This project builds upon the following foundational research:

1.  R. Rombach et al., **"High-Resolution Image Synthesis with Latent Diffusion Models"** (CVPR 2022).
2.  L. Zhang et al., **"Adding Conditional Control to Text-to-Image Diffusion Models"** (ICCV 2023).
3.  A. Kirillov et al., **"Segment Anything"** (ICCV 2023).
4.  T. Brooks et al., **"InstructPix2Pix: Learning to Follow Image Editing Instructions"** (CVPR 2023).
5.  D. Podell et al., **"SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis"** (2023).

## 👥 Authors

* **Michal Yechezkel**
* **Almog Talker**
* **Adar Shapira**
