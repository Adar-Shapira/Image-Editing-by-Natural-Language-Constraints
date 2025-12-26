# Image Editing by Natural Language Constraints

**Authors:** Almog Talker, Michal Yechezkel, Adar Shapira  
**Course:** Generative Models, BGU

## Abstract
This project implements a **Controllable-Attention Pipeline** that integrates a Large Language Model (LLM) for instruction parsing with Grounded-SAM to guide the Stable Diffusion editing process. By injecting mask constraints directly into the Cross-Attention layers, we prevent "semantic leakage" and preserve the original background.

## Modules
1. **Instruction Parser:** Extracts target objects using an LLM.
2. **Mask Generation:** Uses Grounded-SAM to create precise binary masks.
3. **Custom Attention:** Modulates the U-Net noise prediction to strictly enforce edits within the mask.

Image-Editing-by-Natural-Language-Constraints
│
├── data/                   # Store datasets here (MagicBrush, etc.) - ADD TO GITIGNORE
├── outputs/                # Save generated images here - ADD TO GITIGNORE
├── notebooks/              # Jupyter/Colab notebooks for experiments
│   ├── 01_explore_sam.ipynb
│   └── 02_pipeline_test.ipynb
│
├── src/                    # Main source code
│   ├── __init__.py
│   ├── config.py           # Configuration (paths, model names)
│   ├── instruction.py      # LLM Parser logic [cite: 37]
│   ├── segmentation.py     # Grounded-SAM wrapper & resizing logic [cite: 39]
│   ├── attention.py        # Custom Masked CrossAttn Processor [cite: 40]
│   └── pipeline.py         # The main inference loop class
│
├── evaluation/             # Benchmarking scripts [cite: 71]
│   ├── metrics.py          # LPIPS, SSIM, CLIP Score implementations
│   └── run_benchmark.py    # Script to run over a dataset folder
│
├── .gitignore              # Crucial for keeping large files out of Git
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies