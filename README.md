# Image-Editing-by-Natural-Language-Constraints

│
├── notebooks/          # Experimentation (Jupyter notebooks)
│   ├── 01_mask_generation.ipynb
│   ├── 02_attention_hacking.ipynb
│   └── 03_pipeline_test.ipynb
│
├── src/                # Your actual Python modules (the "real" code)
│   ├── parser.py       # The LLM instruction parsing logic
│   ├── segmentation.py # Grounded-SAM logic & resizing
│   └── attention.py    # Your Custom Attention Processor class
│
├── data/               # Small config files (NOT datasets)
├── requirements.txt    # List of libraries (diffusers, transformers, etc.)
└── README.md
