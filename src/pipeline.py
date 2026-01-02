import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

class ControllableEditPipeline:
    # FIX: Added lora_path and **kwargs to __init__ to prevent crashes when called by Evaluator
    def __init__(self, device="cuda", lora_path=None, **kwargs):
        self.device = device
        print("🚀 Loading Robust SDEdit Pipeline...")
        if lora_path:
            print(f"ℹ️ Note: LoRA path '{lora_path}' is ignored in SDEdit mode.")
        
        # Load SDEdit Pipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(device)
        self.pipe.enable_model_cpu_offload()
        
        self.segmenter = None 

    def edit(self, image, prompt, strength=0.5, detect_target=None, **kwargs):
        # A. GENERATE MASK
        mask = None
        if self.segmenter:
            search_term = detect_target
            # Auto-detect target if missing
            if not search_term: 
                if "cat" in prompt.lower(): search_term = "cat"
                elif "dog" in prompt.lower(): search_term = "dog"
            
            if search_term:
                print(f"🔎 Segmenting subject: '{search_term}'...")
                try:
                    mask = self.segmenter.detect_and_segment(image, search_term)
                except Exception as e:
                    print(f"⚠️ Segmentation Warning: {e}")

        # B. SDEDIT
        print(f"🎨 Running SDEdit: '{prompt}' (Strength {strength})...")
        generator = torch.Generator(self.device).manual_seed(42)
        
        edited_image = self.pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator
        ).images[0]

        # C. COMPOSITE
        final_image = edited_image
        if mask is not None:
            print("✂️ Compositing (Restoring Background)...")
            if isinstance(mask, np.ndarray):
                mask = Image.fromarray((mask * 255).astype(np.uint8))
            mask = mask.resize(image.size).convert("L")
            final_image = Image.composite(edited_image, image, mask)

        return final_image, mask