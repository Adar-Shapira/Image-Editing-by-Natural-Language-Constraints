import torch
import numpy as np
from PIL import Image, ImageFilter
import scipy.ndimage
# FIX: Use generic loader to avoid class-specific import errors
from diffusers import DiffusionPipeline

class ControllableEditPipeline:
    def __init__(self, device="cuda", lora_path=None, **kwargs):
        self.device = device
        self.segmenter = None
        print("🚀 Loading Inpainting (via DiffusionPipeline)...")
        # Load model using generic pipeline to bypass import issues
        self.pipe = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16, safety_checker=None
        ).to(device)
        self.pipe.enable_model_cpu_offload()

        if lora_path:
            try:
                self.pipe.load_lora_weights(lora_path)
                print(f"✅ LoRA loaded: {lora_path}")
            except Exception as e: print(f"❌ LoRA Error: {e}")

    def edit(self, image, prompt, strength=0.75, detect_target=None, dilate_pixels=15, blur_radius=2, **kwargs):
        mask = None
        if self.segmenter and detect_target:
            # Auto-detect fallback
            search = detect_target
            if not search: 
                search = "cat" if "cat" in prompt.lower() else "dog"
            
            print(f"🔎 Segmenting: '{search}'")
            try:
                if hasattr(self.segmenter, 'detect_and_segment'): m = self.segmenter.detect_and_segment(image, search)
                else: m = self.segmenter.predict(image, [search])
                
                # Normalize mask
                if isinstance(m, tuple): m = m[0]
                if hasattr(m, 'cpu'): m = m.cpu().numpy()
                if isinstance(m, np.ndarray) and m.ndim > 2: m = m[0]

                if m is not None:
                    # Dilate
                    if dilate_pixels > 0:
                        s = scipy.ndimage.generate_binary_structure(2, 2)
                        m = scipy.ndimage.binary_dilation(m, structure=s, iterations=dilate_pixels).astype(np.float32)
                    else: m = m.astype(np.float32)
                    
                    # Feather
                    mask = Image.fromarray((m * 255).astype(np.uint8))
                    if blur_radius > 0: mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
            except Exception as e: print(f"⚠️ Seg Error: {e}")

        if mask is None:
            print("⚠️ No mask found. Returning original.")
            return image, None

        print(f"🎨 Inpainting: '{prompt}'")
        gen = torch.Generator(self.device).manual_seed(42)
        
        out = self.pipe(
            prompt=prompt, 
            image=image.resize((512,512)), 
            mask_image=mask.resize((512,512)), 
            strength=strength, 
            guidance_scale=7.5, 
            num_inference_steps=50, 
            generator=gen
        ).images[0]
        return out, mask