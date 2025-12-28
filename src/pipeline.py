import torch
import os
from PIL import Image
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from src.parser import InstructionParser
from src.segmentation import GroundedSAM
from src.attention import MaskedAttentionProcessor
from src.inversion import DDIMInverter

class ControllableEditPipeline:
    def __init__(self, device="cuda", lora_path=None):
        print("🚀 Initializing Controllable-Attention Pipeline (Final Version)...")
        self.device = device
        
        # 1. Load Components
        self.parser = InstructionParser(device=device)
        self.segmenter = GroundedSAM(device=device)
        
        # 2. Load Stable Diffusion
        scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            scheduler=scheduler,
            torch_dtype=torch.float16
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.safety_checker = None
        
        # === NEW: LOAD LORA WEIGHTS ===
        if lora_path and os.path.exists(lora_path):
            print(f"🧠 Loading LoRA 'Brain' from: {lora_path}")
            self.pipe.load_lora_weights(lora_path)
            self.pipe.fuse_lora()
        else:
            print("⚠️ No LoRA path provided. Using standard Stable Diffusion.")
        # ==============================
        
        # 3. Initialize Inverter
        self.inverter = DDIMInverter(self.pipe, device)
        print("✅ Pipeline Ready.")

    def image_to_latents(self, image):
        image = image.resize((512, 512))
        img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device).half()
        with torch.no_grad():
            latents = self.pipe.vae.encode(img_tensor).latent_dist.sample()
        return latents * self.pipe.vae.config.scaling_factor

    def replace_attention_processors(self):
        attn_procs = {}
        for name in self.pipe.unet.attn_processors.keys():
            attn_procs[name] = MaskedAttentionProcessor() 
        self.pipe.unet.set_attn_processor(attn_procs)

    def edit(self, image: Image.Image, instruction: str, strength=0.7, guidance_scale=7.5):
        # Factory Reset
        self.replace_attention_processors()

        # Step A: Parse
        parse_result = self.parser.parse(instruction)
        target_obj = parse_result['object']
        print(f"🎯 Target: '{target_obj}' | Task: '{instruction}'")

        # Step B: Segment
        mask_np = self.segmenter.detect_and_segment(image, target_obj)
        mask_tensor = torch.from_numpy(mask_np).to(self.device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0) 
        mask_tensor = mask_tensor / 255.0 
        
        # Step C: INVERSION
        print("🔄 Performing DDIM Inversion...")
        latents = self.image_to_latents(image)
        inv_latents = self.inverter.invert(latents, num_inference_steps=50)

        # Step D: Configure Processors
        for processor in self.pipe.unet.attn_processors.values():
            if isinstance(processor, MaskedAttentionProcessor):
                processor.mask_tensor = mask_tensor

        # Step E: PASS 1 - Record Original
        print("📸 Pass 1: Recording Original Attention Maps...")
        for processor in self.pipe.unet.attn_processors.values():
            if isinstance(processor, MaskedAttentionProcessor):
                processor.is_recording = True
                
        with torch.no_grad():
            _ = self.pipe(
                prompt="", 
                image=image, 
                latents=inv_latents, 
                strength=strength, 
                guidance_scale=guidance_scale, 
                num_inference_steps=50
            )

        # Step F: PASS 2 - Apply Edit
        print("✨ Pass 2: Generating Edited Image...")
        for processor in self.pipe.unet.attn_processors.values():
            if isinstance(processor, MaskedAttentionProcessor):
                processor.is_recording = False

        result = self.pipe(
            prompt=instruction, 
            image=image,
            latents=inv_latents,
            strength=strength, 
            guidance_scale=guidance_scale, 
            num_inference_steps=50
        ).images[0]

        return result, mask_np
