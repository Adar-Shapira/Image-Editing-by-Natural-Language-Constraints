import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from src.parser import InstructionParser
from src.segmentation import GroundedSAM
from src.attention import MaskedAttentionProcessor

class ControllableEditPipeline:
    def __init__(self, device="cuda"):
        print("🚀 Initializing Controllable-Attention Pipeline (VRAM Optimized)...")
        self.device = device
        
        # 1. Load Components
        self.parser = InstructionParser(device=device)
        self.segmenter = GroundedSAM(device=device)
        
        # 2. Load Stable Diffusion
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        
        # VRAM Management: Offload to CPU when idle
        self.pipe.enable_model_cpu_offload()
        self.pipe.safety_checker = None
        print("✅ Pipeline Loaded with CPU Offloading.")

    def replace_attention_processors(self):
        """Swaps standard attention with Custom MaskedProcessor."""
        attn_procs = {}
        for name in self.pipe.unet.attn_processors.keys():
            attn_procs[name] = MaskedAttentionProcessor()
            
        self.pipe.unet.set_attn_processor(attn_procs)

    def edit(self, image: Image.Image, instruction: str, strength=0.8, guidance_scale=7.5):
        """The Main Inference Loop"""
        # Step A: Parse
        parse_result = self.parser.parse(instruction)
        target_obj = parse_result['object']
        print(f"🎯 Target: '{target_obj}' | Task: '{instruction}'")

        # Step B: Segment
        mask_np = self.segmenter.detect_and_segment(image, target_obj)
        mask_tensor = torch.from_numpy(mask_np).to(self.device)
        
        # === CRITICAL FIX START ===
        # F.interpolate requires (Batch, Channel, H, W). 
        # We turn (512, 512) -> (1, 1, 512, 512)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        # === CRITICAL FIX END ===
        
        mask_tensor = mask_tensor / 255.0 
        
        # Step C: Setup Attention
        self.replace_attention_processors()
        for processor in self.pipe.unet.attn_processors.values():
            if isinstance(processor, MaskedAttentionProcessor):
                processor.mask_tensor = mask_tensor

        # Step D: PASS 1 - Record Original
        print("📸 Pass 1: Recording Original Attention Maps...")
        for processor in self.pipe.unet.attn_processors.values():
            if isinstance(processor, MaskedAttentionProcessor):
                processor.is_recording = True
                
        with torch.no_grad():
            _ = self.pipe(prompt="", image=image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=20)

        # Step E: PASS 2 - Apply Edit
        print("✨ Pass 2: Generating Edited Image...")
        for processor in self.pipe.unet.attn_processors.values():
            if isinstance(processor, MaskedAttentionProcessor):
                processor.is_recording = False

        result = self.pipe(prompt=instruction, image=image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=50).images[0]

        return result, mask_np
