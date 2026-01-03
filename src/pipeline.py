import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import cv2
import scipy.ndimage
from diffusers import StableDiffusionXLControlNetInpaintPipeline, AutoencoderKL, ControlNetModel
from transformers import DPTImageProcessor, DPTForDepthEstimation
from src.dynamic_inference import DynamicConfig

class ControllableEditPipeline:
    def __init__(self, device="cuda", **kwargs):
        self.device = device
        self.segmenter = None
        
        print("🚀 Loading SDXL Pipeline (Smart V10)...")
        try:
            self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
            self.feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
            
            self.controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True
            )

            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

            self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                controlnet=self.controlnet,
                vae=vae,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to(device)
            
            self.pipe.enable_model_cpu_offload()
            print("✅ SDXL Pipeline Active.")

        except Exception as e:
            print(f"❌ Error loading SDXL: {e}")

    def get_depth_map(self, image, mask):
        # 1. Generate Depth
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.depth_estimator(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # 2. Resize to Match Image
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False,
        )
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
        
        # 3. Convert to Numpy
        # Result shape is usually (1, 1024, 1024) here
        depth_np = (prediction * 255.0).cpu().numpy().astype(np.uint8)[0]
        
        # --- THE FIX: FORCE SQUEEZE ---
        # Ensures (1, 1024, 1024) becomes (1024, 1024)
        depth_np = np.squeeze(depth_np)
        # ------------------------------
        
        depth_img = Image.fromarray(depth_np)
        
        if mask:
            mask_img = mask.convert("L").resize(image.size)
            blurred_depth = depth_img.filter(ImageFilter.GaussianBlur(radius=20))
            depth_img = Image.composite(depth_img, blurred_depth, mask_img)
        return depth_img

    def edit(self, image, prompt, detect_target=None, **kwargs):
        
        torch.cuda.empty_cache()

        # 1. DYNAMIC CONFIG
        auto_config = DynamicConfig.infer(prompt, detect_target if detect_target else "")
        
        strength = kwargs.get("strength", auto_config["strength"])
        guidance_scale = kwargs.get("guidance_scale", auto_config["guidance_scale"])
        use_controlnet = kwargs.get("use_controlnet", auto_config["use_controlnet"])
        control_scale = kwargs.get("controlnet_scale", auto_config["controlnet_scale"])
        dilate = kwargs.get("dilate_pixels", auto_config["dilate_pixels"])
        blur = kwargs.get("blur_radius", auto_config["blur_radius"])
        mask_strategy = kwargs.get("mask_strategy", auto_config["mask_strategy"])
        
        if mask_strategy == "inverse" and "target" in auto_config:
            detect_target = auto_config["target"]

        SAFE_RES = 1024
        proc_img = image.resize((SAFE_RES, SAFE_RES), Image.LANCZOS)
        
        # 2. MASK GENERATION
        mask = None
        if self.segmenter and detect_target:
            print(f"🔎 Segmenting: '{detect_target}' (Strategy: {mask_strategy})")
            try:
                if hasattr(self.segmenter, 'detect_and_segment'): m = self.segmenter.detect_and_segment(proc_img, detect_target)
                else: m = self.segmenter.predict(proc_img, [detect_target])
                
                # Robust Mask Shaping
                if isinstance(m, tuple): m = m[0]
                if hasattr(m, 'cpu'): m = m.cpu().numpy()
                if m.ndim == 3:
                     if m.shape[0] < 5: m = np.max(m, axis=0) 
                
                # Force Resize (Nuclear Option)
                if m.shape != (SAFE_RES, SAFE_RES):
                     try:
                         m_float = m.astype(np.float32)
                         m = cv2.resize(m_float, (SAFE_RES, SAFE_RES), interpolation=cv2.INTER_NEAREST)
                     except:
                         m = np.zeros((SAFE_RES, SAFE_RES))

                if m is not None:
                    if mask_strategy == "box":
                        rows = np.any(m, axis=1); cols = np.any(m, axis=0)
                        if np.any(rows) and np.any(cols):
                            y_min, y_max = np.where(rows)[0][[0, -1]]
                            x_min, x_max = np.where(cols)[0][[0, -1]]
                            pad = 40
                            y_min = max(0, y_min - pad); y_max = min(m.shape[0], y_max + pad)
                            x_min = max(0, x_min - pad); x_max = min(m.shape[1], x_max + pad)
                            m[:] = 0; m[y_min:y_max, x_min:x_max] = 1.0

                    if mask_strategy == "inverse":
                        print("   🔄 Inverting Mask...")
                        m = 1.0 - m

                    if dilate != 0:
                        s = scipy.ndimage.generate_binary_structure(2, 2)
                        d_pix = int(dilate * 2) 
                        m = scipy.ndimage.binary_dilation(m, structure=s, iterations=d_pix)
                    
                    mask = Image.fromarray((m * 255).astype(np.uint8))
                    if blur > 0: mask = mask.filter(ImageFilter.GaussianBlur(blur))
            except Exception as e: print(f"⚠️ Seg Error: {e}")

        if mask is None: 
            print("⚠️ Mask failed.")
            return image, None

        # 3. GENERATION
        control_image = self.get_depth_map(proc_img, mask)
        final_scale = control_scale if use_controlnet else 0.0
        
        print(f"   ✅ Params: Str={strength}, CNet={final_scale}, Guide={guidance_scale}, Mask={mask_strategy}")

        out = self.pipe(
            prompt=prompt, 
            negative_prompt="bad quality, blurry", 
            image=proc_img, 
            mask_image=mask,
            control_image=control_image, 
            controlnet_conditioning_scale=final_scale,
            strength=strength, 
            guidance_scale=guidance_scale, 
            num_inference_steps=30,
        ).images[0]
        
        return out.resize(image.size, Image.LANCZOS), mask.resize(image.size)
