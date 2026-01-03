import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import cv2
import scipy.ndimage
from diffusers import AutoPipelineForInpainting, ControlNetModel, DiffusionPipeline
from transformers import DPTImageProcessor, DPTForDepthEstimation

class ControllableEditPipeline:
    def __init__(self, device="cuda", lora_path=None, **kwargs):
        self.device = device
        self.segmenter = None
        self.controlnet = None
        
        print("🚀 Loading Pipeline V5 (Manual Box Ready)...")
        try:
            self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
            self.feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
            self.controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth",
                torch_dtype=torch.float16
            )
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                controlnet=self.controlnet,
                torch_dtype=torch.float16,
                safety_checker=None
            ).to(device)
            self.pipe.enable_model_cpu_offload()
            print("✅ Pipeline Active.")
        except Exception as e:
            print(f"❌ ControlNet Error: {e}")
            self.pipe = DiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16,
                safety_checker=None
            ).to(device)

        if lora_path:
            try:
                self.pipe.load_lora_weights(lora_path)
                print(f"✅ LoRA loaded: {lora_path}")
            except Exception as e: print(f"❌ LoRA Error: {e}")

    def get_depth_map(self, image, mask):
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.depth_estimator(**inputs)
            predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False,
        )
        depth_min = torch.amin(prediction, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(prediction, dim=[1, 2, 3], keepdim=True)
        prediction = (prediction - depth_min) / (depth_max - depth_min)
        prediction = (prediction * 255.0).clip(0, 255).squeeze().cpu().numpy().astype(np.uint8)
        depth_img = Image.fromarray(prediction)
        if mask:
            mask_img = mask.convert("L").resize(image.size)
            blurred_depth = depth_img.filter(ImageFilter.GaussianBlur(radius=20))
            depth_img = Image.composite(depth_img, blurred_depth, mask_img)
        return depth_img

    def edit(self, image, prompt, strength=None, detect_target=None, 
             dilate_pixels=None, blur_radius=None, negative_prompt=None, 
             use_controlnet=True, use_box=False, manual_box=None, guidance_scale=None, **kwargs):
        
        # Defaults
        if strength is None: strength = 1.0
        control_scale = 0.8 if use_controlnet else 0.0
        if dilate_pixels is None: dilate_pixels = 0
        if blur_radius is None: blur_radius = 5
        if guidance_scale is None: guidance_scale = 7.5

        if not negative_prompt: negative_prompt = "bad quality, blurry"

        mask = None
        
        # --- MANUAL BOX OVERRIDE ---
        if manual_box:
            print(f"   📦 Using Manual Box: {manual_box}")
            # manual_box = [x_min, y_min, x_max, y_max] (normalized 0.0-1.0)
            w, h = image.size
            x1, y1, x2, y2 = [int(c * dim) for c, dim in zip(manual_box, [w, h, w, h])]
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=255)

        # --- AUTOMATIC SEGMENTATION ---
        elif self.segmenter and detect_target:
            search = detect_target
            if not search: search = "cat" if "cat" in prompt.lower() else "dog"
            print(f"🔎 Segmenting: '{search}'")
            try:
                if hasattr(self.segmenter, 'detect_and_segment'): m = self.segmenter.detect_and_segment(image, search)
                else: m = self.segmenter.predict(image, [search])
                if isinstance(m, tuple): m = m[0]
                if hasattr(m, 'cpu'): m = m.cpu().numpy()
                if isinstance(m, np.ndarray) and m.ndim > 2: m = m[0]

                if m is not None:
                    if use_box:
                        print("   📦 Converting Mask to Bounding Box...")
                        rows = np.any(m, axis=1); cols = np.any(m, axis=0)
                        if np.any(rows) and np.any(cols):
                            y_min, y_max = np.where(rows)[0][[0, -1]]
                            x_min, x_max = np.where(cols)[0][[0, -1]]
                            pad = 20
                            y_min = max(0, y_min - pad); y_max = min(m.shape[0], y_max + pad)
                            x_min = max(0, x_min - pad); x_max = min(m.shape[1], x_max + pad)
                            m[:] = 0; m[y_min:y_max, x_min:x_max] = 1.0

                    if dilate_pixels != 0:
                        s = scipy.ndimage.generate_binary_structure(2, 2)
                        if dilate_pixels > 0: m = scipy.ndimage.binary_dilation(m, structure=s, iterations=dilate_pixels)
                        else: m = scipy.ndimage.binary_erosion(m, structure=s, iterations=abs(dilate_pixels))
                    
                    mask = Image.fromarray((m * 255).astype(np.uint8))
                    if blur_radius > 0: mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
            except Exception as e: print(f"⚠️ Seg Error: {e}")

        if mask is None:
            print("⚠️ No mask found. Returning original.")
            return image, None

        # --- INPAINTING ---
        gen = torch.Generator(self.device).manual_seed(42)
        input_img = image.resize((512, 512))
        input_mask = mask.resize((512, 512))
        control_image = self.get_depth_map(input_img, input_mask)
        
        final_scale = control_scale if use_controlnet is not False else 0.0
        status_msg = f"Active (Scale: {final_scale})" if final_scale > 0 else "Disabled (Scale: 0.0)"

        if self.controlnet:
            print(f"   ✅ ControlNet-Depth {status_msg} | Guidance: {guidance_scale}")
            out = self.pipe(prompt=prompt, negative_prompt=negative_prompt, image=input_img, mask_image=input_mask,
                            control_image=control_image, controlnet_conditioning_scale=final_scale,
                            strength=strength, guidance_scale=guidance_scale, num_inference_steps=50, generator=gen).images[0]
        else:
            print(f"   ⚠️ Standard Inpainting (No CNet) | Guidance: {guidance_scale}")
            out = self.pipe(prompt=prompt, negative_prompt=negative_prompt, image=input_img, mask_image=input_mask,
                            strength=strength, guidance_scale=guidance_scale, num_inference_steps=50, generator=gen).images[0]
        return out, mask
