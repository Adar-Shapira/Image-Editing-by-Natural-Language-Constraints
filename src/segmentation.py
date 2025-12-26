import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from segment_anything import sam_model_registry, SamPredictor
import os
import gc

class GroundedSAM:
    def __init__(self, device="cuda", sam_checkpoint="sam_vit_b_01ec64.pth"):
        print("Loading Grounded-SAM (Lite Version)...")
        self.device = device
        
        # 1. Load Grounding DINO (Keep on CPU initially)
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.detector = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
        
        # 2. Load SAM-Base (Much lighter!)
        if not os.path.exists(sam_checkpoint):
            print(f"⚠️ Checkpoint {sam_checkpoint} not found! Downloading...")
            os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/{sam_checkpoint}")

        # Use the Registry to load the correct architecture (vit_b)
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(self.sam)
        print("✅ Grounded-SAM (Lite) loaded.")

    def detect_and_segment(self, image_pil, text_prompt, box_threshold=0.25):
        # Fix prompt punctuation
        if not text_prompt.endswith("."):
            text_prompt = text_prompt + "."

        print(f"   -> Searching for: '{text_prompt}'...")

        # --- PHASE 1: DETECTOR (Hot-Swap to GPU) ---
        self.detector.to(self.device)
        
        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detector(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image_pil.size[::-1]])
        results = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=box_threshold, target_sizes=target_sizes
        )[0]
        
        # Move Detector back to CPU
        self.detector.to("cpu")
        torch.cuda.empty_cache()

        # Fallback Logic
        if len(results["boxes"]) == 0:
            print(f"⚠️ Exact match failed. Retrying with generic query...")
            self.detector.to(self.device)
            inputs = self.processor(images=image_pil, text="object.", return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.detector(**inputs)
            results = self.processor.image_processor.post_process_object_detection(
                outputs, threshold=0.2, target_sizes=target_sizes
            )[0]
            self.detector.to("cpu")
            torch.cuda.empty_cache()

        if len(results["boxes"]) == 0:
            print(f"❌ Still no object found. Returning empty mask.")
            return np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)

        # Pick best box
        best_box_idx = results["scores"].argmax()
        best_box = results["boxes"][best_box_idx].cpu().detach().numpy()

        # --- PHASE 2: SAM (Hot-Swap to GPU) ---
        self.sam.to(self.device)
        self.predictor.model = self.sam 
        
        image_np = np.array(image_pil)
        self.predictor.set_image(image_np)
        masks, _, _ = self.predictor.predict(box=best_box, multimask_output=False)
        
        # Move SAM back to CPU
        self.sam.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
        
        return masks[0].astype(np.uint8)
