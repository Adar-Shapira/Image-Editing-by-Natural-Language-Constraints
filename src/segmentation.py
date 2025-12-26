import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from segment_anything import build_sam, SamPredictor
import os

class GroundedSAM:
    def __init__(self, device="cuda", sam_checkpoint="sam_vit_h_4b8939.pth"):
        print("Loading Grounded-SAM components...")
        self.device = device
        
        # 1. Load Grounding DINO (Hugging Face)
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.detector = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(self.device)
        
        # 2. Load SAM (Checkpoints)
        if not os.path.exists(sam_checkpoint):
            print(f"⚠️ Warning: SAM checkpoint '{sam_checkpoint}' not found in current directory!")
        
        self.sam = build_sam(checkpoint=sam_checkpoint).to(self.device)
        self.predictor = SamPredictor(self.sam)
        print("✅ Grounded-SAM loaded.")

    def detect_and_segment(self, image_pil, text_prompt, box_threshold=0.25):
        # Fix: Ensure prompt ends with a dot
        if not text_prompt.endswith("."):
            text_prompt = text_prompt + "."

        print(f"   -> Searching for: '{text_prompt}' (Threshold: {box_threshold})")

        # Step A: Detect
        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detector(**inputs)
        
        target_sizes = torch.tensor([image_pil.size[::-1]])
        results = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=box_threshold, target_sizes=target_sizes
        )[0]

        # Fallback Logic
        if len(results["boxes"]) == 0:
            print(f"⚠️ Exact match failed. Retrying with generic 'object.' query...")
            inputs = self.processor(images=image_pil, text="object.", return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.detector(**inputs)
            results = self.processor.image_processor.post_process_object_detection(
                outputs, threshold=0.2, target_sizes=target_sizes
            )[0]

        if len(results["boxes"]) == 0:
            print(f"❌ Still no object found. Returning empty mask.")
            return np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)

        # Step B: Segment
        best_box_idx = results["scores"].argmax()
        best_box = results["boxes"][best_box_idx].cpu().detach().numpy()
        
        image_np = np.array(image_pil)
        self.predictor.set_image(image_np)
        masks, _, _ = self.predictor.predict(box=best_box, multimask_output=False)
        
        return masks[0].astype(np.uint8)
