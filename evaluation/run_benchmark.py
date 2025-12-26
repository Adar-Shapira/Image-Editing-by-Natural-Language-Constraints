import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os

# Import Metrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.multimodal.clip_score import CLIPScore

# Import Pipeline
import sys
sys.path.append(os.getcwd()) # Ensure we can see 'src'
from src.pipeline import ControllableEditPipeline

class Evaluator:
    def __init__(self, device="cuda"):
        print("📊 Initializing Evaluator & Metrics...")
        self.device = device
        
        # 1. Load Metrics (Move to GPU for speed)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
        
        # 2. Load Your Pipeline
        self.pipeline = ControllableEditPipeline(device=device)
        print("✅ Evaluator Ready.")

    def preprocess_for_metric(self, image_pil):
        """Converts PIL image to (1, 3, H, W) tensor normalized [0,1]"""
        img_np = np.array(image_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)

    def calculate_background_metrics(self, orig_pil, edit_pil, mask_np):
        """
        Calculates SSIM and LPIPS only on the background (Inverse Mask).
        Proposal Reference: 'metrics on the parts of the image that were not edited'
        """
        # Create Inverse Mask (1 = Background, 0 = Object)
        inv_mask = 1.0 - (mask_np.astype(np.float32) / 255.0)
        inv_mask_tensor = torch.from_numpy(inv_mask).unsqueeze(0).unsqueeze(0).to(self.device) # (1,1,H,W)
        
        # Convert images to tensors
        orig_t = self.preprocess_for_metric(orig_pil)
        edit_t = self.preprocess_for_metric(edit_pil)
        
        # Apply Inverse Mask (Black out the object)
        # We perform the metric check on ONLY the visible background pixels
        bg_orig = orig_t * inv_mask_tensor
        bg_edit = edit_t * inv_mask_tensor
        
        # Calculate Metrics
        with torch.no_grad():
            score_ssim = self.ssim(bg_edit, bg_orig)
            score_lpips = self.lpips(bg_edit, bg_orig) # Lower is better for LPIPS
            
        return score_ssim.item(), score_lpips.item()

    def calculate_clip_score(self, image_pil, text_prompt):
        """Calculates text-image alignment."""
        img_tensor = (torch.tensor(np.array(image_pil)).permute(2, 0, 1)).to(self.device)
        with torch.no_grad():
            score = self.clip(img_tensor, text_prompt)
        return score.item()

    def run_single_test(self, image_path, instruction):
        """Runs the pipeline on one image and evaluates it."""
        # Load Image
        orig_image = Image.open(image_path).convert("RGB").resize((512, 512))
        
        # Run Pipeline
        print(f"⚡ Editing: '{instruction}'...")
        edited_image, mask_np = self.pipeline.edit(
            orig_image, 
            instruction, 
            strength=0.8
        )
        
        # Calculate Scores
        print("📉 Calculating Metrics...")
        bg_ssim, bg_lpips = self.calculate_background_metrics(orig_image, edited_image, mask_np)
        clip_score = self.calculate_clip_score(edited_image, instruction)
        
        results = {
            "instruction": instruction,
            "clip_score": round(clip_score, 4),     # Higher = Better instruction following
            "bg_ssim": round(bg_ssim, 4),           # Higher = Better background preservation
            "bg_lpips": round(bg_lpips, 4)          # Lower = Better background preservation
        }
        
        return edited_image, results

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Just a dummy test to ensure the script runs
    import requests
    
    # Setup
    evaluator = Evaluator()
    
    # Download Test Image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img_path = "test_cat.jpg"
    Image.open(requests.get(url, stream=True).raw).save(img_path)
    
    # Run Test
    img, metrics = evaluator.run_single_test(img_path, "Change the cat into a dog")
    
    print("\n🏆 Final Metrics:")
    print(metrics)
