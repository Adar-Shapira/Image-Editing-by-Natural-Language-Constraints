import torch
import numpy as np
from PIL import Image
import os
import requests

# Import Metrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.multimodal.clip_score import CLIPScore

# Import Pipeline
import sys
sys.path.append(os.getcwd())
from src.pipeline import ControllableEditPipeline

class Evaluator:
    def __init__(self, device="cuda"):
        print(f"📊 Initializing Evaluator & Metrics on {device.upper()}...")
        self.device = device
        
        # 1. Load Metrics
        # Note: Metrics are loaded to the specific device to avoid mismatch errors
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
        
        # 2. Find LoRA (Check Local first, then Drive)
        lora_path = "lora_instruction_tuned"
        if not os.path.exists(lora_path):
             lora_path = "/content/drive/MyDrive/Projects/Image-Editing-by-Natural-Language-Constraints/lora_instruction_tuned"
        
        # 3. Load Pipeline
        self.pipeline = ControllableEditPipeline(device=device, lora_path=lora_path)
        print("✅ Evaluator Ready.")

    def preprocess_for_metric(self, image_pil):
        img_np = np.array(image_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)

    def calculate_background_metrics(self, orig_pil, edit_pil, mask_np):
        inv_mask = 1.0 - (mask_np.astype(np.float32) / 255.0)
        inv_mask_tensor = torch.from_numpy(inv_mask).unsqueeze(0).unsqueeze(0).to(self.device)
        
        orig_t = self.preprocess_for_metric(orig_pil)
        edit_t = self.preprocess_for_metric(edit_pil)
        
        bg_orig = orig_t * inv_mask_tensor
        bg_edit = edit_t * inv_mask_tensor
        
        with torch.no_grad():
            score_ssim = self.ssim(bg_edit, bg_orig)
            score_lpips = self.lpips(bg_edit, bg_orig)
            
        return score_ssim.item(), score_lpips.item()

    def calculate_clip_score(self, image_pil, text_prompt):
        img_tensor = (torch.tensor(np.array(image_pil)).permute(2, 0, 1)).to(self.device)
        with torch.no_grad():
            score = self.clip(img_tensor, text_prompt)
        return score.item()

    def run_single_test(self, image_path, instruction):
        orig_image = Image.open(image_path).convert("RGB").resize((512, 512))
        
        print(f"⚡ Editing: '{instruction}'...")
        # Use optimal strength 0.7 from previous sweep
        edited_image, mask_np = self.pipeline.edit(
            orig_image, 
            instruction, 
            strength=0.7 
        )
        
        print("📉 Calculating Metrics...")
        bg_ssim, bg_lpips = self.calculate_background_metrics(orig_image, edited_image, mask_np)
        clip_score = self.calculate_clip_score(edited_image, instruction)
        
        results = {
            "instruction": instruction,
            "clip_score": round(clip_score, 4),
            "bg_ssim": round(bg_ssim, 4),
            "bg_lpips": round(bg_lpips, 4)
        }
        
        return edited_image, results

if __name__ == "__main__":
    # --- SMART DEVICE DETECTION ---
    # This block prevents the script from crashing if GPU is not available
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Launching Benchmark Script on: {device_name.upper()}")

    evaluator = Evaluator(device=device_name)
    
    # Run Test
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img_path = "test_cat.jpg"
    if not os.path.exists(img_path):
        try:
            Image.open(requests.get(url, stream=True).raw).save(img_path)
        except:
            print("⚠️ Could not download test image. Please ensure 'test_cat.jpg' exists.")
    
    if os.path.exists(img_path):
        img, metrics = evaluator.run_single_test(img_path, "Change the cat into a dog")
        print("\n🏆 Final Metrics (LoRA + Inversion):")
        print(metrics)
    else:
        print("❌ Error: Test image not found.")