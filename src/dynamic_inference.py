
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

class DynamicConfig:
    _embedder = None
    _regressor = None
    _classifier = None

    @classmethod
    def load_models(cls):
        if cls._embedder is None:
            # Singleton: Load once
            cls._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            with open("brain_regressor_hybrid.pkl", "rb") as f: cls._regressor = pickle.load(f)
            with open("brain_classifier_hybrid.pkl", "rb") as f: cls._classifier = pickle.load(f)

    @staticmethod
    def infer(prompt, target):
        DynamicConfig.load_models()
        
        # 1. AI PREDICTION (The Brain)
        input_text = f"{prompt} | target: {target}"
        vector = DynamicConfig._embedder.encode([input_text])
        
        cont_preds = DynamicConfig._regressor.predict(vector)[0]
        disc_preds = DynamicConfig._classifier.predict(vector)[0]
        
        mask_map = {0: "standard", 1: "box", 2: "inverse"}
        
        config = {
            "strength": float(cont_preds[0]),
            "guidance_scale": float(cont_preds[1]),
            "controlnet_scale": float(cont_preds[2]),
            "dilate_pixels": int(cont_preds[3]),
            "blur_radius": int(cont_preds[4]),
            "use_controlnet": bool(disc_preds[0]),
            "mask_strategy": mask_map[int(disc_preds[1])]
        }
        
        # 2. LOGIC OVERRIDES (The Safety Net)
        
        # Rule A: If target is background, FORCE Inverse.
        # (Fixes the CivitAI data pollution bug)
        if "background" in target.lower() or "background" in prompt.lower():
            config["mask_strategy"] = "inverse"
            config["target"] = "person" if "man" in prompt.lower() or "woman" in prompt.lower() else "subject"
            
        # Rule B: Color Consistency Helper
        if config["dilate_pixels"] > 10 and "matching" not in prompt.lower():
            config["prompt_override"] = "matching " + prompt

        print(f"   🧠 AI+Logic: {config['mask_strategy'].upper()} | Str: {config['strength']:.2f} | CNet: {config['controlnet_scale']:.2f}")
        return config
