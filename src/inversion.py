import torch
from tqdm import tqdm

class DDIMInverter:
    def __init__(self, pipe, device="cuda"):
        self.pipe = pipe
        self.device = device
        self.scheduler = pipe.scheduler

    @torch.no_grad()
    def invert(self, image_latents, num_inference_steps=50):
        """
        Reverses the diffusion process: Image -> Noise (z_T)
        """
        print(f"🔄 Running DDIM Inversion ({num_inference_steps} steps)...")
        
        # 1. Setup Scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps.flip(0)
        
        # 2. Pre-compute Empty Text Embedding (Unconditional)
        # We use the public API 'encode_prompt' which is stable
        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt="", 
            device=self.device, 
            num_images_per_prompt=1, 
            do_classifier_free_guidance=False, 
            negative_prompt=None
        )

        # 3. Inversion Loop
        latents = image_latents.clone()
        
        for t in tqdm(timesteps, desc="Inverting"):
            # A. Predict Noise
            # We pass the pre-computed empty embedding here
            noise_pred = self.pipe.unet(
                latents, 
                t, 
                encoder_hidden_states=prompt_embeds
            ).sample

            # B. Compute Previous Sample (Reverse Step)
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            
            # Get next timestep alphas
            current_step_idx = (self.scheduler.timesteps == t).nonzero().item()
            if current_step_idx < len(self.scheduler.timesteps) - 1:
                next_t = self.scheduler.timesteps[current_step_idx + 1]
                alpha_prod_t_next = self.scheduler.alphas_cumprod[next_t]
            else:
                alpha_prod_t_next = self.scheduler.final_alpha_cumprod

            # DDIM Equation (Inverted)
            f_t = (latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5)
            latents = (alpha_prod_t_next ** 0.5) * f_t + ((1 - alpha_prod_t_next) ** 0.5) * noise_pred
            
        return latents
