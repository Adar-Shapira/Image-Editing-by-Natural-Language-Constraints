
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

class MaskedAttentionProcessor:
    def __init__(self):
        self.is_recording = False
        self.mask_tensor = None

        # STORAGE: A list to hold (Key, Value) for every timestep
        self.storage = []
        self.read_idx = 0 # To keep track of which step we are reading

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        # 1. Standard Attention Setup
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            is_cross_attention = False
        else:
            is_cross_attention = True

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # =======================================================
        # 🚨 MASK PREPARATION
        # =======================================================
        should_mask = (self.mask_tensor is not None) and (not is_cross_attention)
        mask_flat = None

        if should_mask:
            current_key_len = key.shape[1]
            size = int(current_key_len ** 0.5)
            if size * size == current_key_len:
                # FIX: Use bilinear interpolation for smoother masks
                mask_resized = F.interpolate(self.mask_tensor, size=(size, size), mode='bilinear', align_corners=False)
                mask_flat = mask_resized.reshape(1, -1, 1)

        # =======================================================
        # 🔄 TIME-SYNCED INJECTION
        # =======================================================

        # PASS 1: RECORDING (Push to Storage)
        if self.is_recording and not is_cross_attention:
            # Offload to CPU to save VRAM
            self.storage.append((key.detach().cpu(), value.detach().cpu()))

        # PASS 2: EDITING (Pop from Storage)
        elif not self.is_recording and not is_cross_attention:
            if self.read_idx < len(self.storage):
                # Retrieve the correct key/value for THIS specific step
                stored_k_cpu, stored_v_cpu = self.storage[self.read_idx]

                # Move back to GPU
                stored_k = stored_k_cpu.to(key.device)
                stored_v = stored_v_cpu.to(value.device)

                # Increment index so next call gets the next step's data
                self.read_idx += 1

                if mask_flat is not None and key.shape[1] == stored_k.shape[1]:
                    inv_mask = 1.0 - mask_flat

                    # TELEMETRY
                    if self.read_idx % 20 == 0:
                        diff = (key - stored_k).abs().mean().item()
                        m_mean = mask_flat.mean().item()
                        inv_mean = inv_mask.mean().item()
                        print(f"DEBUG Step {self.read_idx}: MaskMean={m_mean:.4f} InvMean={inv_mean:.4f} KeyDiffL1={diff:.4f}")

                    # Inject!
                    key = stored_k * inv_mask + key * mask_flat
                    value = stored_v * inv_mask + value * mask_flat

        # =======================================================
        # Finish Attention Computation
        # =======================================================
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
