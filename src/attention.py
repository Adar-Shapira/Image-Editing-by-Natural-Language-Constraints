import torch
import torch.nn.functional as F

class MaskedAttentionProcessor:
    def __init__(self):
        self.mask_tensor = None  # (1, 1, 512, 512)
        self.orig_attn_map = None 
        self.is_recording = False

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        # 1. Standard Attention Calculation
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Project Q, K, V
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Reshape to (Batch, Heads, SeqLen, Dim)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Get Attention Scores (Softmax)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # =======================================================
        # VRAM-OPTIMIZED BLENDING LOGIC
        # =======================================================
        if self.is_recording:
            # Save original map (Move to CPU to save GPU RAM if needed, but keeping on GPU for speed for now)
            # Optimization: detach() is crucial here to drop gradients
            self.orig_attn_map = attention_probs.detach().clone()
        
        elif self.mask_tensor is not None and self.orig_attn_map is not None:
            # 1. Resize Mask to match this layer's resolution
            # Resolution is sqrt(SequenceLength), e.g., 64x64 or 32x32
            current_res = int(attention_probs.shape[1] ** 0.5) 
            
            # 2. Interpolate Mask (Nearest Neighbor)
            # We assume mask_tensor is already (1, 1, 512, 512) from pipeline.py
            m_resized = F.interpolate(
                self.mask_tensor.float(), 
                size=(current_res, current_res), 
                mode='nearest'
            ).view(-1, 1) # Flatten to (Batch*Heads*SeqLen, 1) to broadcast
            
            # 3. In-Place Blending (Crucial for VRAM)
            # Formula: New = (Current * Mask) + (Old * (1-Mask))
            # Refactored: New = Old + Mask * (Current - Old) -> Saves memory
            
            # diff = Current - Old
            diff = attention_probs.sub(self.orig_attn_map) 
            
            # diff = diff * Mask
            diff.mul_(m_resized) 
            
            # Result = Old + diff
            attention_probs = self.orig_attn_map.add(diff)
            
            # Cleanup temp tensors
            del diff, m_resized

        # =======================================================
        
        # 4. Standard Output Projection
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
