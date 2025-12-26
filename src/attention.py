import torch
import torch.nn.functional as F

class MaskedAttentionProcessor:
    """
    Implements the formula: A' = A_edit * M + A_orig * (1 - M)
    """
    def __init__(self):
        self.mask_tensor = None  # The binary mask (M)
        self.orig_attn_map = None # The original image attention (A_orig)
        self.is_recording = False # Mode: Are we saving A_orig? or Applying Edit?

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        # NOTE: 'temb' and 'scale' args are added to match Diffusers 0.20+ signature
        
        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # =======================================================
        # MASK INJECTION LOGIC
        # =======================================================
        if self.is_recording:
            # Save the original attention map
            self.orig_attn_map = attention_probs.detach().clone()
        
        elif self.mask_tensor is not None:
            # Resize mask to match current attention layer resolution
            current_res = int(attention_probs.shape[1] ** 0.5) 
            
            # Resize mask to (1, HW, 1) to broadcast
            m_resized = F.interpolate(
                self.mask_tensor.float(), 
                size=(current_res, current_res), 
                mode='nearest'
            ).view(-1, 1) 
            
            # Blend: (Edit * Mask) + (Original * (1-Mask))
            if self.orig_attn_map is not None:
                 attention_probs = (attention_probs * m_resized) + \
                                   (self.orig_attn_map * (1 - m_resized))

        # =======================================================
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.dropout(hidden_states)

        return hidden_states
