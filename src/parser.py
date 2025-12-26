import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class InstructionParser:
    def __init__(self, model_name="google/flan-t5-large", device="cuda"):
        print(f"Loading Instruction Parser ({model_name})...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        print("✅ Parser loaded.")

    def parse(self, user_prompt):
        """
        Takes a raw user prompt (e.g., "Change the cat to a tiger")
        and returns a dictionary: {'object': 'cat', 'task': 'tiger'}
        """
        # We force the LLM to think in JSON structure
        input_text = (
            f"Extract the 'target_object' (what to change) and the 'editing_task' (what to do) "
            f"from this sentence: '{user_prompt}'. "
            f"Return the answer as a JSON string."
        )

        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids, 
                max_new_tokens=50,
                temperature=0.0  # Deterministic output for consistency
            )

        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-processing to ensure valid JSON (LLMs sometimes miss quotes)
        try:
            # Try to parse directly
            data = json.loads(decoded_output)
        except json.JSONDecodeError:
            # Fallback: simple text parsing if JSON fails
            # This is a safety net for the "Concerns" section of your proposal regarding error propagation
            print(f"⚠️ Warning: Raw LLM output was not valid JSON: '{decoded_output}'")
            return {"raw_output": decoded_output, "error": "parsing_failed"}

        return data

# ==========================================
# TEST BLOCK (Run this to verify)
# ==========================================
if __name__ == "__main__":
    parser = InstructionParser()
    
    test_prompts = [
        "Change the cat into a tiger",
        "Make the red car blue",
        "Remove the clouds from the sky"
    ]

    print("\n--- Testing Instruction Parser ---")
    for p in test_prompts:
        result = parser.parse(p)
        print(f"Input: '{p}'")
        print(f"Output: {result}\n")