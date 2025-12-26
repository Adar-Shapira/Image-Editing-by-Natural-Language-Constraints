import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class InstructionParser:
    def __init__(self, model_name="google/flan-t5-large", device="cuda"):
        print(f"Loading Instruction Parser ({model_name})...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        print("✅ Parser loaded.")

    def parse(self, user_prompt):
        # STRATEGY: Few-Shot Prompting
        prompt_text = f"""
        Task: Extract the 'object' (noun to change) and the 'task' (full instruction).
        Return JSON format.

        Example 1:
        Input: Change the cat into a tiger
        Output: {{"object": "cat", "task": "Change the cat into a tiger"}}

        Example 2:
        Input: Make the red car blue
        Output: {{"object": "red car", "task": "Make the red car blue"}}

        Example 3:
        Input: Remove the clouds
        Output: {{"object": "clouds", "task": "Remove the clouds"}}

        Current Input:
        Input: {user_prompt}
        Output:
        """

        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=100,
                temperature=0.1
            )

        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Robust Parsing Logic
        try:
            data = json.loads(decoded_output)
        except json.JSONDecodeError:
            obj_match = re.search(r'"object":\s*"([^"]+)"', decoded_output)
            task_match = re.search(r'"task":\s*"([^"]+)"', decoded_output)
            
            if obj_match:
                data = {"object": obj_match.group(1), "task": task_match.group(1) if task_match else user_prompt}
            else:
                data = {"object": "unknown", "task": user_prompt, "error": "parsing_failed"}
        
        return data
