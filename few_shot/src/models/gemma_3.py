# from  https://huggingface.co/google/gemma-3-4b-it
#       https://github.com/huggingface/transformers/releases/tag/v4.49.0-Gemma-3

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
from rich import print

class Gemma3Model:
    def __init__(self, model_id="google/gemma-3-12b-it", device=None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
    def load(self):
        print(f"Loading model {self.model_id}...")
        
        tokenizer = AutoProcessor.from_pretrained(self.model_id)
        
        model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
            
        model.eval()
        
        return model, tokenizer
