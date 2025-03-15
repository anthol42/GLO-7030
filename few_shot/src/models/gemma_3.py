# from  https://huggingface.co/google/gemma-3-4b-it
#       https://github.com/huggingface/transformers/releases/tag/v4.49.0-Gemma-3

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

class Gemma3Model:
    def __init__(self, model_id="google/gemma-3-4b-it", device=None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
    def load(self):
        tokenizer = AutoProcessor.from_pretrained(self.model_id)
        
        model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto"
        )
        
        if self.device == "cuda" and torch.cuda.is_available():
            model = model.to(self.device)
            
        model.eval()
        
        return model, tokenizer
