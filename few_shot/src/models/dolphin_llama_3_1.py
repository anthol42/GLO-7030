from transformers import AutoProcessor, LlamaForCausalLM
import torch

# WIP not working yet
class DolphinLLama3_1:
    def __init__(self, model_id="cognitivecomputations/Dolphin3.0-Llama3.1-8B", device=None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
    def load(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        
        model = LlamaForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto"
        )
        
        model.eval()
        
        return model, processor