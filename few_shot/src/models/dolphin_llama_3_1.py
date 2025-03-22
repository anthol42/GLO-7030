from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# WIP not working yet
class DolphinLLama3_1:
    def __init__(self, model_id="Orenguteng/Llama-3-8B-Lexi-Uncensored", device=None, quantize=True, load_in_4bit=True):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantize = quantize
        self.load_in_4bit = load_in_4bit
        
    def load(self):
        # Chargement du tokenizer
        tokenizer = AutoProcessor.from_pretrained(self.model_id)
        
        # Configuration optimisée pour l'inférence
        if self.quantize and self.load_in_4bit:
            # Chargement avec quantification 4-bit
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
            )
        else:
            # Chargement standard mais avec configuration optimisée pour inférence
            model = LlamaForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        return model, tokenizer
