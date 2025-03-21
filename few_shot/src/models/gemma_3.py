# from  https://huggingface.co/google/gemma-3-12b-it
#       https://github.com/huggingface/transformers/releases/tag/v4.49.0-Gemma-3

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

class Gemma3Model:
    def __init__(self, model_id="google/gemma-3-12b-it", device=None, quantize=True, load_in_4bit=True):
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
            
            model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager"
            )
        else:
            # Chargement standard mais avec configuration optimisée pour inférence
            model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        return model, tokenizer
