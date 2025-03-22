from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

class M2M100Model:
    def __init__(self, model_id="facebook/m2m100_1.2B", device=None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load(self):
        tokenizer = M2M100Tokenizer.from_pretrained(self.model_id)
        model = M2M100ForConditionalGeneration.from_pretrained(self.model_id).to(self.device)

        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        return model, tokenizer