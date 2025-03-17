from rich.progress import track
from rich.console import Console
from rich.table import Table
import numpy as np
import torch
import tqdm
import ast

from torch.utils.data import Dataset, DataLoader

class ToxicityScoreDataset(Dataset):
    """Dataset pour gérer les scores de toxicité et la préparation des prompts."""
    
    def __init__(self, scores, prompt_builder, processor, n_shot=10):
        self.scores = scores
        self.prompt_builder = prompt_builder
        self.processor = processor
        self.n_shot = n_shot
        
    def __len__(self):
        return len(self.scores)
        
    def __getitem__(self, idx):
        score = self.scores[idx]
        # Construction du prompt dans le worker CPU
        prompt = self.prompt_builder.construct_few_shot_prompt_backward(score, n=self.n_shot)
        
        if isinstance(prompt, list):
            inputs = self.processor.apply_chat_template(
                prompt, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt",
                truncation=False,
                max_length=None,
            )
        else:
            inputs = self.processor(
                prompt,
                return_tensors="pt",
                truncation=False,
                padding=False,
                max_length=None,
            )
        
        return {
            "score": score,
            "inputs": inputs,
            "input_len": inputs["input_ids"].shape[-1]
        }

class ToxicityPipeline:
    """
    A class to handle toxicity score prediction and comment generation using Transformers models.
    
    Attributes:
        model: Hugging Face transformers model
        tokenizer: Hugging Face tokenizer
        data_processor: RudditDataProcessor instance
        prompt_builder: ToxicityPromptBuilder instance
        console: Rich console instance for formatted output
    """
    
    def __init__(self, model, tokenizer, data_processor, prompt_builder):
        self.model = model
        self.processor = tokenizer
        self.data_processor = data_processor
        self.prompt_builder = prompt_builder
        self.console = Console()
        
    def _create_score_table(self, data, title):
        """Create a Rich table for displaying results."""
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Input", style="cyan")
        table.add_column("Output", style="green")
        table.add_column("Prompt", style="blue")
        table.add_column("Confidence", style="yellow")
        return table
    
    def forward_pass(self, comments, n_shot=3, max_length=50):
        """
        Predict toxicity scores for a list of comments.
        
        Args:
            comments (list): List of comments to analyze
            n_shot (int): Number of few-shot examples
            max_length (int): Maximum generation length
            
        Returns:
            list: Predicted scores with metadata
        """
        results = []
        table = self._create_score_table("Toxicity Score Predictions")
        
        for comment in track(comments, description="Analyzing comments..."):
            messages = self.prompt_builder.construct_few_shot_prompt_forward(comment, n=n_shot)

            self.console.print(messages)
            
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = self.model.generate(**inputs)
                generation = generation[0][input_len:]
                
            generated_text = self.processor.decode(generation, skip_special_tokens=True)
            
            try:
                score = float(generated_text.split("Predicted Toxicity Score:")[-1].strip())
                score = np.clip(score, -1.0, 1.0)
            except:
                score = None
                
            results.append({
                "comment": comment,
                "score": score,
                "full_output": generated_text
            })
            
            table.add_row(
                comment[:75] + "..." if len(comment) > 75 else comment,
                str(score) if score else "ERROR",
                "✓" if score else "✗"
            )
            
        self.console.print(table)
        return results

    def backward_pass(self, num_samples=50, distribution='uniform', 
                     mean=0.0, std=0.5, max_length=100, batch_size=8, 
                     num_workers=4, n_shot=10):
        """
        Generate comments based on toxicity scores using DataLoader for parallel CPU processing.
        """
        # Generate target scores
        if distribution == 'uniform':
            scores = np.linspace(-1, 1, num_samples)
        elif distribution == 'normal':
            scores = np.random.normal(mean, std, num_samples)
            scores = np.clip(scores, -1.0, 1)
        else:
            raise ValueError("Invalid distribution type")
        
        # Créer dataset et dataloader pour parallélisation
        dataset = ToxicityScoreDataset(scores, self.prompt_builder, self.processor, n_shot=n_shot)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=lambda x: x
        )
        
        results = []
        table = self._create_score_table(data=[], title="Comment Generation")
        
        for batch in tqdm.tqdm(dataloader):
            batch_scores = [item["score"] for item in batch]
            
            # Transfert vers GPU en une seule opération par batch
            batch_inputs = [item["inputs"].to(self.model.device) for item in batch]
            batch_input_lens = [item["input_len"] for item in batch]
            
            # Génération par batch
            batch_outputs = []
            with torch.inference_mode():
                for inputs, input_len in zip(batch_inputs, batch_input_lens):
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        temperature=0.9,
                        top_p=0.85,
                        repetition_penalty=1.5,
                        typical_p=0.9,
                        num_return_sequences=1,
                        do_sample=True
                    )[0][input_len:]
                    batch_outputs.append(generation)
            
            # Traitement des sorties et mise à jour des résultats
            for score, generation in zip(batch_scores, batch_outputs):
                generated_text = self.processor.decode(generation, skip_special_tokens=True)

                try:
                    if generated_text.startswith("[") and "text" in generated_text:
                        generated_text = generated_text.split("]")[0] + "]"
                        parsed = ast.literal_eval(generated_text)
                        if isinstance(parsed, list) and len(parsed) > 0 and 'text' in parsed[0]:
                            comment = parsed[0]['text']
                            if "Generated comment:" in comment:
                                comment = comment.split("Generated comment:")[-1].strip()
                        else:
                            comment = generated_text
                    else:
                        if "Generated comment:" in comment:
                            comment = generated_text.split("Generated comment:")[-1].split("\n")[0].strip()
                        if "Generated Comment:" in comment:
                            comment = generated_text.split("Generated Comment:")[-1].split("\n")[0].strip()
                except:
                    comment = generated_text
                
                results.append({
                    "target_score": score,
                    "generated_comment": comment,
                    "full_output": generated_text
                })
                
                table.add_row(f"{score:.2f}", comment, "", "✓")
        
        self.console.print(table)
        return results