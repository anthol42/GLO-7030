from rich.progress import track
from rich.console import Console
from rich.table import Table
import numpy as np
import torch

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

    def backward_pass(self, num_samples=5, distribution='uniform', 
                     mean=0.0, std=0.5, max_length=100):
        """
        Generate comments based on toxicity scores using GPU if available.
        
        Args:
            num_samples (int): Number of comments to generate
            distribution (str): 'uniform' or 'normal'
            mean (float): Mean for normal distribution
            std (float): Std dev for normal distribution
            max_length (int): Maximum generation length
            
        Returns:
            list: Generated comments with metadata
        """
        # Generate target scores
        if distribution == 'uniform':
            scores = np.linspace(-1, 1, num_samples)
        elif distribution == 'normal':
            scores = np.random.normal(mean, std, num_samples)
            scores = np.clip(scores, -1.0, 1.0)
        else:
            raise ValueError("Invalid distribution type")
            
        results = []
        table = self._create_score_table(data=[], title="Comment Generation")
        
        for score in track(scores, description="Generating comments..."):
            messages = self.prompt_builder.construct_few_shot_prompt_backward(score, n=10)
        
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
            comment = generated_text.split("Generated Comment:")[-1].strip()
            
            results.append({
                "target_score": score,
                "generated_comment": comment,
                "full_output": generated_text
            })
            
            table.add_row(
                f"{score:.2f}",
                comment,
                "",
                "✓"
            )
            
        self.console.print(table)
        return results