from rich.progress import track
from rich.console import Console
from rich.table import Table
from rich import print
import numpy as np
import torch
import tqdm

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
    
    def forward_pass(self, comments, n_shot=10, max_length=50):
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
        table = self._create_score_table(data=[], title="Toxicity Score Prediction")
        
        for comment in track(comments, description="Analyzing comments..."):
            messages = self.prompt_builder.construct_few_shot_prompt_forward(comment, n=n_shot)

            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt",
                truncation=False,
                max_length=None,
            ).to(self.model.device, dtype=torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.9,
                    top_p=0.85,
                    num_return_sequences=1,
                    do_sample=True
                )
                generation = generation[0][input_len:]
                
            generated_text = self.processor.decode(generation, skip_special_tokens=True)
            
            try:
                score = float(generated_text.split("Toxicity Score:")[-1].strip())
                score = np.clip(score, -1.0, 1.0)
            except:
                score = None
                
            results.append({
                "comment": comment,
                "score": score,
                "full_output": generated_text
            })
            
            table.add_row(
                comment, 
                f"{score:.2f}" if score is not None else "Error", 
                "", 
                str(generated_text)
            )
            
        self.console.print(table)
        return results

    def backward_pass(self, num_samples=50, distribution='uniform', 
                    mean=0.0, std=0.5, max_length=100, 
                    n_shot=10):
        """
        Generate comments based on toxicity scores using a standard iterative approach
        """
        # Generate target scores
        if distribution == 'uniform':
            scores = np.linspace(-1, 1, num_samples)
        elif distribution == 'normal':
            scores = np.random.normal(mean, std, num_samples)
            scores = np.clip(scores, -1.0, 1)
        else:
            raise ValueError("Invalid distribution type")

        results = []
        table = self._create_score_table(data=[], title="Comment Generation")
        
        # Process each score one by one
        for score in tqdm.tqdm(scores):
            # Create prompt with few-shot examples
            prompt = self.prompt_builder.construct_few_shot_prompt_backward(score, n=n_shot)
            
            # Tokenize the input
            inputs = self.processor.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False,
                max_length=None
            )
            
            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            if score < -0.5:  # Very positive comments
                temperature = 0.9  # Higher temperature for more creativity
                top_k = 50
            elif score > 0.5:  # Very negative comments
                temperature = 0.8
                top_k = 40
            else:  # Neutral comments
                temperature = 0.75
                top_k = 30
            
            # Generate response
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=0.92,
                    top_k=top_k,
                    do_sample=True,
                    repetition_penalty=1.2
                )
            
            # Decode the generation
            input_len = inputs["input_ids"].shape[-1]
            generated_text = self.processor.decode(generation[0][input_len:], skip_special_tokens=True)
            comment = self._extract_comment(generated_text)
            
            print(f"Generated comment: {comment}")
            
            # Store the result
            results.append({
                "target_score": float(score),
                "generated_comment": comment,
                "full_output": generated_text
            })
            
            table.add_row(f"{score:.2f}", comment, "", "✓")

        self.console.print(table)
        return results
    
    def _extract_comment(self, generated_text):
        """Helper to extract comment from model output"""
        try:
            if "Generated comment:" in generated_text:
                return generated_text.split("Generated comment:")[-1].split("\n")[0].strip()
            if "Generated Comment:" in generated_text:
                return generated_text.split("Generated Comment:")[-1].split("\n")[0].strip()
            return generated_text.strip()
        except:
            return generated_text
        
from openai import OpenAI
import time
        
class ToxicityPipelineOpenRouter(ToxicityPipeline):
    # Same function as ToxicityPipeline but using OpenRouter calls to generate
    # comments and scores
    
    def __init__(self, prompt_builder, api_key, model="google/gemini-2.0-flash-lite-001"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.console = Console()
        self.model = model
        self.prompt_builder = prompt_builder
        
    def call_model(self, prompt, max_tokens, max_retries=10, backoff_factor=2):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=0.9,
                    top_p=0.85,
                    n=1,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    print(f"API call failed: {str(e)}. Retrying in {wait_time}s...")
                    print(response)
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts: {str(e)}")
                    return ""
    
    def generate_comment(self, score, n_shot=10, max_length=100):
        prompt = self.prompt_builder.construct_few_shot_prompt_backward(score, n=n_shot)
        
        generated_text = self.call_model(prompt, max_length)
        
        comment = self._extract_comment(generated_text)
        
        return comment
    
    def backward_pass(self, num_samples=50, distribution='uniform',
                    mean=0.0, std=0.5, max_length=100, 
                    n_shot=10):
        """
        Generate comments based on toxicity scores using a standard iterative approach
        """
        # Generate target scores
        if distribution == 'uniform':
            scores = np.linspace(-1, 1, num_samples)
        elif distribution == 'normal':
            scores = np.random.normal(mean, std, num_samples)
            scores = np.clip(scores, -1.0, 1)
        else:
            raise ValueError("Invalid distribution type")

        results = []
        table = self._create_score_table(data=[], title="Comment Generation")
        
        # Process each score one by one
        for score in tqdm.tqdm(scores):
            # Create prompt with few-shot examples
            prompt = self.prompt_builder.construct_few_shot_prompt_backward(score, n=n_shot)
            
            # Call the OpenRouter API
            generated_text = self.call_model(prompt, max_length)
            
            comment = self._extract_comment(generated_text)
            
            # print(f"Generated comment: {comment}")
            
            # Store the result
            results.append({
                "target_score": float(score),
                "generated_comment": comment,
                "full_output": generated_text
            })
            
            table.add_row(f"{score:.2f}", comment, "", "✓")

        self.console.print(table)
        return results

