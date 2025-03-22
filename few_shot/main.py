from src.prompt_builder_it import ToxicityPromptBuilderIT
from src.prompt_builder import ToxicityPromptBuilder
from src.data_processor import RudditDataProcessor
from src.models.gemma_3 import Gemma3Model
from src.models.dolphin_llama_3_1 import DolphinLLama3_1
from src.pipeline import ToxicityPipeline

import pandas as pd
import numpy as np
import argparse

SEED = 42
N_EXAMPLES = 100

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run Toxicity Pipeline with specified model and mode.')
    parser.add_argument('--model', type=str, choices=['gemma3', 'llama3'], default='gemma3',
                        help='Model to use: "gemma3" or "llama3". Default is "gemma3".')
    parser.add_argument('--mode', type=str, choices=['forward', 'backward'], default='forward',
                        help='Mode to run: "forward" or "backward". Default is "forward".')
    parser.add_argument('--num_samples', type=int, default=6000,
                        help='Number of samples for backward pass. Default is 6000.')
    parser.add_argument('--n_examples', type=int, default=100,
                        help='Number of examples for forward pass. Default is 100.')
    parser.add_argument('--n_shot', type=int, default=10,
                        help='Number of few-shot examples for forward pass. Default is 10.')
    args = parser.parse_args()

    # Initialize data processor
    data_processor = RudditDataProcessor(seed=SEED)
    
    data_processor.load_from_kaggle()
    data_processor.clean_data()
    
    # Initialize prompt builder (adjust if needed based on model)
    prompt_builder = ToxicityPromptBuilderIT(data_processor.df, seed=None)
    # prompt_builder = ToxicityPromptBuilder(data_processor.df, seed=None)
    
    # Load the specified model
    if args.model == "gemma3":
        gemma = Gemma3Model()
        model, tokenizer = gemma.load()
    else:
        dolphin = DolphinLLama3_1()
        model, tokenizer = dolphin.load()
    n_shot = args.n_shot
    
    pipeline = ToxicityPipeline(model, tokenizer, data_processor, prompt_builder)
    
    if args.mode == "backward":
        # Backward pass: Generate comments from scores
        results = pipeline.backward_pass(
            num_samples=args.num_samples,
            max_length=100,
            n_shot=n_shot,
        )
        
        # Save results
        df = pd.DataFrame(results)[['generated_comment', 'target_score']]
        df = df.rename(columns={'generated_comment': 'body', 'target_score': 'score'})
        data_processor.save_to_csv("few_shot_backward.csv", df=df)
        print("Backward pass completed. Results saved to few_shot_backward.csv")
        
    else:
        # Forward pass: Predict scores for existing comments
        n_examples = args.n_examples
        data = data_processor.df.sample(n=n_examples, random_state=SEED)
        comments = data.body.tolist()
        
        results = pipeline.forward_pass(
            comments=comments,
            n_shot=n_shot,
            max_length=10,
        )
        
        # Compare predictions with actual scores
        for i, result in enumerate(results):
            print(f"Comment: {comments[i]}")
            print(f"Predicted Score: {result['score']}")
            print(f"Actual Score: {data.iloc[i]['score']}")
            print("-" * 50)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((data['score'] - [result['score'] for result in results]) ** 2))
        print(f"RMSE: {rmse:.2f}")