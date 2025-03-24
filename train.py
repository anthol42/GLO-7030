import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Define paths
REAL_DATA_PATH = "ruddit_clean.csv"
SYNTHETIC_DATA_PATH = "reform_llama3.1-uncessored.csv"  # Your LLM-generated dataset
OUTPUT_DIR = "finetune_output_reform"
MODEL_NAME = "GroNLP/hateBERT"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define custom dataset class
class ToxicityDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length=128):
        self.encodings = tokenizer(list(texts), truncation=True, padding='max_length', 
                                  max_length=max_length, return_tensors='pt')
        self.scores = torch.tensor(scores, dtype=torch.float)
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.scores[idx]
        return item
    
    def __len__(self):
        return len(self.scores)

# Function to load and preprocess data
def load_data(path):
    df = pd.read_csv(path)
    # Ensure column names match
    if 'body' in df.columns and 'score' in df.columns:
        texts = df['body'].tolist()
        scores = df['score'].tolist()
        # Filter out None or NaN values
        valid_data = [(t, s) for t, s in zip(texts, scores) if t is not None and pd.notna(t) and pd.notna(s)]
        if not valid_data:
            print(f"Warning: No valid data found in {path}")
            return [], []
        texts, scores = zip(*valid_data)
        return list(texts), list(scores)
    else:
        print(f"Warning: Expected 'body' and 'score' columns in {path}. Found: {df.columns}")
        return [], []

# Function to prepare datasets with different mixing ratios
def prepare_mixed_dataset(train_real_texts, train_real_scores, synthetic_texts, synthetic_scores, mix_ratio):
    """
    Prepare dataset by mixing real and synthetic data
    mix_ratio: tuple of (real_parts, synthetic_parts) - e.g., (1, 1) for 1:1 ratio
    """
    real_parts, synthetic_parts = mix_ratio
    
    if synthetic_parts == 0:
        # Only real data
        return train_real_texts, train_real_scores
    
    # Calculate how many synthetic samples to use
    # For 1:1 ratio, use same number of synthetic as real
    # For 1:2 ratio, use twice as many synthetic as real, etc.
    real_count = len(train_real_texts)
    synthetic_count = min(int(real_count * synthetic_parts / real_parts), len(synthetic_texts))
    
    print(f"Using all {real_count} real training examples and {synthetic_count} synthetic examples")
    print(f"Ratio - Real:{real_parts}, Synthetic:{synthetic_parts}")
    
    # Sample from synthetic data
    indices = np.random.choice(len(synthetic_texts), synthetic_count, replace=False)
    sampled_synthetic_texts = [synthetic_texts[i] for i in indices]
    sampled_synthetic_scores = [synthetic_scores[i] for i in indices]
    
    # Combine datasets
    mixed_texts = train_real_texts + sampled_synthetic_texts
    mixed_scores = train_real_scores + sampled_synthetic_scores
    
    # Shuffle
    combined = list(zip(mixed_texts, mixed_scores))
    np.random.shuffle(combined)
    mixed_texts, mixed_scores = zip(*combined)
    
    print(f"Combined dataset size: {len(mixed_texts)}")
    return list(mixed_texts), list(mixed_scores)

# Function for evaluation
def evaluate_model(model, test_dataset, batch_size=16):
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model.eval()
    model.to(device)
    
    true_scores = []
    pred_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
            
            # Get predictions and true scores
            predictions = outputs.logits.squeeze().cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            
            true_scores.extend(labels)
            pred_scores.extend(predictions)
    
    # Calculate metrics
    mse = mean_squared_error(true_scores, pred_scores)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_scores, pred_scores)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'true_scores': true_scores,
        'pred_scores': pred_scores
    }

# Main function to run experiments with different mixing ratios
def run_experiments(real_data_path, synthetic_data_path, mix_ratios=[(1, 0), (1, 1), (1, 2), (1, 5)]):
    results = {}
    
    # Load data
    print("Loading data...")
    real_texts, real_scores = load_data(real_data_path)
    synthetic_texts, synthetic_scores = load_data(synthetic_data_path)
    
    if not real_texts or not synthetic_texts:
        print("Error: Could not load data. Please check file paths and formats.")
        return {}
    
    print(f"Loaded {len(real_texts)} real examples and {len(synthetic_texts)} synthetic examples")
    
    # Create a fixed test set from real data only (20% of real data)
    train_texts, test_texts, train_scores, test_scores = train_test_split(
        real_texts, real_scores, test_size=0.2, random_state=seed
    )
    
    print(f"Split real data into {len(train_texts)} training and {len(test_texts)} testing examples")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create test dataset
    test_dataset = ToxicityDataset(test_texts, test_scores, tokenizer)
    
    # Run experiment for each mix ratio
    for mix_ratio in mix_ratios:
        ratio_name = f"{mix_ratio[0]}:{mix_ratio[1]}"
        print(f"\n--- Running experiment with mix_ratio={ratio_name} ---")
        
        try:
            # Prepare mixed dataset
            mixed_texts, mixed_scores = prepare_mixed_dataset(
                train_texts, train_scores, synthetic_texts, synthetic_scores, mix_ratio
            )
            
            # Create train dataset
            train_dataset = ToxicityDataset(mixed_texts, mixed_scores, tokenizer)
            
            # Initialize model
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=1
            ).to(device)
            
            # Define training arguments
            training_args = TrainingArguments(
                output_dir=f"{OUTPUT_DIR}/mix_{ratio_name}",
                num_train_epochs=4,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=f"{OUTPUT_DIR}/logs",
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
            )
            
            # Define trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
            )
            
            # Train model
            print(f"Training model with mix_ratio={ratio_name}...")
            trainer.train()
            
            # Save model
            model_save_path = f"{OUTPUT_DIR}/model_mix_{ratio_name.replace(':', '_')}"
            trainer.save_model(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            # Evaluate model
            print(f"Evaluating model with mix_ratio={ratio_name}...")
            eval_results = evaluate_model(model, test_dataset)
            
            # Store results
            results[ratio_name] = eval_results
            print(f"Results for mix_ratio={ratio_name}:")
            print(f"  MSE: {eval_results['mse']:.4f}")
            print(f"  RMSE: {eval_results['rmse']:.4f}")
            print(f"  MAE: {eval_results['mae']:.4f}")
        
        except Exception as e:
            print(f"Error during experiment with mix_ratio={ratio_name}: {str(e)}")
            continue
    
    return results

# Visualize results
def visualize_results(results):
    if not results:
        print("No results to visualize.")
        return
        
    mix_ratios = list(results.keys())
    mse_values = [results[r]['mse'] for r in mix_ratios]
    rmse_values = [results[r]['rmse'] for r in mix_ratios]
    mae_values = [results[r]['mae'] for r in mix_ratios]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(range(len(mix_ratios)), mse_values, 'o-', label='MSE')
    plt.xticks(range(len(mix_ratios)), mix_ratios)
    plt.xlabel('Mix Ratio (Real:Synthetic)')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error vs Mix Ratio')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(range(len(mix_ratios)), rmse_values, 'o-', color='orange', label='RMSE')
    plt.xticks(range(len(mix_ratios)), mix_ratios)
    plt.xlabel('Mix Ratio (Real:Synthetic)')
    plt.ylabel('RMSE')
    plt.title('Root Mean Squared Error vs Mix Ratio')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(range(len(mix_ratios)), mae_values, 'o-', color='green', label='MAE')
    plt.xticks(range(len(mix_ratios)), mix_ratios)
    plt.xlabel('Mix Ratio (Real:Synthetic)')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error vs Mix Ratio')
    plt.grid(True)
    
    # For one mix ratio, plot predicted vs true values
    plt.subplot(2, 2, 4)
    best_mix = mix_ratios[np.argmin(rmse_values)]
    plt.scatter(results[best_mix]['true_scores'], results[best_mix]['pred_scores'], alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlabel('True Toxicity Score')
    plt.ylabel('Predicted Toxicity Score')
    plt.title(f'Predictions vs True Values (Mix Ratio = {best_mix})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/results_comparison.png")
    plt.show()
    
    # Print the best mix ratio
    best_mix = mix_ratios[np.argmin(rmse_values)]
    print(f"Best mix ratio based on RMSE: {best_mix}")

if __name__ == "__main__":
    # Define the mix ratios to experiment with
    # Format: (real_parts, synthetic_parts)
    mix_ratios = [
        (1, 1), # 1:1 ratio (equal amounts)
        (1.1, 1), # 1.1:1 ratio (slightly more real)
        (1.2, 1), 
        (1.5, 1),
        (1.8, 1),
        (2, 1),  # 2:1 ratio (twice as much real amounts)
        (2.1, 1),  # 2.1:1 ratio (thrice as much real)
        (2.2, 1),  # 2.2:1 ratio (four times as much real)
        (2.5, 1),  # 2.5:1 ratio (five times as much real)
        (1, 0)  # Real data only
    ]
    
    # Run experiments
    results = run_experiments(REAL_DATA_PATH, SYNTHETIC_DATA_PATH, mix_ratios)
    
    # Visualize results
    visualize_results(results)