# Few-Shot Toxicity Analysis and Generation Pipeline

This pipeline analyzes or generates text using toxicity scores, supporting both forward (score prediction) and backward (text generation) modes with Gemma3 or Llama3 models.

## Setup

### Prerequisites
- Python 3.10+
- Access to model weights:
  - **Gemma3**: Requires HuggingFace account and access to the model.
  - **Llama3**: Use the `dolphin-llama3` variant from HuggingFace Hub.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/anthol42/GLO-7030
   cd few_shot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m pip install wheel setuptools
   pip install flash-attn --no-build-isolation
   ```

3. Configure Hugging Face Token (only for Gemma3):
   ```bash
   huggingface-cli login
   ```
   Follow the prompts to enter your Hugging Face token. To generate a token, visit [Hugging Face Tokens](https://huggingface.co/settings/tokens).

## Usage

### Basic Command
```bash
python main.py \
  --model [gemma3|llama3] \
  --mode [forward|backward] \
  --n_shot [number_of_examples]
```

### Modes

#### 1. Forward Pass (Score Prediction)
Predict toxicity scores for existing comments:
```bash
python main.py \
  --model gemma3 \
  --mode forward \
  --n_examples 50 \
  --n_shot 5
```

#### 2. Backward Pass (Comment Generation)
Generate comments from target toxicity scores:
```bash
python main.py \
  --model llama3 \
  --mode backward \
  --num_samples 1000 \
  --n_shot 8
```

### Parameters
| Argument        | Description                           | Default | Valid Values        |
|-----------------|---------------------------------------|---------|---------------------|
| `--model`       | Model architecture                    | gemma3  | gemma3, llama3      |
| `--mode`        | Pipeline direction                    | forward | forward, backward   |
| `--num_samples` | Comments to generate (backward mode) | 6000    | Any integer > 0     |
| `--n_examples`  | Comments to evaluate (forward mode)   | 100     | Any integer > 0     |
| `--n_shot`      | Example count for prompting           | 10      | 1-6000                |

## Outputs

### Forward Mode
- Console output with predictions vs actual scores:
  ```
  Comment: I disagree but respect your perspective
  Predicted Score: 0.15
  Actual Score: 0.12
  ----------------------------------------------
  RMSE: 0.08
  ```

### Backward Mode
- CSV file (`few_shot_backward.csv`) containing:
  ```csv
  body,score
  "This was mildly frustrating",0.35
  "Absolutely perfect experience!",0.85
  ```
