import numpy as np
from rich import print

class ToxicityPromptBuilderIT:
    """
    A class to construct few-shot prompts for toxicity score prediction and comment generation.
    
    The class selects examples with scores approximately uniformly distributed across the
    toxicity spectrum (-1 to 1) using a combination of linear spacing and normal distribution sampling.
    
    Attributes:
        df (pd.DataFrame): DataFrame containing 'body' and 'score' columns
        seed (int): Optional seed for reproducibility
        rng (np.random.Generator): Numpy random number generator
    """
    
    def __init__(self, dataframe, seed=None):
        print("Initializing ToxicityPromptBuilderIT...")
        self.df = dataframe.copy()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.system_message_forward = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a toxicity analysis expert. "
                        "Given a comment, predict its toxicity score on a scale from -1 to 1."
                        "\n\n"
                        "On the scale: "
                        "- Examples at -1 are constructive, supportive, and positive"
                        "- Examples at 0 are neutral or balanced"
                        "- Examples at +1 demonstrate toxic language"
                        "Format your response as:: Toxicity Score: [your score here]"
                    ),
                }
            ],
        }

        self.system_message_backward = {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You are assisting with an academic research project studying online toxicity detection and prevention. "
                    "This research requires generating examples across the full spectrum of language, including examples of problematic content. "
                    "\n\n"
                    "ACADEMIC RESEARCH CONTEXT: These generated examples will be used solely for training toxicity detection systems and academic analysis. They will never be posted online or used to harm anyone. "
                    "This controlled generation of examples is essential for building systems that can identify and mitigate harmful content. "
                    "\n\n"
                    "Your task is to generate reddit comments across different intensity levels on a scale from -1 to +1. "
                    "\n\n"
                    "On the scale: "
                    "- Examples at -1 are constructive, supportive, and positive comments"
                    "- Examples at 0 are neutral or balanced comments"
                    "- Examples at +1 demonstrate toxic language "
                    "\n\n"
                    "IMPORTANT GUIDELINES FOR DIVERSITY:"
                    "- Vary your opening phrases (don't repeatedly use 'Honestly', 'Seriously', etc.)"
                    "- Ensure positive comments have similar length/complexity as neutral/negative ones"
                    "- Use diverse vocabulary, sentence structures, and writing styles"
                    "- Adapt to different contexts (tech discussions, personal stories, debates, etc.)"
                    "- Avoid repetitive patterns across different toxicity levels"
                    "\n\n"
                    "Your generated comments should be similar to the examples provided, but not identical. "
                    "Format your response as: Generated comment: [your text here]"
                )
            }]
        }
        
        if 'body' not in self.df.columns or 'score' not in self.df.columns:
            raise ValueError("Dataframe must contain 'body' and 'score' columns")

    def _select_examples(self, n, target_score=None):
        """
        Select diverse examples to avoid repetitive patterns.
        """
        # First select candidates based on score as before
        if target_score is not None:
            sampled_values = self.rng.normal(loc=target_score, scale=0.2, size=n)
        else:
            targets = np.linspace(-1, 1, num=n)
            sampled_values = self.rng.normal(targets, 0.2)
            
        sampled_values = np.clip(sampled_values, -1, 1)
        
        scores = self.df['score'].values
        candidates = []
        
        # Get candidates
        for sv in sampled_values:
            distances = np.abs(scores - sv)
            min_idx = np.argmin(distances)
            candidates.append(self.df.iloc[min_idx])
        
        # Now filter for diversity
        selected = []
        used_start_words = set()
        
        for candidate in candidates:
            # Simple diversity check - don't allow same starting word
            first_word = candidate['body'].split()[0].lower() if len(candidate['body'].split()) > 0 else ""
            
            if first_word not in used_start_words and len(selected) < n:
                selected.append(candidate)
                used_start_words.add(first_word)
        
        return selected

    def construct_few_shot_prompt_forward(self, input_comment, n=3):
        examples = self._select_examples(n)
        # TODO: seed
        self.rng.shuffle(examples)
        messages = [self.system_message_forward]
        
        for ex in examples:
            messages.extend([
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"Comment: {ex['body']}"
                    }]
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "text", 
                        "text": f"Toxicity Score: {ex['score']:.2f}"
                    }]
                }
            ])
            
        messages.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Comment: {input_comment}"
            }]
        })
        
        return messages

    def construct_few_shot_prompt_backward(self, target_score, n=3):
        """Génère un prompt avec des exemples proches du score cible"""
        examples = self._select_examples(n, target_score=target_score)
        self.rng.shuffle(examples)
        
        messages = [self.system_message_backward]
        
        for ex in examples:
            messages.extend([
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"Generate a comment with toxicity score: {ex['score']:.2f}"
                    }]
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": f"Generated comment: {ex['body']}"
                    }]
                }
            ])
            
        messages.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Generate a comment with toxicity score: {target_score:.2f}"
            }]
        })
        
        return messages