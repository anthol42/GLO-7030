import numpy as np

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
        self.df = dataframe.copy()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.system_message_forward = {
            "role": "system",
            "content": [{
                "type": "text",
                "text": ("You are a toxicity analysis expert. "
                            "Given a comment, predict its toxicity score on a scale from -1 to 1.")
            }]
        }
        
        self.system_message_backward = {
            "role": "system",
            "content": [{
                "type": "text",
                "text": ("You are an ethical content generator. "
                        "Generate comments that match toxicity scores while strictly adhering to: "
                        "- No harmful language\n"
                        "- No personal attacks\n"
                        "- No discriminatory content\n"
                        "Scores range from -1 (non-toxic) to 1 (toxic). "
                        "Even for high scores, maintain constructive criticism."
                        "Given a toxicity score, generate a comment that matches this score on a scale from -1 to 1 where 1 is very toxic and -1 is not toxic at all.")
            }]
        }
        
        if 'body' not in self.df.columns or 'score' not in self.df.columns:
            raise ValueError("Dataframe must contain 'body' and 'score' columns")


    def _select_examples(self, n):
        """
        Select examples with scores approximately uniformly distributed between -1 and 1.
        
        Args:
            n (int): Number of examples to select
            
        Returns:
            list: Selected examples as pandas Series objects
        """
        targets = np.linspace(-1, 1, num=n)
        selected = []
        candidates = self.df.copy()

        for target in targets:
            # Sample from normal distribution around target
            sampled_value = self.rng.normal(target, 0.15)
            
            # Find closest matches in dataframe
            candidates['distance'] = (candidates['score'] - sampled_value).abs()
            closest_index = candidates['distance'].idxmin()
            
            selected.append(candidates.loc[closest_index])
            candidates = candidates.drop(closest_index)

            # Stop if we run out of candidates
            if candidates.empty:
                break

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
                        "text": f"Comment: {ex['body']}\nWhat is the toxicity score?"
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
                "text": f"Comment: {input_comment}\nWhat is the toxicity score?"
            }]
        })
        
        return messages

    def construct_few_shot_prompt_backward(self, target_score, n=3):
        examples = self._select_examples(n)
        # TODO: seed
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
                        "text": ex['body']
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