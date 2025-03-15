from src.prompt_builder import ToxicityPromptBuilder
from src.data_processor import RudditDataProcessor
from src.models.gemma_3 import Gemma3Model
from src.pipeline import ToxicityPipeline

SEED = 42
N_EXAMPLES = 5

if __name__ == "__main__":
    data_processor = RudditDataProcessor(seed=SEED)
    
    data_processor.load_from_kaggle()
    data_processor.pprint(max_rows=5, max_body_length=150)
    
    data_processor.clean_data()
    data_processor.pprint(max_rows=5, max_body_length=150)
    
    prompt_builder = ToxicityPromptBuilder(data_processor.df, seed=SEED)
    
    gemma = Gemma3Model()
    model, tokenizer = gemma.load()
    
    pipeline = ToxicityPipeline(model, tokenizer, data_processor, prompt_builder)
    pipeline.backward_pass()
