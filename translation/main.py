from src.data_processor import RudditDataProcessor
from src.models.m2m100 import M2M100Model
from src.back_translate import back_translate

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Toxicity Pipeline with specified model and mode.')
    parser.add_argument('--inter_lang', type=str, default='ja',
                        help='Intermediate language for back-translation. Default is "ja".')
    args = parser.parse_args()
    
    data_processor = RudditDataProcessor()
    
    data_processor.load_from_kaggle()
    data_processor.clean_data()

    m2m100 = M2M100Model()
    model, tokenizer = m2m100.load()
    
    df = data_processor.df
    
    new_df = back_translate(df, model, tokenizer, args.inter_lang)
    data_processor.save_to_csv("back_translated_comments.csv", df=new_df)
