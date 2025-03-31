from src.data_processor import RudditDataProcessor
from src.models.m2m100 import M2M100Model
from src.back_translate import back_translate

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run toxicity pipeline with back-translation')
    parser.add_argument('--inter_langs', type=str, default='ja',
                        help='Comma-separated intermediate languages (e.g. "ja,ko,zh")')
    args = parser.parse_args()
    
    data_processor = RudditDataProcessor()
    data_processor.load_from_kaggle()
    data_processor.clean_data()

    m2m100 = M2M100Model()
    model, tokenizer = m2m100.load()
    
    intermediate_langs = args.inter_langs.split(',')
    
    new_df = back_translate(data_processor.df, model, tokenizer, intermediate_langs)
    data_processor.save_to_csv("back_translated_comments.csv", df=new_df)