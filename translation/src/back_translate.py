from tqdm import tqdm

def back_translate_batch(texts, model, tokenizer, intermediate_langs):
    current_texts = texts
    current_lang = "en"
    
    # Process all intermediate language steps
    for lang in intermediate_langs:
        tokenizer.src_lang = current_lang
        inputs = tokenizer(
            current_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to("cuda")
        generated = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.get_lang_id(lang)
        )
        current_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        current_lang = lang
    
    # Final translation back to English
    tokenizer.src_lang = current_lang
    inputs = tokenizer(
        current_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to("cuda")
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.get_lang_id("en")
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def back_translate(df, model, tokenizer, intermediate_langs):
    new_df = df.copy()
    batch_size = 16  # Adjust based on GPU memory
    batches = [df['body'][i:i+batch_size] for i in range(0, len(df), batch_size)]
    results = []
    
    for batch in tqdm(batches, desc="Back translating"):
        results.extend(back_translate_batch(
            batch.tolist(), 
            model, 
            tokenizer, 
            intermediate_langs
        ))
    
    new_df['translated'] = results
    return new_df
