import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Add src to path
# Add src to path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from lynchstreamlit.Transformer import transformer, predict, preprocess_sentence, VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, UNITS, DROPOUT

# 1. Instantiate Model and Load Weights
print("Instantiating model and loading weights...")
try:
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, '..', 'models', 'production', 'model_v3.h5')
    
    # New Hyperparameters Matching model_v3
    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    )
    
    model.load_weights(model_path)
    print("Model weights loaded successfully.")
    
except Exception as e:
    print(f"Error loading weights from {model_path}: {e}")
    exit()

# 2. Evaluation on Testing Set
print("Evaluating on testing.csv...")
try:
    test_path = os.path.join(base_dir, '..', 'data', 'testing', 'new_testing.csv')
    test_df = pd.read_csv(test_path)
    test_questions = test_df["question"].dropna().tolist()
    answer_col = "answer" if "answer" in test_df.columns else " answer"
    test_answers = test_df[answer_col].dropna().tolist()
    
    references = []
    candidates = []
    
    for i, question in enumerate(test_questions):
        # Pass model to predict function
        prediction = predict(question, model)
        truth = preprocess_sentence(test_answers[i])
        
        # NLTK BLEU expects reference as a list of lists of tokens
        tokenized_ref = truth.split()
        tokenized_cand = prediction.split()
        
        references.append([tokenized_ref])
        candidates.append(tokenized_cand)
        
        if i < 3: # Print first 3 examples
            print(f"\nQ: {question}")
            print(f"Ref: {truth}")
            print(f"Pred: {prediction}")

    # Calculate BLEU with Smoothing
    smooth = SmoothingFunction().method1
    
    bleu1 = corpus_bleu(references, candidates, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    print("\n" + "="*30)
    print("STANDARD EVALUATION METRICS")
    print("="*30)
    print(f"BLEU-1 Score: {bleu1:.4f}")
    print(f"BLEU-4 Score: {bleu4:.4f}")
    print("="*30)
    print("Methodology: Standard NLTK corpus_bleu calculation.")
    print(f"Test Set Size: {len(references)} samples.")

except Exception as e:
    print(f"Error during evaluation: {e}")
