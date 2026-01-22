import tensorflow as tf
import pandas as pd
import re
import numpy as np
import time
import sys
import os

# Add src to path
# Add src to path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from lynchstreamlit.Transformer import transformer, CustomSchedule, loss_function, accuracy
from lynchstreamlit.Transformer import MAX_LENGTH, D_MODEL, NUM_LAYERS, NUM_HEADS, UNITS, DROPOUT, EPOCHS, BUFFER_SIZE, BATCH_SIZE
import tensorflow_datasets as tfds

# 1. Preprocessing and Loading Data
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

print("Loading dataset...")
# Using the larger dataset
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, '..', 'data', 'processed', 'final_augmented_data.csv')
train_df = pd.read_csv(data_path)
questions = train_df["question"].dropna().apply(preprocess_sentence).tolist()
# Handle potential column name variance (leading space)
answer_col = "answer" if "answer" in train_df.columns else " answer"
answers = train_df[answer_col].dropna().apply(preprocess_sentence).tolist()

print(f"Data loaded. {len(questions)} pairs.")

# 2. Tokenizer
print("Building tokenizer...")
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13
)

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2

print(f"Vocab size: {VOCAB_SIZE}")

# 3. Tokenize and Filter
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []
    for (sentence1, sentence2) in zip(inputs, outputs):
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)
    
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
        
    return tokenized_inputs, tokenized_outputs

questions_tokens, answers_tokens = tokenize_and_filter(questions, answers)
print(f"Tokenized samples: {len(questions_tokens)}")

dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions_tokens,
        'dec_inputs': answers_tokens[:, :-1]
    },
    {
        'outputs': answers_tokens[:, 1:]
    }
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# 4. Build Model
print("Building model...")
tf.keras.backend.clear_session()

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT
)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

# 5. Train
print(f"Starting training for {EPOCHS} epochs...")
# Define Checkpoint to save best model based on loss (or just end)
# Since we don't have a distinct val set separate from the 5000 lines (though duplication implies we sort of do), 
# we'll save the final model as model_v2.h5

history = model.fit(
    dataset,
    epochs=EPOCHS
)

print("Training complete.")
model_save_path = os.path.join(base_dir, '..', 'models', 'production', 'model_v3.h5')
model.save_weights(model_save_path)
print(f"Model saved to {model_save_path}")
