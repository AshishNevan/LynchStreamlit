import pandas as pd
import os
import random
import nltk
from nltk.corpus import wordnet

# Download wordnet if not present
try:
    nltk.classes
except:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
            
    sentence = ' '.join(new_words)
    return sentence

def augment_dataset(input_file, output_file, num_variations=3):
    df = pd.read_csv(input_file)
    new_rows = []
    
    print(f"Augmenting {len(df)} rows with {num_variations} variations each...")
    
    for _, row in df.iterrows():
        question = row['question']
        answer = row['answer']
        
        # Keep original
        new_rows.append({'question': question, 'answer': answer})
        
        # Generate variations
        for _ in range(num_variations):
            # 1. Synonym Replacement
            aug_q = synonym_replacement(question, n=2)
            new_rows.append({'question': aug_q, 'answer': answer})
            
    new_df = pd.DataFrame(new_rows)
    # Remove duplicates
    new_df = new_df.drop_duplicates()
    
    print(f"New dataset size: {len(new_df)}")
    new_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # Use the clean dataset source
    base_dir = os.path.dirname(__file__)
    target_file = os.path.join(base_dir, '..', 'data', 'processed', 'clean_training_data.csv')
    output_file = os.path.join(base_dir, '..', 'data', 'processed', 'final_augmented_data.csv')
    augment_dataset(target_file, output_file, num_variations=4)
