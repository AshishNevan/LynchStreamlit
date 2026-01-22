import pandas as pd
import os

base_dir = os.path.dirname(__file__)
# Load the messy, large file from raw data
input_file = os.path.join(base_dir, "..", "data", "raw", "navigator_ft-6748e9cd1cdbe50f6a3f0afd-data.csv")
output_file = os.path.join(base_dir, "..", "data", "processed", "clean_training_data.csv")

print(f"Reading {input_file}...")
try:
    df = pd.read_csv(input_file)
    print(f"Original row count: {len(df)}")
    
    # Check for mixed column names (some files had " answer" vs "answer")
    if " answer" in df.columns:
        df = df.rename(columns={" answer": "answer"})
    
    # Drop duplicates
    df_clean = df.drop_duplicates(subset=["question", "answer"])
    
    # Drop NaNs just in case
    df_clean = df_clean.dropna()
    
    print(f"Unique row count: {len(df_clean)}")
    
    df_clean.to_csv(output_file, index=False)
    print(f"Saved clean data to {output_file}")
    
except Exception as e:
    print(f"Error: {e}")
