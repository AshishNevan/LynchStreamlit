# Peter Lynch Financial AI: The "Tenbagger" Transformer

A custom-built Transformer model trained to think and speak like legendary investor Peter Lynch. This AI generates investment wisdom in Lynch's signature style, answering questions about market strategy, "tenbaggers," and fundamental analysis.

## 🚀 Project Overview

**[🔴 Live Demo: Chat with Peter Lynch](https://ashishnevan-lynchstreamlit-app-cw5l66.streamlit.app/)**

This project implements a **sequence-to-sequence Transformer** model from scratch using TensorFlow/Keras. It is designed to learn the semantic relationships between financial questions and Peter Lynch's investment philosophy, acting as a domain-specific conversational agent.

### Technical Highlights
-   **Architecture**: Full Encoder-Decoder Transformer (Vaswani et al., 2017).
-   **Custom Implementation**: built `MultiHeadAttention`, `PositionalEncoding`, and masking layers from scratch (no pre-made Hugging Face heads).
-   **Framework**: TensorFlow 2.x (Keras).
-   **Tokenizer**: Subword Text Encoder (approx. 3.3k vocabulary).

---

## 📈 The Optimization Journey: From Overfitting to Generalization

A key achievement of this project was diagnosing and solving a severe data leakage and overfitting problem, transitioning the model from a "parrot" to a "student."

### Phase 1: Overcoming Data Scarcity (Initial Baselines)
The initial dataset consisted of only 100 Q&A pairs. Training a Transformer on such a small corpus led to immediate overfitting, where the model essentially memorized the training data but failed to generalize to new inputs. We needed to transition the model from specific memorization to generalized learning.

### Phase 2: The Correction (Regularization)
We implemented a robust fix pipeline:
1.  **Architecture Tuning**: Reduced `d_model` (256 &rarr; 128) and increased Dropout (0.1) to penalize memorization.
2.  **Result**: The model stopped exact-matching (Training Accuracy dropped 1.0 &rarr; 0.59), but still "hallucinated" because it lacked enough examples to learn language rules.

### Phase 3: Scaling & Augmentation (Success)
We realized 100 examples were insufficient for deep learning. We built a data pipeline to solve this:
-   **Data Cleaning**: Extracted 925 unique pairs from raw logs (`clean_data.py`).
-   **Augmentation**: Built an NLTK-based synonym replacement engine (`augment_data.py`) to generate 5 variations of every question.
-   **Final Dataset**: **4,445 high-quality samples**.

### 🏆 Final Results (Model v3)

| Metric | Model v3 (Final) |
| :--- | :--- |
| **Dataset Size** | **4,445 (Unique)** |
| **BLEU-4 Score** | **0.041** |
| **Qualitative Result** | **Generalization** |

**Example of Generalization:**
> **Q:** "What is a tenbagger?"
> **Ground Truth:** "...increases in value by ten times..."
> **Model Output:** *"A tenbagger is a stock that **grows tenfold in value**..."*
> *(The model learned the concept of "tenfold" rather than just copying "ten times".)*

---

## 🛠 Project Structure

The project follows a standard MLOps structure:

```
/
├── src/               # Core source code
│   ├── Transformer.py # Model architecture definition
│   └── lynchapp.py    # Streamlit application
├── scripts/           # Execution pipelines
│   ├── train_model.py      # Training script
│   ├── evaluate_metrics.py # Evaluation script
│   └── augment_data.py     # Data augmentation script
├── data/
│   ├── raw/           # Original data sources
│   ├── processed/     # Cleaned and augmented datasets for training
│   └── testing/       # Held-out test sets
├── models/
│   ├── production/    # Final deployed model (model_v3.h5)
│   └── archive/       # Model history
├── notebooks/         # Exploratory Jupyter notebooks
└── assets/            # Images and static files
```

## 💿 Installation & Reproducibility (Mac/Linux/Windows)

This project uses **[uv](https://github.com/astral-sh/uv)** for precise dependency management and reproducibility.

1.  **Install uv** (if not installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone & Sync**:
    ```bash
    git clone https://github.com/yourusername/LynchStreamlit.git
    cd LynchStreamlit
    uv sync
    ```

3.  **Run the App**:
    ```bash
    uv run streamlit run app.py
    ```

## 💻 Usage

### 1. Augment Data (Optional)
If you have new data in `data/processed/clean_training_data.csv`:
```bash
python scripts/augment_data.py
# Generates data/processed/final_augmented_data.csv
```

### 2. Train the Model
```bash
python scripts/train_model.py
# Saves weights to models/production/model_v3.h5
```

### 3. Evaluate Performance
```bash
python scripts/evaluate_metrics.py
# Reports BLEU scores on data/testing/new_testing.csv
```

### 4. Run the App
```bash
streamlit run src/lynchapp.py
```
