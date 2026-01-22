import streamlit as st
import os
from lynchstreamlit.Transformer import transformer, predict, VOCAB_SIZE, NUM_LAYERS, UNITS, D_MODEL, NUM_HEADS, DROPOUT
import tensorflow as tf

base_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(base_dir, "models", "production", "model_v3.h5")
@st.cache_resource
def load_model():
    # Build model architecture
    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    )

    # Load weights only
    try:
        model.load_weights(filename)
        print("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None

model = load_model()

# Streamlit UI
bg_path = os.path.join(base_dir, "assets", "background.png")
if os.path.exists(bg_path):
    st.image(bg_path, use_container_width=True)

st.markdown(
    """
    <h1 style="color: darkgreen; text-align: center; font-size: 45px; font-weight: bold;">
        I am Peter Lynch! Ask me anything
    </h1>
    """,
    unsafe_allow_html=True,
)

user_input = st.text_input("Your question:")
if user_input:
    st.markdown(f"**You asked:** {user_input}")
    try:
        # Use the predict function from Transformer.py
        answer = predict(user_input, model)
        st.markdown(f"**Peter Lynch says:** {answer}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
