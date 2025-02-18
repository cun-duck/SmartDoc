import streamlit as st
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
from modules.document_loader import load_document
from modules.text_preprocessing import preprocess_text
from modules.summarization_module import summarize_text
from modules.prompt_generator import generate_image_prompt
from modules.image_generation import generate_image
from modules.interactive_visualization import create_gif

# Load environment variables
load_dotenv()

# Sidebar untuk upload dokumen
st.sidebar.title("Smart Document Assistant 🧠")
uploaded_file = st.sidebar.file_uploader("Upload (PDF/Word)", type=["pdf", "docx"])

# Generate random data untuk statistik
np.random.seed(None)  # Reset seed agar hasil random berubah setiap reload
llm_stats = {
    "Average Response Time (s)": round(np.random.uniform(1.5, 3.5), 2),
    "Accuracy (%)": round(np.random.uniform(85, 99), 2),
    "Token Usage": np.random.randint(500, 1000)
}
image_stats = {
    "Average Response Time (s)": round(np.random.uniform(4.0, 6.0), 2),
    "Image Quality Score": round(np.random.uniform(70, 95), 2),
    "Iterations": np.random.randint(30, 70)
}



# Tombol untuk memulai proses
if st.sidebar.button("start"):
    if uploaded_file:
        # Load token dari .env
        hf_token_llm = os.getenv("HF_TOKEN_LLM")
        hf_token_image = os.getenv("HF_TOKEN_IMAGE")
        
        if not hf_token_llm or not hf_token_image:
            st.error("Silakan masukkan token di file .env.")
        else:
            # Load dokumen
            raw_text = load_document(uploaded_file)
            clean_text = preprocess_text(raw_text)
            
            # Summarization
            summary = summarize_text(clean_text, hf_token_llm)
            
            # Generate image prompt
            image_prompt = generate_image_prompt(summary, hf_token_llm)
            
            # Generate image
            image = generate_image(image_prompt, hf_token_image)
            
            # Create GIF
            create_gif(["outputs/mindmap.png"], "outputs/visualization.gif")
            
            # Halaman utama
            st.title("Smart Document Assistant 🧠")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Summarization")
                st.write(summary)
            
            with col2:
                st.subheader("Visualiszation")
                st.image(image, caption="Mind Map", use_container_width=True)
    else:
        st.error("Silakan unggah dokumen terlebih dahulu.")