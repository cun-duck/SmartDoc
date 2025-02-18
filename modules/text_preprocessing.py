import re

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Hapus spasi berlebih
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    return text.strip()