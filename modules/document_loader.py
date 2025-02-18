import PyPDF2
from docx import Document

def load_document(file):
    file_name = file.name
    if file_name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file)
        text = ''.join(page.extract_text() for page in reader.pages)
    elif file_name.endswith('.docx'):
        doc = Document(file)
        text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    else:
        raise ValueError("Format dokumen tidak didukung.")
    return text