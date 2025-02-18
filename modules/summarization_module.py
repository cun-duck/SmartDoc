from huggingface_hub import InferenceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter

def summarize_text(text, hf_token_llm):
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    
    client = InferenceClient(provider="hf-inference", api_key=hf_token_llm)
    
  
    summaries = []
    for chunk in chunks:
        messages = [{"role": "user", "content": f"Summarize the text neatly, professionally and Get to the point. :\n{chunk}"}]
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=messages,
            max_tokens=1500
        )
        summaries.append(completion.choices[0].message.content)
    

    full_summary = "\n".join(summaries)
    return full_summary
