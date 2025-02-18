from huggingface_hub import InferenceClient

def generate_image_prompt(summary, hf_token_llm):
    client = InferenceClient(provider="hf-inference", api_key=hf_token_llm)
    messages = [{
        "role": "user",
        "content": f"Create a text to image prompt to create a book or document cover based on the following information.:\n{summary} : , and add this world to the prompt A 3D-rendered cover, hyperrealistic, balanced composition, against a backdrop of a vibrant,boundless imagination  "
    }]
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=messages,
        max_tokens=100
    )
    return completion.choices[0].message.content