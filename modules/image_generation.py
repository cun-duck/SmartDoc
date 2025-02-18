from huggingface_hub import InferenceClient

def generate_image(prompt, hf_token_image):
    client = InferenceClient(provider="hf-inference", api_key=hf_token_image)
    image = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-dev")
    image.save("outputs/mindmap.png")
    return image