from transformers import GPT2LMHeadModel, GPT2Tokenizer

def get_model_tokenizer(model_path="./fine-tuned-gpt2", tokenizer_path="gpt2") :
    # Load fine-tuned GPT-2 model and tokenizer
    model_path = "./fine-tuned-gpt2" 
    tokenizer_path = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def test_premodel(model, tokenizer, prompt="The flower with", max_length=100, temperature=1, top_k=1) :

    # Generate text samples
    prompt = "The flower with"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=temperature, top_k=top_k) # top_k=50

    # Decode generated output
    generated_texts = tokenizer.batch_decode(output, skip_special_tokens=True)

    # Print generated texts
    for i, text in enumerate(generated_texts):
        print(f"Generated Text {i+1}: {text}")