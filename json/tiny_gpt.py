from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load a smaller model and tokenizer
model_name = "gpt2"  # Use the smaller version of GPT-2
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad_token_id to eos_token_id to avoid warning
tokenizer.pad_token = tokenizer.eos_token

# Example summary input
summary = "John is a 30-year-old software engineer who loves programming in Python."

# Tokenize the input text with attention mask
inputs = tokenizer(summary, return_tensors="pt", max_length=512, truncation=True, padding=True)

# Generate text with sample-based generation settings
output = model.generate(
    input_ids=inputs['input_ids'],  # Input IDs for generation
    attention_mask=inputs['attention_mask'],  # Ensure attention mask is passed
    pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
    do_sample=True,                    # Enable sampling (for using temperature and top_p)
    temperature=0.7,                   # Controls randomness of predictions
    top_p=0.92,                        # Top-p sampling for diversity
    max_length=100                     # Maximum length for the generated text
)

# Decode the generated token IDs to text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Output the generated text
print(f"Generated Text: {generated_text}")
