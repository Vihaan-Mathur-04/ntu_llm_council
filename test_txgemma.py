from model_manager import ModelManager

tx = ModelManager.get_txgemma()

tokenizer = tx["tokenizer"]
model = tx["model"]

prompt = "What genes are associated with T cells in PBMC data?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\n--- TXGEMMA OUTPUT ---\n")
print(response)