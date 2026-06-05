import torch
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "google/medgemma-4b-it"

processor = AutoProcessor.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

messages = [
    {
        "role": "user",
        "content": "Explain what PBMC cells are in immunology."
    }
]

# STEP 1: apply chat template (returns STRING)
chat_text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True
)

# STEP 2: convert STRING → TENSORS
inputs = processor(
    text=chat_text,
    return_tensors="pt"
)

# move to device (this now works)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

response = processor.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\n--- OUTPUT ---\n")
print(response)