import requests

MODELS = [
    "llama3.1:8b",
    "phi4-mini:3.8b",
    "medgemma"
]

PROMPT = """
### Instruction:
You are a medical cell annotation expert.

Marker genes:
NANOG, SOX2, POU5F1, LIN28

Return ONLY JSON.

### Response:
"""

for model in MODELS:

    print("\n" + "=" * 60)
    print(model)
    print("=" * 60)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": PROMPT,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 256
            }
        }
    )

    print("STATUS:", response.status_code)

    try:
        data = response.json()
        print(data)
    except Exception:
        print(response.text)