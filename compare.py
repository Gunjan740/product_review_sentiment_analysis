import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm  # pip install tqdm

# 1) Setup
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2) Load test examples
with open("test_data.jsonl") as f:
    examples = [json.loads(line) for line in f]

# 3) Models to compare
models = {
    "base":       "gpt-4o-2024-08-06",   # replace if yours differs
    "fine_tuned": "ft:gpt-4o-2024-08-06:viscom::BZll9oDU"
}

# 4) Run evaluation
results = {}
for name, model_id in models.items():
    correct = 0
    print(f"\n▶ Evaluating {name} ({model_id})")
    for ex in tqdm(examples, desc=name, unit="ex"):
        # Base model: force deterministic single‐token output
        if name == "base":
            resp = client.chat.completions.create(
                model=model_id,
                messages=ex["messages"],
                temperature=0,
                top_p=1,
                max_tokens=1
            )
        else:
            # Fine-tuned: default sampling is fine
            resp = client.chat.completions.create(
                model=model_id,
                messages=ex["messages"]
            )

        pred = resp.choices[0].message.content.strip().split()[0].lower()
        if pred == ex["label"]:
            correct += 1

    accuracy = correct / len(examples)
    results[name] = accuracy
    print(f"  ↳ {name} accuracy: {correct}/{len(examples)} = {accuracy:.2%}")

# 5) Summary
print("\n=== Comparison ===")
for name, acc in results.items():
    print(f"• {name.ljust(11)} → {acc:.2%}")
