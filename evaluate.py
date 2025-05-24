import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# 1) Setup
load_dotenv()  # assumes your .env has OPENAI_API_KEY
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 2) Load test data
test_file = "test_data.jsonl"
examples = []
with open(test_file, "r") as f:
    for line in f:
        examples.append(json.loads(line))

# 3) Inference loop
results = []
model_id = "ft:gpt-4o-2024-08-06:viscom::BZll9oDU"  # ← put your fine-tuned model ID here

for ex in examples:
    prompt = ex["messages"][0]["content"]
    true_label = ex["label"]
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}]
    )
    # Grab the first word of the assistant’s reply as the prediction
    pred = resp.choices[0].message.content.strip().split()[0].lower()
    results.append({
        "prompt": prompt,
        "true_label": true_label,
        "predicted_label": pred,
        "correct": pred == true_label
    })

# 4) Compute accuracy
correct = sum(r["correct"] for r in results)
total = len(results)
print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")
