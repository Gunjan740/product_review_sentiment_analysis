import json
import random
import pandas as pd

# Define new review templates distinct from training data
positive_reviews = [
    "This device has improved my daily routine beyond belief.",
    "Absolutely stellar performance and reliability.",
    "I’ve recommended this to all my friends; it’s that good!",
    "The craftsmanship is top-tier and it feels very premium.",
    "Exceeded every expectation—I’m impressed."
]

negative_reviews = [
    "Completely unreliable, it failed right out of the box.",
    "The quality control seems non-existent; very disappointed.",
    "Not worth even a single penny; avoid at all costs.",
    "Support was useless and the product is a disaster.",
    "I wish I could get my money back; it’s that bad."
]

# Generate 50 synthetic test examples without overlap
num_examples = 50
data = []
for _ in range(num_examples):
    if random.random() < 0.5:
        review = random.choice(positive_reviews)
        label = "positive"
    else:
        review = random.choice(negative_reviews)
        label = "negative"
    entry = {
        "messages": [
            {"role": "user", "content": f'Review: "{review}"\nSentiment:'}
        ],
        "label": label
    }
    data.append(entry)

# Save to JSONL
output_path = 'test_data.jsonl'
with open(output_path, 'w') as f:
    for entry in data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Display a sample in a DataFrame
df = pd.DataFrame([{"review": e["messages"][0]["content"], "label": e["label"]} for e in data])

print(f"Generated {len(data)} new test examples and saved to {output_path}")
