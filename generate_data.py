import json
import random

# A small pool of positive/negative templates
positive_reviews = [
    "I absolutely love this product! It exceeded all my expectations.",
    "Fantastic quality and super easy to use—totally worth it.",
    "Great value for money; I’m very satisfied with my purchase.",
    "Five stars! It works perfectly and the support team was amazing.",
    "I couldn’t be happier—this product changed my day for the better."
]

negative_reviews = [
    "Terrible experience, it broke on the first use.",
    "Very disappointed—poor quality and not as described.",
    "Waste of money; I want a refund immediately.",
    "One star. The support never responded and it’s unusable.",
    "I regret buying this. It’s flimsy and falls apart quickly."
]

def make_example(review, label):
    user_msg = f"Review: \"{review}\"\nSentiment:"
    # strip leading space since chat 'content' needn’t start with it
    assistant_msg = f"{label} — because {('it’s positive and well-made.' if label=='positive' else 'it’s negative and poorly made.')}"
    return {
        "messages": [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    }

def generate_synthetic_data(n_samples=200, out_file="ft_data.jsonl"):
    with open(out_file, "w") as f:
        for _ in range(n_samples):
            if random.random() < 0.5:
                rev = random.choice(positive_reviews)
                lbl = "positive"
            else:
                rev = random.choice(negative_reviews)
                lbl = "negative"
            entry = make_example(rev, lbl)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Generated {n_samples} chat-style examples into {out_file}")

if __name__ == "__main__":
    generate_synthetic_data()
