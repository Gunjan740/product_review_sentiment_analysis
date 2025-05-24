import json
from tabulate import tabulate

def load_data(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def format_data(data, dataset_name):
    formatted = []
    for entry in data:
        review = entry['messages'][0]['content'].split('"')[1]  # Extract review text
        label = entry['label'] if 'label' in entry else entry['messages'][1]['content'].split('—')[0].strip()
        formatted.append([dataset_name, review, label])
    return formatted

def main():
    # Load datasets
    train_data = load_data('ft_data.jsonl')
    test_data = load_data('test_data.jsonl')
    
    # Format data
    train_formatted = format_data(train_data, 'Train')
    test_formatted = format_data(test_data, 'Test')
    
    # Combine and display
    all_data = train_formatted + test_formatted
    
    # Print counts
    print(f"Training data size: {len(train_formatted)} examples")
    print(f"Test data size: {len(test_formatted)} examples")
    print("\n=== Data Distribution ===")
    
    # Calculate distribution
    train_pos = sum(1 for x in train_formatted if x[2] == 'positive')
    train_neg = len(train_formatted) - train_pos
    test_pos = sum(1 for x in test_formatted if x[2] == 'positive')
    test_neg = len(test_formatted) - test_pos
    
    print(f"Train - Positive: {train_pos}, Negative: {train_neg}")
    print(f"Test - Positive: {test_pos}, Negative: {test_neg}")
    print("\n=== Sample Data ===")
    
    # Display first few examples from each set
    print("\n=== First 5 Training Examples ===")
    print(tabulate(train_formatted[:5], headers=['Dataset', 'Review', 'Label'], tablefmt='grid'))
    
    print("\n=== First 5 Test Examples ===")
    print(tabulate(test_formatted[:5], headers=['Dataset', 'Review', 'Label'], tablefmt='grid'))

if __name__ == "__main__":
    main()
