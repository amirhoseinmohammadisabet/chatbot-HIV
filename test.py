import json
from difflib import get_close_matches

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset

def get_response(question, dataset):
    if question in dataset:
        return dataset[question]
    else:
        close_matches = get_close_matches(question, dataset.keys())
        if close_matches:
            closest_match = close_matches[0]
            return dataset[closest_match]
        else:
            return "I'm sorry, I don't have information on that topic."


def main():
    file_path = 'HIV_dataset.json'
    dataset = load_dataset(file_path)
    while True:
        question = input("ask about HIV: ")
        response = get_response(question, dataset)
        print(f"Q: {question}\nA: {response}\n")

if __name__ == "__main__":
    main()
