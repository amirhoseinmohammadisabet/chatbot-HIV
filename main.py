import json
from difflib import get_close_matches
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset

def get_response(question, dataset):
    # Check for an exact match
    if question in dataset:
        return dataset[question]
    else:
        # Use difflib to find close matches
        close_matches = get_close_matches(question, dataset.keys())
        if close_matches:
            closest_match = close_matches[0]
            return dataset[closest_match]
        else:
            return "I'm sorry, I don't have information on that topic."

def generate_response_GPT(question, model, tokenizer):
    input_text = f"Q: {question}\nA:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True).replace("A:", "").strip()
    return response



def main():
    # Load your dataset from a JSON file
    file_path = 'HIV_dataset.json'
    dataset = load_dataset(file_path)

    # Example questions (you can add more)

    model = GPT2LMHeadModel.from_pretrained('EleutherAI/gpt-neo-2.7B')
    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
    flag = 0
    while True:
        # questions
        #     "What is HIV?",
        #     "How is HIV transmitted?",
        #     "Can HIV be cured?",
        #     "Prevention of HIV",
        if not flag == 1:
            model_selector = input("which model do you want to use? ")
            flag = 1
        else:
            if model_selector == 'gpt':
                question = input("ask about HIV: ")
                response = generate_response_GPT(question, model, tokenizer)
                print(f"Q: {question}\nA: {response}\n")
            elif model_selector == "standard":
                question = input("ask about HIV: ")
                response = get_response(question, dataset)
                print(f"Q: {question}\nA: {response}\n")
        # response = get_response(question, dataset)
        # print(f"Q: {question}\nA: {response}\n")


if __name__ == "__main__":
    main()