import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset

def generate_response(question, model, tokenizer):
    input_text = f"Q: {question}\nA:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate a response from the model
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Decode and return the response
    response = tokenizer.decode(output[0], skip_special_tokens=True).replace("A:", "").strip()
    return response

def main():
    # Load your dataset from a JSON file
    file_path = 'HIV_dataset.json'
    dataset = load_dataset(file_path)

    # Load GPT-3.5 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('EleutherAI/gpt-neo-2.7B')
    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')

    # Example questions (you can add more)
    questions = [
        "What is HIV?",
        "How is HIV transmitted?",
        "Can HIV be cured?",
        "Prevention of HIV",
        # Add more questions as needed
    ]

    # Get responses for each question using NLP
    for question in questions:
        response = generate_response(question, model, tokenizer)
        print(f"Q: {question}\nA: {response}\n")

if __name__ == "__main__":
    main()
