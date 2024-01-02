# from flask import Flask, render_template, request, jsonify
# import json

# app = Flask(__name__)

# # Load questions and answers from the JSON file
# with open('HIV_dataset.json', 'r') as file:
#     data = json.load(file)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/ask', methods=['POST'])
# def ask():
#     # Get the question from the form
#     user_question = request.form['question']

#     # Logic to find the answer based on the user's question
#     # You can replace this logic with your specific implementation
#     # For example, you can search through the data dictionary for a matching question

#     # For simplicity, let's assume the data dictionary has a key for each question
#     answer = data.get(user_question, "Sorry, I don't have an answer for that.")

#     return render_template('index.html', question=user_question, answer=answer)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request

app = Flask(__name__)

# Load questions and answers from the JSON file (replace with your actual data loading logic)
data = {
    "What is your name?": "My name is ChatGPT.",
    "How does this work?": "This app uses Flask and HTML to answer your questions.",
    # Add more questions and answers as needed
}

@app.route('/', methods=['GET', 'POST'])
def index():
    question = None
    answer = None

    if request.method == 'POST':
        # Get the question from the form
        user_question = request.form['question']

        # Logic to find the answer based on the user's question
        # You can replace this logic with your specific implementation
        # For example, you can search through the data dictionary for a matching question
        answer = data.get(user_question, "Sorry, I don't have an answer for that.")

        # Pass the question and answer back to the template
        question = user_question

    return render_template('index.html', question=question, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
