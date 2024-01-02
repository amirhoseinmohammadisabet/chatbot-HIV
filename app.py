from flask import Flask, render_template, request, jsonify
import models
import json
qa_history = []

app = Flask(__name__)

dataset = models.load_dataset('HIV_dataset.json')

@app.route('/', methods=['GET', 'POST'])
def index():
    question = None
    answer = None

    if request.method == 'POST':
        user_question = request.form['question']
        answer = models.get_response_standard(user_question, dataset)

        qa_history.append((user_question, answer))

    return render_template('index.html', qa_history=qa_history)



if __name__ == '__main__':
    app.run(debug=True)

