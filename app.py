from flask import Flask, render_template, request, jsonify
import models

qa_history = []

app = Flask(__name__)

dataset = models.load_dataset('HIV_dataset.json')
model_dt = models.to_ui(dataset, "DT")
model_rf = models.to_ui(dataset, "RF")

@app.route('/', methods=['GET', 'POST'])
def index():
    user_question = None
    answer = None

    if request.method == 'POST':
        user_question = request.form['question']
        user_question = user_question.lower()
        answer = models.get_response_standard(user_question, dataset)
        qa_history.insert(0, (user_question, answer))

    return render_template('index.html', qa_history=qa_history)

@app.route('/dt', methods=['GET', 'POST'])
def dt():
    user_question = None
    answer = None

    if request.method == 'POST':
        user_question = request.form['question']
        user_question = user_question.lower()
        answer = models.get_response_ML(user_question, model_dt)
        qa_history.insert(0, (user_question, answer))

    return render_template('dt.html', qa_history=qa_history)

@app.route('/rf', methods=['GET', 'POST'])
def rf():
    user_question = None
    answer = None

    if request.method == 'POST':
        user_question = request.form['question']
        user_question = user_question.lower()
        answer = models.get_response_ML(user_question, model_rf)
        qa_history.insert(0, (user_question, answer))

    return render_template('rf.html', qa_history=qa_history)




if __name__ == '__main__':
    app.run(debug=True)

