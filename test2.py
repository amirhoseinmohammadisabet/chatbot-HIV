import json
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression



def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset

    

def train_model_NB(dataset):
    questions = list(dataset.keys())
    responses = list(dataset.values())
    vectorizer = TfidfVectorizer()
    classifier = MultinomialNB()
    model = make_pipeline(vectorizer, classifier)
    model.fit(questions, responses)

    return model


def train_model(dataset):
    questions = list(dataset.keys())
    responses = list(dataset.values())
    vectorizer = TfidfVectorizer()
    chosen_model = input("which model you wish to use: ")
    if chosen_model == 'svm':
        classifier = SVC()
    elif chosen_model == 'NB':
        classifier = MultinomialNB()
    elif chosen_model == 'RF':
        classifier = RandomForestClassifier()
    elif chosen_model == 'DT':
        classifier = DecisionTreeClassifier()
    elif chosen_model == 'knn':
        classifier = KNeighborsClassifier()
    elif chosen_model == 'GB':
        classifier = GradientBoostingClassifier()
    elif chosen_model == 'LR':
        classifier = LogisticRegression()
    else: print('just choose a model from the list')
    model = make_pipeline(vectorizer, classifier)
    model.fit(questions, responses)

    return model



def get_response(question, model):
    prediction = model.predict([question])[0]
    return prediction

def main():
    file_path = 'HIV_dataset.json'
    dataset = load_dataset(file_path)
    model = train_model(dataset)

    while True:
        question = input("ask about HIV: ")
        response = get_response(question, model)
        print(f"Q: {question}\nA: {response}\n")

if __name__ == "__main__":
    main()
