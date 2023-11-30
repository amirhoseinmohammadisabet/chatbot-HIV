import json
from difflib import get_close_matches
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer




def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset

def get_response_standard(question, dataset):
    if question in dataset:
        return dataset[question]
    else:
        close_matches = get_close_matches(question, dataset.keys())
        if close_matches:
            closest_match = close_matches[0]
            return dataset[closest_match]
        else:
            return "I'm sorry, I don't have information on that topic."


def train_model(dataset):
    questions = list(dataset.keys())
    responses = list(dataset.values())
    vectorizer = TfidfVectorizer()
    while True:
        chosen_model = input("which model you wish to use: ")
        if chosen_model == 'svm':
            classifier = SVC()
            break
        elif chosen_model == 'NB':
            classifier = MultinomialNB()
            break
        elif chosen_model == 'RF':
            classifier = RandomForestClassifier()
            break
        elif chosen_model == 'DT':
            classifier = DecisionTreeClassifier()
            break
        elif chosen_model == 'knn':
            classifier = KNeighborsClassifier()
            break
        elif chosen_model == 'GB':
            classifier = GradientBoostingClassifier()
            break
        elif chosen_model == 'LR':
            classifier = LogisticRegression()
            break
        else: print('just choose a model from the list')
    model = make_pipeline(vectorizer, classifier)
    model.fit(questions, responses)
    return model


def get_response_ML(question, model):
    prediction = model.predict([question])[0]
    return prediction


def main():
    # Load your dataset from a JSON file
    file_path = 'HIV_dataset.json'
    dataset = load_dataset(file_path)


    flag = 0
    while True:
        if not flag == 1:
            model_selector = input("which model do you want to use? ")
            if model_selector:
                flag = 1
            else: flag = 0
        else:
            if model_selector == 'ML':
                model = train_model(dataset)
                question = input("ask about HIV: ")
                response = get_response_ML(question, model)
                print(f"Q: {question}\nA: {response}\n")
            elif model_selector == "standard":
                question = input("ask about HIV: ")
                if question == "exit":
                    return
                else:
                    response = get_response_standard(question, dataset)
                    print(f"Q: {question}\nA: {response}\n")
            elif model_selector == 'exit':
                return
            else: pass




if __name__ == "__main__":
    main()
