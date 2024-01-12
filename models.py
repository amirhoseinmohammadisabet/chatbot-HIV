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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier



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

def select_ml_algorithm():
    while True:
        print("please select one algorithm from all these options:")
        print("[SVM], [Naive Beyse], [Random Forest], [Decision Tree]")
        print("[KNN], [Gradient Boosting], [Logistic Regression], [ANN]")
        chosen_model = input("which model you wish to use: ").upper()
        if chosen_model == "ANN":
            classifier = MLPClassifier(
                hidden_layer_sizes=(100,),  
                activation='relu',           
                solver='adam',               
                alpha=0.0001,                
                learning_rate='constant',    
                max_iter=200,              
                batch_size='auto'          
            )
            return classifier

        elif chosen_model == 'SVM':
            classifier = SVC(
                C=1.0,
                kernel='rbf', 
                degree=3,     
                gamma='scale'  
            )
            return classifier
        elif chosen_model == 'NB':
            classifier = MultinomialNB(alpha=1.0)
            return classifier
        elif chosen_model == 'RF':
            classifier = RandomForestClassifier(
                n_estimators=100,
                criterion='gini', 
            )
            return classifier
        elif chosen_model == 'DT':
            classifier = DecisionTreeClassifier(
                criterion='gini', 
                max_depth=None,     
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt' 
            )
            return classifier
        elif chosen_model == 'KNN':
            classifier = KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',  
                algorithm='auto',   
                p=2  
            )
            return classifier
        elif chosen_model == 'GB':
            classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                subsample=1.0
            )
            return classifier
        elif chosen_model == 'LR':
            classifier = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                max_iter=100
            )
            return classifier
        else:
            print('just choose a model from the list')

def train_model(dataset):
    questions = list(dataset.keys())
    responses = list(dataset.values())
    vectorizer = TfidfVectorizer()
    classifier = select_ml_algorithm()           
    model = make_pipeline(vectorizer, classifier)
    model.fit(questions, responses)
    return model


def get_response_ML(question, model):
    prediction = model.predict([question])[0]
    return prediction

def to_ui(dataset, ml):
    if ml == 'RF':
        questions = list(dataset.keys())
        responses = list(dataset.values())
        vectorizer = TfidfVectorizer()
        classifier = RandomForestClassifier()          
        model = make_pipeline(vectorizer, classifier)
        x = model.fit(questions, responses)
        return x
    elif ml == 'DT':
        questions = list(dataset.keys())
        responses = list(dataset.values())
        vectorizer = TfidfVectorizer()
        classifier = DecisionTreeClassifier()
        model = make_pipeline(vectorizer, classifier)
        x= model.fit(questions, responses)
        return x



def eval(dataset):
    questions = list(dataset.keys())
    responses = list(dataset.values())

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(questions)
    model = select_ml_algorithm()
    model.fit(X_vectorized, responses)

    y_pred = model.predict(X_vectorized)

    accuracy = accuracy_score(responses, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    precision = precision_score(responses, y_pred, average='weighted',zero_division=1)
    recall = recall_score(responses, y_pred, average='weighted',zero_division=1)
    f1 = f1_score(responses, y_pred, average='weighted')

    print("Other Evaluation Metrics:")
    print(f"Precision: {precision*100:.4f}")
    print(f"Recall: {recall*100:.4f}")
    print(f"F1-Score: {f1*100:.4f}")

def heart(dataset):
    flag = 0
    while True:
        if not flag == 1:
            print("options are [ML] and [standard] and [eval]")
            model_selector = input("which model do you want to use? ").upper()
            if model_selector:
                flag = 1
            else: flag = 0
        else:
            if model_selector == 'ML':
                model = train_model(dataset)
                while True:
                    question = input("ask about HIV: ")
                    question = question.lower()
                    response = get_response_ML(question, model)
                    print(f"Q: {question}\nA: {response}\n")
                    

            elif model_selector == "STANDARD":
                question = input("ask about HIV: ")
                question = question.lower()
                if question == "exit":
                    return
                else:
                    response = get_response_standard(question, dataset)
                    print(f"Q: {question}\nA: {response}\n")
            elif model_selector == "EVAL":
                eval(dataset)
                return
            elif model_selector == 'exit':
                return
            else:
                print("just select from options")
                flag = 0

