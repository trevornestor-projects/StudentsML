# Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load Data
dataset = pd.read_csv("/Data.csv")

# Function to Display Graphs
def display_graphs():
    graph_options = {
        1: ('Class', 'Class Count Graph'),
        2: ('Semester', 'Semester-wise Class Graph'),
        3: ('gender', 'Gender-wise Class Graph'),
        4: ('NationalITy', 'Nationality-wise Class Graph'),
        5: ('GradeID', 'Grade-wise Class Graph'),
        6: ('SectionID', 'Section-wise Class Graph'),
        7: ('Topic', 'Topic-wise Class Graph'),
        8: ('StageID', 'Stage-wise Class Graph'),
        9: ('StudentAbsenceDays', 'Absent Days-wise Class Graph')
    }

    while True:
        print("\nSelect a Graph to Display:")
        for key, value in graph_options.items():
            print(f"{key}. {value[1]}")
        print("10. Exit\n")

        choice = int(input("Enter your choice (1-10): "))
        if choice == 10:
            break
        elif choice in graph_options:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=graph_options[choice][0], hue='Class', data=dataset, order=['L', 'M', 'H'])
            plt.title(graph_options[choice][1])
            plt.show()

# Preprocess Data
def preprocess_data(data):
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == type(object):
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le
    return data, label_encoders

# Train Models
def train_models(X_train, y_train):
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Perceptron": Perceptron(),
        "Logistic Regression": LogisticRegression(),
        "MLP Classifier": MLPClassifier(activation="logistic")
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        models[name] = model
    return models

# Evaluate Models
def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        predictions = model.predict(X_test)
        print(f"\n{name} Evaluation:")
        print(classification_report(y_test, predictions))

# Main Execution
if __name__ == "__main__":
    display_graphs()

    # Data Preprocessing
    processed_data, encoders = preprocess_data(dataset)
    features = processed_data.drop('Class', axis=1)
    labels = processed_data['Class']

    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Training Models
    trained_models = train_models(X_train, y_train)

    # Evaluating Models
    evaluate_models(trained_models, X_test, y_test)

    # Additional User Interaction for Prediction (if required)
