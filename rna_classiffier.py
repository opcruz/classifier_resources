import os

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

trained_model_url = 'https://raw.githubusercontent.com/opcruz/classifier_resources/main/rna_unbalanced/rna_unbalanced.keras'
model_file_path = 'rna_unbalanced.keras'
dataset_url = 'https://raw.githubusercontent.com/opcruz/classifier_resources/main/rna_unbalanced/test_stars_sy-pn-rg.csv'
dataset_file_path = 'test_stars_sy-pn-rg.csv'


def download_trained_model():
    # Check if the file already exists
    if not os.path.exists(model_file_path):
        response = requests.get(trained_model_url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            with open(model_file_path, 'wb') as file:
                file.write(response.content)
            print('Model downloaded successfully.')
        else:
            print(f'Failed to download the file. Status code: {response.status_code}')


def load_test_dataset():
    # Check if the file already exists
    if not os.path.exists(dataset_file_path):
        response = requests.get(dataset_url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            with open(dataset_file_path, 'wb') as file:
                file.write(response.content)
            print('Dataset downloaded successfully.')
        else:
            print(f'Failed to download the file. Status code: {response.status_code}')

    return pd.read_csv(dataset_file_path)


def print_info():
    # Tabla Data
    data = [
        ["Symbiotic Stars", 'Sy', 0],
        ["Planetary Nebulae", 'PN', 1],
        ["Red Giants", 'RG', 2]
    ]

    labels = ["Stars Types", "Main Type", "Class"]

    for label in labels:
        print(f"{label: <20}", end="")
    print()

    print("-" * (20 * len(labels)))

    for row in data:
        for valor in row:
            print(f"{valor: <20}", end="")
        print()
    print()


if __name__ == "__main__":
    print_info()
    download_trained_model()

    rna_unbalanced_model = tf.keras.models.load_model('rna_unbalanced.keras')
    print("Loaded Model")

    df_test = load_test_dataset()
    print("Loaded Test Dataset")

    # Separate the features (X) and the label (y).
    df_test_features = df_test.iloc[:, :-1]
    df_test_class = df_test.iloc[:, -1]

    # Predict probabilities for test data
    y_probs = rna_unbalanced_model.predict(df_test_features)
    y_pred = np.argmax(y_probs, axis=-1)
    cm = confusion_matrix(df_test_class, y_pred)

    print("\n***** Confusion Matrix *****\n")
    print(cm)

    report = classification_report(df_test_class, y_pred, digits=4)
    print("\n***** Classification Report *****\n")
    print(report)

    print("Accuracy: ", round(accuracy_score(df_test_class, y_pred), 4))
    print("Cohen's kappa: ", round(cohen_kappa_score(df_test_class, y_pred), 4))
    print("AUC-ROC Score:", round(roc_auc_score(df_test_class, y_probs, multi_class='ovr', average='macro'), 4))
