from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def train_models(df):

    # Separarea caracteristicilor si a etichetei
    X_train = df.drop(columns=['fumator', 'risc_diabet'])
    y_train = df['risc_diabet']
    # Antrenarea diferitelor modele
    models = {
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} model trained.")
    return models

def evaluate_models(models, df):
    # Separarea caracteristicilor si a etichetei
    X_test = df.drop(columns=['fumator', 'risc_diabet'])
    y_test = df['risc_diabet']
    accuracy_arr = []
    f1_arr = []
    roc_auc_arr = []
    # Evaluarea modelelor
    print("-" * 30)
    for name, model in models.items():
        accuracy = model.score(X_test, y_test)
        print(f"{name} model accuracy: {accuracy:.2f}")
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        print(f"{name} model F1 score: {f1:.2f}")
        roc_auc = roc_auc_score(y_test, y_pred)
        print(f"{name} model ROC AUC score: {roc_auc:.2f}")
        print("-" * 30)
        accuracy_arr.append(accuracy)
        f1_arr.append(f1)
        roc_auc_arr.append(roc_auc)
    return accuracy_arr, f1_arr, roc_auc_arr

def comparative_table(models, accuracy_arr, f1_arr, roc_auc_arr):
    results = []
    for idx, name in enumerate(models.keys()):
        results.append({
            "Algorithm": name,
            "Accuracy": accuracy_arr[idx],
            "F1 Score": f1_arr[idx],
            "ROC AUC": roc_auc_arr[idx]
        })
    table = pd.DataFrame(results)
    print("\nComparative Table:")
    print(table)
    return table

def plot_confusion_matrices(models, df):
    X = df.drop(columns=['fumator', 'risc_diabet'])
    y = df['risc_diabet']
    for name, model in models.items():
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matricea de confuzie - {name}')
        plt.xlabel('Predictie')
        plt.ylabel('Realitate')
        plt.savefig(f"images/confusion_matrix_{name}.png")
        plt.show()
