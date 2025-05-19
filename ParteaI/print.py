from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from model import train


#Print accuracy and report
def printf(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    df = pd.read_csv("test.csv")
    train_df, test_df, y_pred, y_test = train(df)
    printf(y_test, y_pred)