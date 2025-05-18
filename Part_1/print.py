from sklearn.metrics import accuracy_score, classification_report

#Print accuracy and report
def printf(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))