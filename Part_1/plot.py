import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model import train


# Matrice Corelații
def Heatmap(train_df):
    corr = train_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Corelații")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv("train.csv")
    train_df, test_df, y_pred, y_test = train(df)
    Heatmap(train_df)