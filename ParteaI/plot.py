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
    plt.savefig(f'images/Heatmap.png')

def violin_plots(df):
    target = 'risc_diabet'
    cols = [df.columns[i] for i in range(0, 7)]
    target = 'risc_diabet'

    for col in cols:
        plt.figure(figsize=(6, 4))
        sns.violinplot(x=target, y=col, data=df)
        plt.title(f'Violin Plot: {col} vs. {target}')
        plt.xlabel('Risc Diabet')
        plt.ylabel(col.capitalize())
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'images/violin/violin_{col}_vs_{target}.png')  # salvează imaginea

def Heatmap2(train_df):
    corr = train_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Corelații")
    plt.tight_layout()
    plt.savefig(f'images/Heatmap-test.png')

def violin_plots2(df):
    target = 'risc_diabet'
    cols = [df.columns[i] for i in range(0, 7)]
    target = 'risc_diabet'

    for col in cols:
        plt.figure(figsize=(6, 4))
        sns.violinplot(x=target, y=col, data=df)
        plt.title(f'Violin Plot: {col} vs. {target}')
        plt.xlabel('Risc Diabet')
        plt.ylabel(col.capitalize())
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'images/violin/violin_{col}_vs_{target}-test.png')  # salvează imaginea


if __name__ == '__main__':
    df = pd.read_csv("test.csv")
    train_df, test_df, y_pred, y_test = train(df)
    Heatmap(train_df)
    violin_plots(df)