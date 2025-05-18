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
    plt.savefig(f'Heatmap.png')

def violin_plots(df):
    arget = 'risc_diabet'
    cols = ['fumator', 'activitate_fizica', 'glicemie', 'tensiune']
    target = 'risc_diabet'

    for col in cols:
        plt.figure(figsize=(6, 4))
        sns.violinplot(x=target, y=col, data=df)
        plt.title(f'Violin Plot: {col} vs. {target}')
        plt.xlabel('Risc Diabet')
        plt.ylabel(col.capitalize())
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'violin_{col}_vs_{target}.png')  # salvează imaginea


if __name__ == '__main__':
    df = pd.read_csv("train.csv")
    train_df, test_df, y_pred, y_test = train(df)
    Heatmap(train_df)
    violin_plots(df)