import matplotlib.pyplot as plt
import seaborn as sns

# Matrice Corelații
def Heatmap(train_df):
    corr = train_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Corelații")
    plt.tight_layout()
    plt.show()