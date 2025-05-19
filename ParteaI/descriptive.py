import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

 
def description(df):
    print("Statistici descriptive numerice:")
    print(df.describe().T)
    print()

    print("Statistici descriptive categorice:")
    print(df.describe(include=['object', 'category']).T)
    print()

def variable_distribution(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns 
    
    # histograma pentru variabile numerice
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribuția: {col}')
        plt.savefig(f'images/hist/{col}_hist.png')
        plt.close()

    # Countplot pentru categorice
    for col in categorical_cols:
        plt.figure()
        sns.countplot(x=col, data=df)
        plt.title(f'Distribuție categorică: {col}')
        plt.savefig(f'images/hist/{col}_countplot.png')
        plt.close()


def boxplot(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot: {col}')
        plt.savefig(f'images/boxplot/{col}_boxplot.png')
        plt.close()

def variable_distribution2(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns 
    
    # histograma pentru variabile numerice
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribuția: {col}')
        plt.savefig(f'images/hist/{col}_hist-test.png')
        plt.close()

    # Countplot pentru categorice
    for col in categorical_cols:
        plt.figure()
        sns.countplot(x=col, data=df)
        plt.title(f'Distribuție categorică: {col}')
        plt.savefig(f'images/hist/{col}_countplot-test.png')
        plt.close()

def boxplot2(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot: {col}')
        plt.savefig(f'images/boxplot/{col}_boxplot-test.png')
        plt.close()

def replace_absurd_values(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limita_inf = Q1 - 1.2 * IQR
        limita_sup = Q3 + 1.2 * IQR

        mediana = df[col].median()
        df.loc[df[col] < limita_inf, col] = mediana
        df.loc[df[col] > limita_sup, col] = mediana

if __name__ == '__main__':
    df = pd.read_csv("test.csv")
    variable_distribution(df)
    boxplot(df)
    replace_absurd_values(df)