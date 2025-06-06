import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(df, subset):
    print("Statistici descriptive:")
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    print(df.describe())
    pd.reset_option('display.float_format')
    df['fumator'] = df['fumator'].astype('category')
    df['risc_diabet'] = df['risc_diabet'].astype('category')

    numeric_cols = df.select_dtypes(include=['float64', 'int']).columns
    # Histograma pentru fiecare caracteristica numerica
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f'Histograma pentru {col}')
        plt.xlabel(col)
        plt.ylabel('Frecventa')
        # Salveaza histograma ca imagine
        plt.savefig(f"images/hist_{col}_{subset}.png")
        plt.show()
        plt.close()
        # Histogramele arata distributia fiecarei variabile numerice.
        # Se pot observa posibile valori atipice.

    # Countplot pentru fiecare caracteristica categorica
    categoric_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categoric_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=df)
        plt.title(f'Countplot pentru {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.savefig(f"images/countplot_{col}_{subset}.png")
        plt.show()
        plt.close()
        # Countplot-urile arata distributia categoriilor.
        # Daca o categorie domina, modelul poate fi dezechilibrat.

    # Detectare outlieri folosind regula IQR pentru fiecare caracteristica numerica
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.2 * IQR
        upper_bound = Q3 + 1.2 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f'Numar de outlieri in {col}: {outliers.shape[0]}')
        # Prezenta outlierilor poate afecta performanta modelului.

    # Matrice de corelatie
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matrice de corelatie')
    plt.savefig(f"images/correlation_matrix_{subset}.png")
    plt.show()
    plt.close()

    # Violin plot pentru variabilele continue
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.violinplot(x='risc_diabet', y=col, data=df)
        plt.title(f'Violin plot pentru {col} vs. risc_diabet')
        plt.xlabel('Risc Diabet')
        plt.ylabel(col)
        plt.savefig(f"images/violin_{col}_{subset}.png")
        plt.show()
        plt.close()
        # Violin plot-urile arata distributia variabilelor numerice in functie de target.
        # In functie de distributie, se observa daca o variabila este relevanta pentru clasificare.
    

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Randurile cu valori lipsa pentru varsta vor deveni media varstei totale
    df['varsta'].fillna(df['varsta'].mean(), inplace=True)

    scaler = StandardScaler()
    df[['varsta', 'greutate', 'inaltime', 'glicemie', 'activitate_fizica', 'tensiune']] = scaler.fit_transform(df[['varsta', 'greutate', 'inaltime', 'glicemie', 'activitate_fizica', 'tensiune']])
    return df
