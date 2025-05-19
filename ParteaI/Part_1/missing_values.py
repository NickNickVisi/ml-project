import numpy as np
import pandas as pd

def print2(df):
    print()
    print("Valori lipsă pe coloană:")
    missing_count = df.isnull().sum()
    missing_percent = df.isnull().mean() * 100
    missing_df = pd.DataFrame({
    "Valori lipsă": missing_count,
    "Procent (%)": missing_percent.round(2)
    })
    print(missing_df)
    print()

def generate_missing_values(df):
    for col in df.columns:
        procent_lipsa = np.random.uniform(0, 0.05)
        nr_lipsa = int(procent_lipsa * len(df))
        index_random = df.sample(n=nr_lipsa, random_state=42).index
        df.loc[index_random, col] = np.nan

    ### Afisam procente pentru valorile lipsa
    print2(df)

def fill_missing_values(df): 
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].mean())
        elif df[col].dtype in ['object', 'category']:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(np.random.choice([0, 1]))

if __name__ == '__main__':
    df = pd.read_csv("test.csv")
    generate_missing_values(df)
    fill_missing_values(df)
