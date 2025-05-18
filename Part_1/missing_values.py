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
    nr_lipsa = int(0.05 * len(df)) # Consider 5% dintre valori sunt lipsa
    index_random = df.sample(n=nr_lipsa, random_state=42).index
    df.loc[index_random, 'glicemie'] = np.nan
    nr_lipsa = int(0.025 * len(df)) # Consider 2.5% dintre valori sunt lipsa
    index_random = df.sample(n=nr_lipsa, random_state=42).index
    df.loc[index_random, 'fumator'] = np.nan
    df.loc[index_random, 'activitate_fizica'] = np.nan

    ### Afisam procente pentru valorile lipsa
    print2(df)



def fill_missing_values(df): 
    df['glicemie'] = df['glicemie'].fillna(df['glicemie'].mean())
    df['activitate_fizica'] = df['activitate_fizica'].fillna(df['activitate_fizica'].mode()[0])
    df['fumator'] = df['fumator'].fillna(df['fumator'].mean())
