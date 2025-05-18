import numpy as np

def generate_missing_values(df):
    nr_lipsa = int(0.05 * len(df)) # Consider 5% dintre valori sunt lipsa
    index_random = df.sample(n=nr_lipsa, random_state=42).index
    df.loc[index_random, 'glicemie'] = np.nan
    df.loc[index_random, 'fumator'] = np.nan
    df.loc[index_random, 'activitate_fizica'] = np.nan

def fill_missing_values(df): 
    df['glicemie'] = df['glicemie'].fillna(df['glicemie'].mean())
    df['activitate_fizica'] = df['activitate_fizica'].fillna(df['activitate_fizica'].mode()[0])
    df['fumator'] = df['fumator'].fillna(df['fumator'].mean())
