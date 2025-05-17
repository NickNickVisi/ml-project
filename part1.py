from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import random

np.random.seed(42)

def generate_patient_data(n):
    data = []

    for _ in range(n):
        varsta = np.random.randint(18, 80)
        greutate = np.random.uniform(50, 120)
        inaltime = np.random.uniform(1.5, 2.0)
        sex = random.choice(['masculin', 'feminin'])
        fumator = np.random.choice([0, 1])
        activitate_fizica = random.choices(['scazuta', 'medie', 'intensa'], weights=[0.4, 0.4, 0.2])[0]
        glicemie = np.random.normal(100, 30)
        tensiune = np.random.normal(125, 15)

        risc = 0
        if glicemie > 140 or (fumator == 1 and activitate_fizica == 'scazuta') or tensiune > 150:
            risc = 1

        data.append([
            int(varsta), round(greutate, 1), round(inaltime, 2), sex, fumator,
            activitate_fizica, round(glicemie, 1), round(tensiune, 1), risc
        ])

    columns = ['varsta', 'greutate', 'inaltime', 'sex', 'fumator', 'activitate_fizica', 'glicemie', 'tensiune', 'risc']
    return pd.DataFrame(data, columns=columns)

df = generate_patient_data(800)

train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Datele au fost salvate cu succes.")
numeric_cols = ['varsta', 'greutate', 'inaltime', 'glicemie', 'tensiune']

corr = train_df[numeric_cols + ['risc']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Matrice de corelații")
plt.show()
