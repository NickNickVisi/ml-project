import pandas as pd
import numpy as np
import random
from model import train

# Generează datele
def generate_patient_data(n):
    data = []
    for i in range(n):
        varsta = np.random.randint(18, 80)
        greutate = np.random.uniform(50, 120)
        inaltime = np.random.uniform(1.5, 2.0)
        fumator = np.random.choice([0, 1])
        zgomot_glicemie = np.random.normal(0, 7.5) 
        activitate_fizica = random.choices(['scazuta', 'medie', 'intensa'], weights=[0.4, 0.4, 0.2])[0]
        glicemie = np.random.normal(100, 30) + zgomot_glicemie
        tensiune = np.random.normal(120, 15)

        risc = 0
        if glicemie > 140 or (fumator == 1 and activitate_fizica == 'scazuta') or tensiune > 150:
            risc = 1

        data.append([
            int(varsta), round(greutate, 1), round(inaltime, 2), fumator,
            activitate_fizica, round(glicemie, 1), round(tensiune, 1), risc
        ])

    columns = ['varsta', 'greutate', 'inaltime', 'fumator',
               'activitate_fizica', 'glicemie', 'tensiune', 'risc_diabet']
               
    
    return pd.DataFrame(data, columns=columns)


if __name__ == '__main__':
    df = generate_patient_data(1400)
    train_df, test_df, y_pred, y_test = train(df)
    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)
