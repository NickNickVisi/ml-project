# data_utils.py

import pandas as pd
import numpy as np
import random

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

    columns = ['varsta', 'greutate', 'inaltime', 'sex', 'fumator',
               'activitate_fizica', 'glicemie', 'tensiune', 'risc_diabet']
    return pd.DataFrame(data, columns=columns)

if __name__ == "__main__":
    df = generate_patient_data(10)
    print(df.head())
