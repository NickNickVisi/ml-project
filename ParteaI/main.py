import numpy as np
import pandas as pd
from data_load import generate_patient_data
from plot import Heatmap, Heatmap2
from print import printf
from model import train
from read_data import read
from missing_values import generate_missing_values, fill_missing_values
from descriptive import description, variable_distribution, boxplot, variable_distribution2, boxplot2, replace_absurd_values
from plot import violin_plots, violin_plots2
from type import type

print()
print("Do you want to generate a new dataset? (y/n)")
response = input()
if response.lower() == 'y':
    x = read()
    df = generate_patient_data(int (x))

else:
    df = pd.read_csv("train.csv")

# Le facem de tip category
df = type(df)


# Generam valori lipsa
generate_missing_values(df)
# Inlocuim valorile lipsa
fill_missing_values(df)
# Facem o descriere a datelor
description(df)
variable_distribution(df)
boxplot(df)
# Inlocuim valorile absurde
replace_absurd_values(df)




train_df, test_df, y_pred, y_test = train(df)
# Printam predictiile vs realitate
violin_plots(df)
Heatmap(df)
printf(y_test, y_pred)
test_df = type(test_df)

# Pentru datele de test
# Facem o descriere a datelor
print("Statistici teste:")
print()
# Generam valori lipsa
generate_missing_values(test_df)
# Inlocuim valorile lipsa
fill_missing_values(test_df)
# Facem o descriere a datelor
description(test_df)
variable_distribution2(test_df)
boxplot2(test_df)

# Plotam datele
violin_plots2(test_df)
Heatmap2(test_df)