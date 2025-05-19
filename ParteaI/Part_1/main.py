import numpy as np
import pandas as pd
from data_load import generate_patient_data
from plot import Heatmap
from print import printf
from model import train
from read_data import read
from missing_values import generate_missing_values
from missing_values import fill_missing_values
from descriptive import description
from descriptive import variable_distribution
from descriptive import boxplot
from descriptive import replace_absurd_values
from plot import violin_plots

print()
print("Do you want to generate a new dataset? (y/n)")
response = input()
if response.lower() == 'y':
    x = read()
    df = generate_patient_data(int (x))

else:
    df = pd.read_csv("train.csv")

# Le facem de tip category
df['fumator'] = df['fumator'].astype('category')
df['risc_diabet'] = df['risc_diabet'].astype('category')
df['activitate_fizica'] = df['activitate_fizica'].astype('category')


generate_missing_values(df)
fill_missing_values(df)
description(df)
variable_distribution(df)
boxplot(df)
replace_absurd_values(df)




train_df, test_df, y_pred, y_test = train(df)
printf(y_test, y_pred)

violin_plots(df)
Heatmap(train_df)